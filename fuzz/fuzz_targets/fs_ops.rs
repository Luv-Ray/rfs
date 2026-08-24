#![no_main]
//! Differential fuzz of the `Fs` library layer against an in-memory oracle.
//!
//! A random byte stream is decoded into a sequence of filesystem operations
//! (write / unlink / sync / gc / snapshot / reopen). Each op runs against a
//! real image-backed `Fs` and, in parallel, against a `HashMap` shadow model
//! of the *durable* extent contents. After every mutation the two are checked
//! for agreement, and `gc` / `reopen` are checked to preserve that agreement —
//! so this catches GC reclaiming a live block, journal replay losing a write,
//! or a snapshot changing what the active subvolume sees, not merely panics.
//!
//! Every write is `journal_commit`ed immediately, so the shadow model always
//! equals what a crash-recovery `open` must reconstruct: the oracle invariant
//! is simply "committed extents == shadow", checked continuously and again
//! after each reopen.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;

use rfs::block_btree::BLOCK_SIZE;
use rfs::fs::{DirentV1, FILE_KIND_REGULAR, Fs, InodeV1, ROOT_INO, ROOT_SUBVOL};

const N_FILES: u64 = 5; // inodes 2..7
const N_OFFSETS: u64 = 4; // block offsets 0..4
const MAX_OPS: usize = 256;
const BLOCK: u64 = BLOCK_SIZE as u64;

#[derive(Arbitrary, Debug)]
enum Op {
    /// Write a single block of `fill`-bytes (length 1..=BLOCK) at slot.
    Write { file: u8, off: u8, fill: u8, len: u16 },
    /// Unlink a file: tombstone dirent+inode, then run reclaim.
    Unlink { file: u8 },
    /// Checkpoint + persist superblock.
    Sync,
    /// Mark-and-sweep GC.
    Gc,
    /// Reclaim a slice of deleted-inode work items.
    Reclaim,
    /// Snapshot the root subvolume (does not switch the active view).
    Snapshot,
    /// Close and reopen the image, forcing journal-replay recovery.
    Reopen,
}

/// One occupied block: the fill byte and its length. `read_data_block` returns
/// a full block, so we check the first `len` bytes are `fill` and the rest is
/// zero-padded (put_extent zero-fills the tail).
#[derive(Clone, Copy)]
struct Cell {
    fill: u8,
    len: usize,
}

fn ino_of(file: u8) -> u64 {
    2 + (file as u64 % N_FILES)
}

fn name_of(file: u8) -> [u8; 1] {
    [b'a' + (file % N_FILES as u8)]
}

fn offset_of(off: u8) -> u64 {
    (off as u64 % N_OFFSETS) * BLOCK
}

/// Build a fresh image with the root dir and N_FILES empty regular files.
fn fresh(path: &std::path::Path) -> Fs {
    let mut fs = Fs::create(path).expect("create image");
    // Root inode + self dirent, mirroring the test helpers.
    let root = InodeV1 {
        mode: 0o040755,
        uid: 0,
        gid: 0,
        nlink: 2,
        size: 0,
        atime: 0,
        mtime: 0,
        ctime: 0,
        parent_ino: ROOT_INO,
    };
    fs.put_inode(ROOT_INO, &root).expect("put root inode");
    for file in 0..N_FILES as u8 {
        let ino = ino_of(file);
        let inode = InodeV1 {
            mode: 0o100644,
            uid: 0,
            gid: 0,
            nlink: 1,
            size: 0,
            atime: 0,
            mtime: 0,
            ctime: 0,
            parent_ino: ROOT_INO,
        };
        fs.put_inode(ino, &inode).expect("put inode");
        fs.put_dirent(ROOT_INO, &name_of(file), &DirentV1::new(ino, FILE_KIND_REGULAR))
            .expect("put dirent");
    }
    fs.journal_commit().expect("initial commit");
    fs
}

/// Assert every shadow cell matches the fs, and the fs holds nothing extra for
/// the tracked inodes. Only valid while the active subvol is the original one.
fn check_agreement(fs: &Fs, shadow: &HashMap<(u64, u64), Cell>, live: &[bool]) {
    for file in 0..N_FILES as u8 {
        if !live[file as usize] {
            continue;
        }
        let ino = ino_of(file);
        for off_idx in 0..N_OFFSETS as u8 {
            let off = offset_of(off_idx);
            let got = fs.get_extent(ino, off).expect("get_extent");
            match (shadow.get(&(ino, off)), got) {
                (Some(cell), Some(ext)) => {
                    assert_eq!(
                        ext.len as usize, cell.len,
                        "len mismatch ino={ino} off={off}"
                    );
                    let block = fs.read_data_block(ext.data_block).expect("read block");
                    for (i, b) in block.iter().enumerate() {
                        let want = if i < cell.len { cell.fill } else { 0 };
                        assert_eq!(
                            *b, want,
                            "byte {i} mismatch ino={ino} off={off}: got {b} want {want}"
                        );
                    }
                }
                (None, None) => {}
                (Some(_), None) => panic!("shadow has extent but fs lost it: ino={ino} off={off}"),
                (None, Some(_)) => panic!("fs has extent shadow never wrote: ino={ino} off={off}"),
            }
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("fuzz.img");
    let mut fs = fresh(&path);

    // Shadow of durable extent contents, and which inodes are still linked.
    let mut shadow: HashMap<(u64, u64), Cell> = HashMap::new();
    let mut live = [true; N_FILES as usize];
    // We never switch the active subvolume, so the shadow of the root
    // subvol's durable extents stays authoritative for the whole run —
    // snapshotting must not change what the active view reads.

    let mut u = arbitrary::Unstructured::new(data);
    let mut ops = 0;
    while ops < MAX_OPS && !u.is_empty() {
        let Ok(op) = Op::arbitrary(&mut u) else { break };
        ops += 1;
        match op {
            Op::Write { file, off, fill, len } => {
                if !live[(file % N_FILES as u8) as usize] {
                    continue;
                }
                let ino = ino_of(file);
                let offset = offset_of(off);
                let len = (len as usize % BLOCK_SIZE) + 1; // 1..=BLOCK
                let payload = vec![fill; len];
                fs.put_extent(ino, offset, &payload).expect("put_extent");
                fs.journal_commit().expect("commit write");
                shadow.insert((ino, offset), Cell { fill, len });
            }
            Op::Unlink { file } => {
                let slot = (file % N_FILES as u8) as usize;
                if !live[slot] {
                    continue;
                }
                // Unlink may legitimately fail (e.g. mid-state); tolerate it.
                if fs.unlink(ROOT_INO, &name_of(file)).is_ok() {
                    fs.journal_commit().expect("commit unlink");
                    // Reclaim drops the extents; drop them from the shadow too.
                    let _ = fs.reclaim_deleted_inodes(usize::MAX);
                    fs.journal_commit().expect("commit reclaim");
                    let ino = ino_of(file);
                    shadow.retain(|&(i, _), _| i != ino);
                    live[slot] = false;
                }
            }
            Op::Sync => {
                fs.sync().expect("sync");
            }
            Op::Gc => {
                fs.gc().expect("gc");
            }
            Op::Reclaim => {
                let _ = fs.reclaim_deleted_inodes(usize::MAX);
                fs.journal_commit().expect("commit reclaim");
            }
            Op::Snapshot => {
                // May exhaust the snap-id space over a long run; that's fine.
                let _ = fs.snapshot_subvol(ROOT_SUBVOL);
                fs.journal_commit().expect("commit snapshot");
            }
            Op::Reopen => {
                // Ensure the current view is durable, then reopen from disk.
                fs.sync().expect("sync before reopen");
                drop(fs);
                fs = Fs::open(&path).expect("reopen image");
            }
        }
        check_agreement(&fs, &shadow, &live);
    }

    // Final gauntlet: gc then reopen must both preserve the durable state.
    fs.gc().expect("final gc");
    check_agreement(&fs, &shadow, &live);
    fs.sync().expect("final sync");
    drop(fs);
    let fs = Fs::open(&path).expect("final reopen");
    check_agreement(&fs, &shadow, &live);
});
