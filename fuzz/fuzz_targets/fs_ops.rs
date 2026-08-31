#![no_main]
//! Differential fuzz of the `Fs` library layer against an in-memory oracle.
//!
//! A random byte stream is decoded into a sequence of filesystem operations
//! (write / unlink / sync / gc / reclaim / snapshot / delete-snapshot /
//! reopen). Each op runs against a real image-backed `Fs` and, in parallel,
//! against `HashMap` shadow models — one for the active subvolume and one
//! frozen copy per readonly snapshot. After every mutation both the active
//! view and every live snapshot are checked for agreement, so this catches:
//!   - GC reclaiming a live block,
//!   - journal replay losing a write,
//!   - and (the class of the reclaim bug we fixed) unlink/reclaim/gc freeing
//!     a COW-shared block that a sibling readonly snapshot still references.
//!
//! Snapshot oracle: `snapshot_subvol` bumps the active subvol to a fresh
//! writable snap_id and creates a readonly snapshot at a sibling snap_id.
//! Subsequent active writes land on the writable snap_id and never touch the
//! ancestor snap_ids the readonly view depends on, so a readonly snapshot's
//! content is exactly the active content *frozen at creation time* and must
//! stay byte-identical forever. We model that by cloning the active shadow at
//! snapshot time and re-verifying it against the snapshot subvol after every
//! later op. Writes always target the root subvol; snapshots are read-only.
//!
//! Every write is `journal_commit`ed immediately, so the shadow equals what a
//! crash-recovery `open` must reconstruct: the invariant is "committed extents
//! == shadow", checked continuously and again after each reopen.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;

use rfs::block_btree::BLOCK_SIZE;
use rfs::fs::{DirentV1, FILE_KIND_REGULAR, Fs, InodeV1, ROOT_INO, ROOT_SUBVOL, SubvolId};
use rfs::fuse::do_read;

const N_FILES: u64 = 5; // inodes 2..7
const N_OFFSETS: u64 = 4; // block offsets 0..4
const MAX_OPS: usize = 256;
const MAX_SNAPS: usize = 4; // cap tracked snapshots to bound verification cost
const BLOCK: u64 = BLOCK_SIZE as u64;

type Shadow = HashMap<(u64, u64), Cell>;

#[derive(Arbitrary, Debug)]
enum Op {
    /// Write a single block of `fill`-bytes (length 1..=BLOCK) at slot.
    Write { file: u8, off: u8, fill: u8, len: u16 },
    /// Read an arbitrary byte range via `do_read` and diff against the oracle.
    /// `off`/`len` are intentionally unconstrained (non-block-aligned, may
    /// straddle blocks and run past EOF) to exercise the read path's scan
    /// window and zero-fill.
    Read { file: u8, off: u16, len: u16 },
    /// Unlink a file: tombstone dirent+inode, then run reclaim.
    Unlink { file: u8 },
    /// Checkpoint + persist superblock.
    Sync,
    /// Mark-and-sweep GC.
    Gc,
    /// Reclaim a slice of deleted-inode work items.
    Reclaim,
    /// Snapshot the root subvolume, capturing a frozen readonly view.
    Snapshot,
    /// Delete a previously-taken readonly snapshot.
    DeleteSnap { which: u8 },
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

/// A tracked readonly snapshot: its subvol id and the frozen content it must
/// forever return.
struct Snap {
    subvol: SubvolId,
    content: Shadow,
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

/// Compare `fs`'s current view against `shadow` over the whole tracked
/// file/offset space: matching cells must be byte-identical, and neither side
/// may hold an extent the other lacks. `label`/`subvol` are for panic context.
fn check_view(fs: &Fs, shadow: &Shadow, label: &str, subvol: SubvolId) {
    for file in 0..N_FILES as u8 {
        let ino = ino_of(file);
        for off_idx in 0..N_OFFSETS as u8 {
            let off = offset_of(off_idx);
            let got = fs.get_extent(ino, off).expect("get_extent");
            match (shadow.get(&(ino, off)), got) {
                (Some(cell), Some(ext)) => {
                    assert_eq!(
                        ext.len as usize, cell.len,
                        "{label} subvol={subvol} len mismatch ino={ino} off={off}"
                    );
                    let block = fs.read_data_block(ext.data_block).expect("read block");
                    for (i, b) in block.iter().enumerate() {
                        let want = if i < cell.len { cell.fill } else { 0 };
                        assert_eq!(
                            *b, want,
                            "{label} subvol={subvol} byte {i} mismatch ino={ino} off={off}: \
                             got {b} want {want}"
                        );
                    }
                }
                (None, None) => {}
                (Some(_), None) => panic!(
                    "{label} subvol={subvol} lost an extent it must have: ino={ino} off={off}"
                ),
                (None, Some(_)) => panic!(
                    "{label} subvol={subvol} sees an extent it must not: ino={ino} off={off}"
                ),
            }
        }
    }
}

/// Reconstruct what `do_read(ino, off, len)` must return from the shadow.
/// Mirrors the read contract: clip `[off, off+len)` to `size`, then fill each
/// byte from its covering cell (a block-keyed extent holds `fill` for its
/// first `cell.len` bytes, zero past that up to the block end) or zero where no
/// extent exists.
fn expected_read(shadow: &Shadow, ino: u64, size: u64, off: u64, len: u32) -> Vec<u8> {
    if off >= size {
        return Vec::new();
    }
    let end = (off + len as u64).min(size);
    let mut out = vec![0u8; (end - off) as usize];
    for (i, b) in out.iter_mut().enumerate() {
        let pos = off + i as u64;
        let block_off = pos & !(BLOCK - 1);
        if let Some(cell) = shadow.get(&(ino, block_off)) {
            let in_block = (pos - block_off) as usize;
            if in_block < cell.len {
                *b = cell.fill;
            }
        }
    }
    out
}

/// Verify every tracked snapshot still returns its frozen content. Switches
/// into each snapshot subvol and back to root; requires `&mut` for the switch.
fn check_snapshots(fs: &mut Fs, snaps: &[Snap]) {
    for snap in snaps {
        fs.switch_subvol(snap.subvol).expect("switch into snapshot");
        check_view(fs, &snap.content, "snapshot", snap.subvol);
    }
    fs.switch_subvol(ROOT_SUBVOL).expect("switch back to root");
}

fuzz_target!(|data: &[u8]| {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("fuzz.img");
    let mut fs = fresh(&path);

    let mut active: Shadow = HashMap::new();
    // Per-file EOF, mirroring inode.size. Grows monotonically on write (like
    // the real `max`), so a shrinking overwrite does not lower it — which the
    // block-granular `active` map alone cannot represent. Read clamping needs
    // this. Reset to 0 on unlink (files are never recreated here).
    let mut sizes = [0u64; N_FILES as usize];
    let mut live = [true; N_FILES as usize];
    let mut snaps: Vec<Snap> = Vec::new();

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
                fs.put_extent(ino, offset, &vec![fill; len]).expect("put_extent");
                // Mirror the FUSE write path: it grows inode.size to cover the
                // written range (see `FuseFs::write`). do_read clamps to
                // inode.size, so the oracle must track it too.
                let mut inode = fs.get_inode(ino).expect("get_inode").expect("inode exists");
                inode.size = inode.size.max(offset + len as u64);
                fs.put_inode(ino, &inode).expect("put_inode");
                fs.journal_commit().expect("commit write");
                active.insert((ino, offset), Cell { fill, len });
                let slot = (file % N_FILES as u8) as usize;
                sizes[slot] = sizes[slot].max(offset + len as u64);
            }
            Op::Read { file, off, len } => {
                let slot = (file % N_FILES as u8) as usize;
                let ino = ino_of(file);
                // Keep the range within the tracked block space (0..N_OFFSETS
                // blocks) but allow non-aligned starts and past-EOF tails.
                let span = N_OFFSETS * BLOCK;
                let off = off as u64 % (span + 1);
                let len = len as u32 % (span as u32 + 1);
                let got = do_read(&fs, ino, off, len).expect("do_read");
                let want = expected_read(&active, ino, sizes[slot], off, len);
                assert_eq!(
                    got, want,
                    "read mismatch ino={ino} off={off} len={len} size={}",
                    sizes[slot]
                );
            }
            Op::Unlink { file } => {
                let slot = (file % N_FILES as u8) as usize;
                if !live[slot] {
                    continue;
                }
                if fs.unlink(ROOT_INO, &name_of(file)).is_ok() {
                    fs.journal_commit().expect("commit unlink");
                    let _ = fs.reclaim_deleted_inodes(usize::MAX);
                    fs.journal_commit().expect("commit reclaim");
                    let ino = ino_of(file);
                    active.retain(|&(i, _), _| i != ino);
                    sizes[slot] = 0;
                    live[slot] = false;
                }
            }
            Op::Sync => fs.sync().expect("sync"),
            Op::Gc => {
                fs.gc().expect("gc");
            }
            Op::Reclaim => {
                let _ = fs.reclaim_deleted_inodes(usize::MAX);
                fs.journal_commit().expect("commit reclaim");
            }
            Op::Snapshot => {
                // Always exercise the code; only track up to MAX_SNAPS so
                // verification stays bounded. Tracking requires active ==
                // ROOT_SUBVOL (always true here — we never switch for writes).
                match fs.snapshot_subvol(ROOT_SUBVOL) {
                    Ok(sub) => {
                        fs.journal_commit().expect("commit snapshot");
                        if snaps.len() < MAX_SNAPS {
                            snaps.push(Snap { subvol: sub, content: active.clone() });
                        }
                    }
                    Err(_) => {} // snap-id exhaustion etc. — tolerate.
                }
            }
            Op::DeleteSnap { which } => {
                if snaps.is_empty() {
                    continue;
                }
                let idx = (which as usize) % snaps.len();
                let sub = snaps[idx].subvol;
                // delete_snapshot requires a readonly, non-active, leaf
                // snapshot — all our tracked snaps qualify. On success, stop
                // tracking it (its exclusively-owned blocks may now be gc'd).
                if fs.delete_snapshot(sub).is_ok() {
                    fs.journal_commit().expect("commit delete_snapshot");
                    snaps.remove(idx);
                }
            }
            Op::Reopen => {
                fs.sync().expect("sync before reopen");
                drop(fs);
                fs = Fs::open(&path).expect("reopen image");
            }
        }
        check_view(&fs, &active, "active", ROOT_SUBVOL);
        check_snapshots(&mut fs, &snaps);
    }

    // Final gauntlet: gc then reopen must both preserve every view.
    fs.gc().expect("final gc");
    check_view(&fs, &active, "active", ROOT_SUBVOL);
    check_snapshots(&mut fs, &snaps);
    fs.sync().expect("final sync");
    drop(fs);
    let mut fs = Fs::open(&path).expect("final reopen");
    check_view(&fs, &active, "active", ROOT_SUBVOL);
    check_snapshots(&mut fs, &snaps);
});
