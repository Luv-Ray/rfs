//! Microbenchmarks for the `Fs` layer, run against the pure-RAM store
//! (`Fs::new()` → `MemDevice`) so they measure the b-tree / extent / read /
//! write logic in isolation, with no file I/O or page-cache noise in the way.
//!
//! Harness: nightly libtest `#[bench]` (zero extra dependencies). Run with
//!   `cargo +nightly bench`
//! or a single bench with
//!   `cargo +nightly bench read_small_range_from_large_file`.
//!
//! What each group exercises:
//!  * `read_*`  — `Fs::read_at`, i.e. the bounded extent scan + block copy.
//!  * `write_*` — `Fs::write_at`, i.e. per-block read-modify-write + extent
//!                insert/update in the b-tree.
//!  * `meta_*`  — inode / dirent point operations and inode allocation.
//!
//! These are steady-state numbers: state is built once outside the timed loop
//! where possible so the measured closure is just the operation under test.
//! Where the operation must mutate fresh state (extent *growth*), that is
//! called out in the bench's own comment.

#![feature(test)]

extern crate test;

use rfs::fs::{Fs, InodeV1, ROOT_INO};
use test::{Bencher, black_box};

const BLOCK: u64 = 4096;
const S_IFREG: u32 = 0o100000;

/// Fresh RAM-backed `Fs` with a single regular file inode of `size` bytes.
/// Returns the inode number. No extents are written — the caller decides.
fn fresh_file(size: u64) -> (Fs, u64) {
    let mut fs = Fs::new();
    let ino = fs.alloc_ino();
    let inode = InodeV1 {
        mode: S_IFREG | 0o644,
        uid: 0,
        gid: 0,
        nlink: 1,
        size,
        atime: 0,
        mtime: 0,
        ctime: 0,
        parent_ino: ROOT_INO,
    };
    fs.put_inode(ino, &inode).unwrap();
    (fs, ino)
}

/// Fresh file of `size` bytes with every block filled, `inode.size` set to
/// match so `read_at` returns real data rather than clipping at EOF.
///
/// Blocks are written front-to-back and contiguously (no holes). `sparse_file`
/// breaks the contiguity to expose how much of the read number depends on this
/// fully-dense best case.
fn filled_file(size: u64) -> (Fs, u64) {
    let (mut fs, ino) = fresh_file(size);
    let chunk = vec![0xABu8; size as usize];
    fs.write_at(ino, 0, &chunk).unwrap();
    (fs, ino)
}

/// A file of logical size `size` where only every `1/density`-th block is
/// actually written; the rest are holes. `inode.size` spans the whole range,
/// so reads across it hit the zero-fill path between real extents and the
/// bounded scan skips large gaps. Exercises the hole/extent interleaving that
/// `filled_file` (fully dense) never touches.
fn sparse_file(size: u64, every_nth: u64) -> (Fs, u64) {
    let (mut fs, ino) = fresh_file(size);
    let nblocks = size / BLOCK;
    let block = vec![0xABu8; BLOCK as usize];
    let mut b = 0;
    while b < nblocks {
        fs.write_at(ino, b * BLOCK, &block).unwrap();
        b += every_nth;
    }
    (fs, ino)
}

/// Filled file, then snapshotted `depth` times so the active subvolume's
/// ancestor chain is `depth+1` deep. Every read now walks that chain in
/// `find_visible`, and the file's live extents sit at the *root* snap while
/// the current snap is the deepest child — the worst case for visibility
/// fallback. Returns the fs (switched to the deepest writable subvol) and ino.
fn snapshotted_file(size: u64, depth: u32) -> (Fs, u64) {
    let (mut fs, ino) = filled_file(size);
    let sv = fs.current_subvol();
    for _ in 0..depth {
        // snapshot_subvol bumps the source subvol to a fresh child snap_id and
        // returns a new readonly sibling; staying on `sv` keeps writing/reading
        // under the deepening writable chain.
        let _ro = fs.snapshot_subvol(sv).unwrap();
        // Source subvol id is unchanged; its snap_id got deeper. Keep using it.
        fs.switch_subvol(sv).unwrap();
    }
    (fs, ino)
}

// ---------- read path ----------

/// Read a whole 1 MiB file end to end: 256 extents, full sequential scan.
#[bench]
fn read_seq_1mib(b: &mut Bencher) {
    let size = 1024 * 1024;
    let (fs, ino) = filled_file(size);
    b.iter(|| {
        let out = fs.read_at(black_box(ino), 0, size as u32).unwrap();
        black_box(out);
    });
}

/// Read one 4 KiB block from the middle of a 16 MiB file. This is the payoff
/// of the bounded extent scan: the timed cost should track the read range
/// (one overlapping extent), not the file's total extent count (4096).
#[bench]
fn read_small_range_from_large_file(b: &mut Bencher) {
    let size = 16 * 1024 * 1024;
    let (fs, ino) = filled_file(size);
    let mid = size / 2;
    b.iter(|| {
        let out = fs
            .read_at(black_box(ino), black_box(mid), BLOCK as u32)
            .unwrap();
        black_box(out);
    });
}

/// Same tiny read, but from a small (64 KiB) file, as a baseline. Comparing
/// this against `read_small_range_from_large_file` shows how flat the read
/// cost stays as the file (and total extent count) grows.
#[bench]
fn read_small_range_from_small_file(b: &mut Bencher) {
    let size = 64 * 1024;
    let (fs, ino) = filled_file(size);
    let mid = size / 2;
    b.iter(|| {
        let out = fs
            .read_at(black_box(ino), black_box(mid), BLOCK as u32)
            .unwrap();
        black_box(out);
    });
}

/// Unaligned read spanning two extents (crosses a block boundary), the case
/// the block-aligned `scan_start` lower bound exists to cover.
#[bench]
fn read_unaligned_cross_block(b: &mut Bencher) {
    let size = 1024 * 1024;
    let (fs, ino) = filled_file(size);
    let off = BLOCK - 100; // spans block 0 and block 1
    b.iter(|| {
        let out = fs.read_at(black_box(ino), black_box(off), 200).unwrap();
        black_box(out);
    });
}

// ---------- read path: sparse (hole/extent interleaving) ----------
// Insertion *order* turns out not to affect read cost — the b-tree is
// self-sorting, so a scrambled-insert tree and a sequential-insert tree are
// the same tree at query time (measured: <1% delta). The fragmentation that
// does matter is in the *key space*: holes between extents. `sparse_file`
// builds that; compare against the fully-dense `filled_file` reads above.

/// Read a 64 KiB window over a sparse region: logical 16 MiB file, one real
/// block every 8 => the window spans real extents and holes. Exercises the
/// zero-fill-between-extents path plus scanning past skipped gaps.
#[bench]
fn read_window_sparse(b: &mut Bencher) {
    let size = 16 * 1024 * 1024;
    let (fs, ino) = sparse_file(size, 8);
    let off = size / 2;
    b.iter(|| {
        let out = fs
            .read_at(black_box(ino), black_box(off), 64 * 1024)
            .unwrap();
        black_box(out);
    });
}

// ---------- read path: under snapshots (CoW visibility) ----------
// The distinguishing cost of this filesystem: a read resolves each key
// through the active subvolume's ancestor snap chain (`find_visible`). Deeper
// chains = more fallback work per key. These hold the file fixed and vary
// only snapshot depth, so the delta is the visibility walk itself.

/// Baseline: small read, no snapshots (chain depth 1). Pair with the _depthN
/// variants below to read off the per-level cost of the ancestor walk.
#[bench]
fn read_small_range_snap_depth0(b: &mut Bencher) {
    let size = 1024 * 1024;
    let (fs, ino) = snapshotted_file(size, 0);
    let mid = size / 2;
    b.iter(|| {
        let out = fs
            .read_at(black_box(ino), black_box(mid), BLOCK as u32)
            .unwrap();
        black_box(out);
    });
}

/// Same read after 8 snapshots: the live extent still sits at the root snap,
/// so `find_visible` walks 8 ancestor levels before finding it.
#[bench]
fn read_small_range_snap_depth8(b: &mut Bencher) {
    let size = 1024 * 1024;
    let (fs, ino) = snapshotted_file(size, 8);
    let mid = size / 2;
    b.iter(|| {
        let out = fs
            .read_at(black_box(ino), black_box(mid), BLOCK as u32)
            .unwrap();
        black_box(out);
    });
}

/// And after 32 snapshots, to confirm the walk scales linearly with depth
/// rather than hiding a worse-than-linear term.
#[bench]
fn read_small_range_snap_depth32(b: &mut Bencher) {
    let size = 1024 * 1024;
    let (fs, ino) = snapshotted_file(size, 32);
    let mid = size / 2;
    b.iter(|| {
        let out = fs
            .read_at(black_box(ino), black_box(mid), BLOCK as u32)
            .unwrap();
        black_box(out);
    });
}

// NOTE: there is deliberately no `snapshot_create` bench. Taking a snapshot
// mutates unbounded state — each `snapshot_subvol` grows the snapshot/subvol
// metadata tree and deepens the active chain — and libtest `#[bench]` has no
// per-iteration setup hook to reset between calls. Timing it in a loop would
// measure "snapshot into an ever-growing tree" (tens of thousands of entries
// by the end of a run), not the cost on a normal fs. That needs a harness with
// per-iteration setup (e.g. criterion's iter_batched); it can't be measured
// honestly here. The read-side snapshot benches above are clean because the
// fixture is built once and reads don't mutate.

// ---------- write path ----------

/// Steady-state overwrite: rewrite an existing 64 KiB file in place every
/// iteration. Exercises the per-block read-modify-write + in-tree extent
/// update, with no b-tree growth (blocks already exist).
#[bench]
fn write_overwrite_64kib(b: &mut Bencher) {
    let size = 64 * 1024;
    let (mut fs, ino) = filled_file(size);
    let data = vec![0xCDu8; size as usize];
    b.iter(|| {
        let n = fs.write_at(black_box(ino), 0, black_box(&data)).unwrap();
        black_box(n);
    });
}

/// Growth: write 64 KiB into a *fresh, empty* file each iteration, so every
/// block is a new extent insertion into the b-tree. Includes `Fs::new` +
/// inode setup in the timed loop (cheap relative to 16 extent inserts) since
/// libtest `#[bench]` has no per-iteration setup hook; treat this as
/// "cold write" throughput, not a pure insert microbench.
#[bench]
fn write_grow_64kib_fresh(b: &mut Bencher) {
    let size = 64 * 1024;
    let data = vec![0xCDu8; size as usize];
    b.iter(|| {
        let (mut fs, ino) = fresh_file(size);
        let n = fs.write_at(ino, 0, black_box(&data)).unwrap();
        black_box(n);
    });
}

/// Partial-block write (100 bytes at an unaligned offset) against a file that
/// already has the covering block, i.e. a pure read-modify-write of one block.
#[bench]
fn write_partial_block_rmw(b: &mut Bencher) {
    let (mut fs, ino) = filled_file(BLOCK);
    let data = [0xEEu8; 100];
    b.iter(|| {
        let n = fs.write_at(black_box(ino), 50, black_box(&data)).unwrap();
        black_box(n);
    });
}

// ---------- metadata / point ops ----------

/// Exact-match extent lookup — the write path's read-modify-write probe.
#[bench]
fn meta_get_extent(b: &mut Bencher) {
    let (fs, ino) = filled_file(1024 * 1024);
    let off = 512 * 1024; // block-aligned
    b.iter(|| {
        let e = fs.get_extent(black_box(ino), black_box(off)).unwrap();
        black_box(e);
    });
}

/// Inode read (point lookup in the b-tree).
#[bench]
fn meta_get_inode(b: &mut Bencher) {
    let (fs, ino) = fresh_file(0);
    b.iter(|| {
        let i = fs.get_inode(black_box(ino)).unwrap();
        black_box(i);
    });
}

/// Directory lookup by name in a directory holding 1000 entries.
#[bench]
fn meta_lookup_dirent_1k(b: &mut Bencher) {
    let (mut fs, _dir) = fresh_file(0);
    let parent = ROOT_INO;
    for i in 0..1000u32 {
        let child = fs.alloc_ino();
        let d = rfs::fs::DirentV1::new(child, (S_IFREG >> 12) as u8);
        fs.put_dirent(parent, format!("file{i:04}").as_bytes(), &d)
            .unwrap();
    }
    let needle = b"file0500";
    b.iter(|| {
        let d = fs
            .lookup_dirent(black_box(parent), black_box(needle))
            .unwrap();
        black_box(d);
    });
}

/// Inode-number allocation throughput.
#[bench]
fn meta_alloc_ino(b: &mut Bencher) {
    let mut fs = Fs::new();
    b.iter(|| {
        black_box(fs.alloc_ino());
    });
}
