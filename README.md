A demo fs for learning from bcachefs.

Backed by a single image file (default mode is still in-memory; pass
`--image <path>` to mount on a persistent file). Synchronous writes only —
no journal, no crash recovery yet.

See [docs/snapshot-delete-plan.md](docs/snapshot-delete-plan.md) for the
delete + snapshot design, including the bcachefs mapping.

## Status

**Done**
- [x] COW B-tree with split / rebalance
- [x] Zerocopy on-disk layout (`NodeHeader` / `DiskEntry`, 4 KB nodes)
- [x] Btree API returns `Result` (Io / BadMagic / ChecksumMismatch / BlockNotFound)
- [x] Multi-tree view via key prefix: inode / dirent / extent sharing one physical Btree (bcachefs style)
- [x] FUSE via `fuser` 0.17 pure-rust (no `libfuse-dev` dependency)
- [x] `lookup / getattr / readdir / read / write / create / mkdir` — enough for `mkdir / echo > / cat / ls / cd ..`
- [x] Multi-block writes (4 KB chunks + read-modify-write) and zero-filled sparse reads
- [x] **Btree delete with bcachefs-style `Deleted` / `Whiteout` distinction**
- [x] **snap_id embedded in every key + iterator ancestor filter** (`BTREE_ITER_FILTER_SNAPSHOTS` semantics)
- [x] **Snapshot tree + Subvolume tree** (`KIND_SNAPSHOT` / `KIND_SUBVOL`)
- [x] **`Btree::transaction` for atomic multi-key ops** (used by `unlink`, `rmdir`, `rename`)
- [x] **`unlink / rmdir / rename` exposed via FUSE**
- [x] **`Fs::snapshot_subvol` + `Fs::switch_subvol`** — bcachefs-style writable snapshots: src keeps writing under a new id, snapshot subvol gets a readonly id, both inheriting from the old snap_id
- [x] **Multi-bset node layout** (bcachefs `bset` infra): per-node `BsetHeader { seq, nkeys, flags }`, up to `BSET_TREE_NR_MAX=4` sorted runs per leaf, k-way merged search/iter (highest-seq wins on dup), append-to-last-bset insert, soft-limit roll-over, 4-bsets-full → compact, split compacts multi-bset source to single bset
- [x] **In-place delete optimization**: when `visible_snap == snap` (delete-of-own-key), flip `kind` byte from `Live` to `Deleted` in the existing entry instead of sort-inserting a fresh tombstone — `nkeys` unchanged, no new bset opened, no possible split. Cross-snap deletes still write a `Whiteout` entry.
- [x] **Persistence (Phase 1: image file, no journal)** — single backing file; superblock at block 0 with magic / version / CRC32 / root_block / next_block_nr / next_bset_seq / next_ino / next_snap_id / next_subvol_id / current_subvol; CRC32 stamped into every node block on write and verified on read; magic mismatch and checksum mismatch surfaced as typed `btree::Error` variants. Allocator unified between btree nodes and data blocks (`store.alloc()` returns next free block_nr from a single namespace). `Fs::create(path)` / `Fs::open(path)` / `Fs::sync()`; FUSE picks up `--image <path>`, auto-syncs on `destroy()`.
- [x] **`BlockStore` with append-only cache** — `elsa::sync::FrozenMap` keeps cached blocks borrow-stable across cache faults so `read_node(&self, nr) -> &BtreeNodeRaw` doesn't need `&mut self`. COW guarantees a freshly-allocated `block_nr` is written exactly once before being read, matching FrozenMap's append-only contract perfectly. `BtreeNodeRaw` and `DataBlock` use separate cache lanes so 4 KB-aligned zerocopy casts stay sound.

**TODO**
- [ ] Btree node size: currently 4 KB to match an OS page and keep COW cheap; bcachefs uses 256 KB by default. Once persistence + journal land, raise `BLOCK_SIZE` to 64 KB / 256 KB so `MAX_ENTRIES` (now 29) and `BSET_SOFT_LIMIT` (now 7) become big enough for multi-bset to actually pay off vs single-bset binary search. Needs benchmarks for COW write amplification (`clone_to_heap` cost scales linearly with node size).
- [ ] `needs_whiteout` bit + whiteout-only compaction
- [ ] Snapshot deletion: walk all btrees, drop keys at the gone snap_id, clean dependent whiteouts
- [ ] Sibling merge / rebalance on sparse leaves
- [ ] `setattr`: truncate / chmod / utimens
- [ ] Reclaim old data block on extent overwrite / unlink (snapshot inheritance complicates this — needs per-block refcounting)
- [ ] Subvolume management exposed via FUSE (currently only via Rust API)

**TODO (persistence — phase 2 and beyond)**
- [ ] Write-ahead journal + crash recovery: append-only journal region; each transaction stages dirty block_nrs, flushes, then advances the superblock; on open, find the last valid journal entry and replay. `NodeHeader.generation` already in place for this.
- [ ] Block GC (mark-and-sweep, bcachefs style): periodically walk live roots (current + every snapshot's), compute `live_set`, free `(0..next_block_nr) - live_set`, drop those entries from the cache, prefer them on next `alloc()`. Until this lands, the image file grows monotonically.
- [ ] Bounded cache + LRU eviction: today the `FrozenMap` cache never shrinks (matches "no GC" but blows RAM on large images). Needs a real cache with eviction, which probably means dropping the FrozenMap append-only invariant and going through an `RwLock<HashMap>` or a per-entry lock as bcachefs does.
- [ ] Direct I/O (`O_DIRECT`) + 4 KB-aligned buffers: bypass the OS page cache, control writeback ordering ourselves. Today we use buffered IO and rely on `fsync` for ordering — fine for a learning project, not OK for a real fs.
- [ ] Multi-superblock + watermark for atomic superblock update: bcachefs writes N copies of the superblock; we currently overwrite block 0 in place and a torn write at the wrong moment is unrecoverable.
