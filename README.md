A demo fs for learning from bcachefs.

All in-memory for now.

See [docs/snapshot-delete-plan.md](docs/snapshot-delete-plan.md) for the
delete + snapshot design, including the bcachefs mapping.

## Status

**Done**
- [x] COW B-tree with split / rebalance; all nodes live in `HashMap<u64, BtreeNodeRaw>`
- [x] Zerocopy on-disk layout (`NodeHeader` / `DiskEntry`, 4 KB nodes)
- [x] Btree API returns `Result` (prep for disk-backed I/O errors)
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

**TODO**
- [ ] Btree node size: currently 4 KB to match an OS page and keep COW cheap; bcachefs uses 256 KB by default. Once persistence + journal land, raise `BLOCK_SIZE` to 64 KB / 256 KB so `MAX_ENTRIES` (now 29) and `BSET_SOFT_LIMIT` (now 7) become big enough for multi-bset to actually pay off vs single-bset binary search. Needs benchmarks for COW write amplification (`clone_to_heap` cost scales linearly with node size).
- [ ] `needs_whiteout` bit + whiteout-only compaction (only meaningful once on-disk)
- [ ] Snapshot deletion: walk all btrees, drop keys at the gone snap_id, clean dependent whiteouts
- [ ] Sibling merge / rebalance on sparse leaves
- [ ] `setattr`: truncate / chmod / utimens
- [ ] Reclaim old data block on extent overwrite / unlink (snapshot inheritance complicates this — needs per-block refcounting)
- [ ] Persist block_map + data_blocks to a real block device; magic / CRC on read
- [ ] Expand `btree::Error` with Io / BadMagic / ChecksumMismatch once disk-backed
- [ ] Crash recovery using `NodeHeader.generation`
- [ ] Subvolume management exposed via FUSE (currently only via Rust API)
