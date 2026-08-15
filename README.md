A demo fs for learning from bcachefs.

Backed by a single image file (default mode is still in-memory; pass
`--image <path>` to mount on a persistent file). A write-ahead journal
provides crash recovery on the image backend.

## Features

Core storage:
- COW B-tree with split, multi-bset node layout (k-way merged sorted runs)
- Zerocopy on-disk layout (4 KB nodes, `NodeHeader` / `DiskEntry`)
- Multi-tree view via key prefix (inode / dirent / extent in one physical btree, bcachefs style)
- In-place delete optimization (flip kind byte when deleting own-snap key)

Snapshots:
- snap_id embedded in every key, iterator ancestor filtering
- Snapshot tree + Subvolume tree
- Writable snapshots via `snapshot_subvol` / `switch_subvol`

FUSE (`fuser` 0.17, pure Rust):
- `lookup / getattr / readdir / read / write / create / mkdir / unlink / rmdir / rename`
- Multi-block writes (4 KB chunks) and zero-filled sparse reads
- Atomic multi-key transactions for metadata ops

Persistence:
- Single backing image file with superblock, CRC32 per node block
- `BlockStore` with a dirty-tracked mutable node/data cache
- In-place bset append: between checkpoints, writes a hot leaf can absorb
  mutate the cached node at a stable block number (no per-op root→leaf COW);
  a checkpoint relocates the dirty nodes onto fresh blocks and swaps the root
- `Fs::create` / `Fs::open` / `Fs::sync`; FUSE auto-syncs on destroy
- Write-ahead journal (ring buffer, seq + CRC per frame) recording key-level
  logged ops in atomic commit groups; replay-on-open recovery from the last
  superblock checkpoint; superblock checkpoint on sync

## TODO

Functionality gaps (user-visible):
- [ ] `setattr`: truncate / chmod / utimens (truncate especially — files can
      currently only grow; `O_TRUNC` / `ftruncate` don't work)
- [ ] Expose subvolume / snapshot management via FUSE (`snapshot_subvol` /
      `switch_subvol` exist but have no mount-side entry point)

Snapshot lifecycle (one connected piece):
- [x] `needs_whiteout` bit + whiteout-only compaction (per-bset flag,
      `Btree::compact_whiteouts` drops only whiteouts at dead snap_ids; a
      winning whiteout drop also drops shadowed duplicate copies via merged
      compaction, so lower-seq versions cannot resurface)
- [x] Snapshot deletion (`Fs::delete_snapshot`: validates leaf snapshot /
      non-active subvol, atomically tombstones snapshot+subvol metadata,
      `Btree::drop_snapshot_keys` compacts away every key version at the
      dead snap_id, then sweeps unreferenced internal snapshot nodes)
- [ ] `deleted_inodes` btree + background reclaim (bcachefs style): make unlink
      bounded by recording the orphan inode and reclaiming its extents lazily,
      instead of deleting every extent in one transaction. Removes the current
      limit that unlinking a large file (> ~3.6 MB) can overflow the journal
      ring in a single commit group.

Write-amplification path (prerequisite chain for larger nodes):
- [x] Node cache rewrite: `FrozenMap` → dirty-tracked mutable cache
- [x] Journal: fixed-size checkpoint → variable-length logical WAL (key-level
      ops in commit groups) + replay-from-checkpoint recovery
- [x] In-place bset append: hot writes mutate the cached node at a stable block
      number; checkpoint relocates dirty nodes and swaps the root
- [ ] On-disk incremental flush: persist appended bsets alone (per-bset checksum
      + `journal_seq`, bsets found by scan not authoritative `bset_count`);
      recovery drops a torn tail bset. Currently a checkpoint still rewrites each
      dirty node whole — this amortizes that for large nodes.
- [ ] Raise node size to 64–256 KB (needs COW write-amp benchmarks)

Optimizations / deferrable (don't block anything):
- [ ] Block reclaim / GC: mark-and-sweep for orphaned COW blocks + reclaim on
      overwrite / unlink (needs per-block refcounting with snapshots)
- [ ] Sibling merge / rebalance on sparse leaves (delete never shrinks the tree)
- [ ] Direct I/O + aligned buffers
- [ ] Multi-superblock for atomic superblock update
