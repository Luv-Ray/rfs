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
- `BlockStore` with append-only `FrozenMap` cache (borrow-stable across faults)
- `Fs::create` / `Fs::open` / `Fs::sync`; FUSE auto-syncs on destroy
- Write-ahead journal (ring buffer, seq + CRC per entry) with commit after every
  write op and replay-on-open crash recovery; superblock checkpoint on sync

## TODO

Functionality gaps (user-visible):
- [ ] `setattr`: truncate / chmod / utimens (truncate especially — files can
      currently only grow; `O_TRUNC` / `ftruncate` don't work)
- [ ] Expose subvolume / snapshot management via FUSE (`snapshot_subvol` /
      `switch_subvol` exist but have no mount-side entry point)

Snapshot lifecycle (one connected piece):
- [ ] `needs_whiteout` bit + whiteout-only compaction (let compaction safely
      drop whiteouts, not just `Deleted`)
- [ ] Snapshot deletion (walk btrees, drop gone snap_id keys, clean whiteouts)

Write-amplification path (prerequisite chain for larger nodes, in order):
- [ ] Node cache rewrite: `FrozenMap` → dirty-tracked mutable cache. Current
      COW-once + borrow-stable cache forbids in-place node mutation; bcachefs-style
      in-place bset append needs this. Subsumes bounded cache + LRU eviction.
- [ ] Journal: fixed-size checkpoint → variable-length logical WAL (record
      key-level ops, not just a `root_block` snapshot)
- [ ] Recovery: adopt-root → replay logged ops from last checkpoint
- [ ] Incremental flush: write appended bsets alone; amortize full-node rewrite
      into compaction / split
- [ ] Raise node size to 64–256 KB (needs COW write-amp benchmarks)

Optimizations / deferrable (don't block anything):
- [ ] Block reclaim / GC: mark-and-sweep for orphaned COW blocks + reclaim on
      overwrite / unlink (needs per-block refcounting with snapshots)
- [ ] Sibling merge / rebalance on sparse leaves (delete never shrinks the tree)
- [ ] Direct I/O + aligned buffers
- [ ] Multi-superblock for atomic superblock update
