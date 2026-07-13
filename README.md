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

Near-term:
- [ ] `needs_whiteout` bit + whiteout-only compaction
- [ ] Snapshot deletion (walk btrees, drop gone snap_id keys, clean whiteouts)
- [ ] Sibling merge / rebalance on sparse leaves
- [ ] `setattr`: truncate / chmod / utimens
- [ ] Block reclaim on overwrite / unlink (needs per-block refcounting with snapshots)
- [ ] Expose subvolume management via FUSE

Persistence:
- [ ] Block GC (mark-and-sweep)
- [ ] Bounded cache + LRU eviction
- [ ] Direct I/O + aligned buffers
- [ ] Multi-superblock for atomic superblock update

Longer-term:
- [ ] Raise node size to 64–256 KB (needs COW write-amp benchmarks)
