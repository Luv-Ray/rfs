A demo fs for learning from bcachefs.

All in-memory for now.

## Status

**Done**
- [x] COW B-tree with split / rebalance; all nodes live in `HashMap<u64, BtreeNodeRaw>`
- [x] Zerocopy on-disk layout (`NodeHeader` / `DiskEntry`, 4 KB nodes)
- [x] Btree API returns `Result` (prep for disk-backed I/O errors)
- [x] Multi-tree view via key prefix: inode / dirent / extent sharing one physical Btree (bcachefs style)
- [x] FUSE via `fuser` 0.17 pure-rust (no `libfuse-dev` dependency)
- [x] `lookup / getattr / readdir / read / write / create / mkdir` — enough for `mkdir / echo > / cat / ls / cd ..`
- [x] Multi-block writes (4 KB chunks + read-modify-write) and zero-filled sparse reads

**TODO**
- [ ] Btree delete (COW rebalance) — unblocks `unlink / rmdir / rename`
- [ ] `setattr`: truncate / chmod / utimens
- [ ] Reclaim old data block on extent overwrite (currently leaks)
- [ ] Persist block_map + data_blocks to a real block device; magic / CRC on read
- [ ] Expand `btree::Error` with Io / BadMagic / ChecksumMismatch once disk-backed
- [ ] Snapshots: expose the COW root-swap as a user-visible `snapshot(name)` op
- [ ] Crash recovery using `NodeHeader.generation`
