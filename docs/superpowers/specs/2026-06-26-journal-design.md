# Journal Design (WAL Phase 2)

## Summary

Append-only journal ring providing crash recovery for rfs. Each FUSE write
op commits a journal entry before returning; recovery scans the ring to find
the last valid commit and restores Fs state from it.

## Constraints

- COW btree: journal records state snapshots, not redo data
- One journal entry per FUSE write op (no batching)
- Fixed 256 KB journal region (64 blocks), compile-time constant
- Superblock version bump to 2; no migration from v1

## Disk Layout

```
Block 0:        Superblock (version 2)
Block 1..64:    Journal ring (64 blocks = 256 KB)
Block 65+:      Btree nodes + data blocks
```

### Superblock v2

Added field:

| Field | Type | Description |
|-------|------|-------------|
| `journal_seq` | u64 | Seq of last checkpoint written to superblock |

`SUPERBLOCK_VERSION` bumps from 1 to 2. Old images are rejected on open.

### Journal Entry (128 bytes, fixed)

```rust
#[repr(C)]
struct JournalEntry {
    magic: u32,
    checksum: u32,       // CRC32 with this field zeroed (same pattern as Superblock)
    seq: u64,
    root_block: u64,
    next_block_nr: u64,
    next_bset_seq: u64,
    next_ino: u64,
    next_snap_id: u32,
    next_subvol_id: u32,
    current_subvol: u32,
    _reserved: [u8; 60],
}
```

128 bytes per entry. 31 entries per 4 KB block (3968 bytes used, 128 padding).
64 blocks x 31 = 1984 entries total capacity.

### Ring Addressing

```
block_index = (seq % 1984) / 31 + 1      // +1 because block 0 is superblock
slot_index  = (seq % 1984) % 31
file_offset = block_index * 4096 + slot_index * 128
```

## Write Path

Each write FUSE op (create/mkdir/write/unlink/rmdir/rename):

1. Perform COW btree operations (produces new node blocks via pwrite)
2. Construct JournalEntry with seq = next_journal_seq++
3. Compute CRC32 over entry bytes [8..128]
4. pwrite the entire 4 KB journal block containing this entry
5. fdatasync (flushes node blocks + journal block together)
6. Return success to FUSE caller

Key properties:
- Single fdatasync per op covers both data and journal
- 4 KB aligned write is atomic on most hardware; CRC detects partial writes
- Superblock is NOT updated on every op

## Checkpoint

Triggered by:
- `Fs::sync()` (called on unmount via `destroy()`)
- Optionally: when ring is near capacity (seq - superblock.journal_seq > ~1800)

Checkpoint procedure:
1. Write superblock with current counters + `journal_seq = current_seq - 1`
2. fdatasync

After checkpoint, journal slots with seq <= superblock.journal_seq may be
overwritten by future entries.

## Recovery

On `Fs::open(path)`:

1. Read + verify superblock (magic / version 2 / CRC)
2. Extract `journal_seq` from superblock
3. Scan from `journal_seq + 1` forward:
   - For each candidate seq, compute block+slot position
   - Read entry, validate: magic matches AND CRC matches AND entry.seq == expected
   - Valid → record as last_valid, advance to next seq
   - Invalid → stop scanning
4. If last_valid found: use its counters as Fs state
5. If not found: use superblock counters (clean shutdown)
6. Set next_journal_seq = (last_valid.seq or journal_seq) + 1

Fresh image: `journal_seq = 0`, first committed entry gets seq 1. Seq 0 is
never written to the ring and serves as the "no journal activity" sentinel.

Safety guarantees:
- COW preserves old nodes: crash mid-COW leaves previous root intact
- Triple validation (magic + CRC + seq match) prevents accepting garbage or
  stale entries from a previous ring cycle
- Maximum scan length: 1984 entries (full ring)

## Code Changes

### storage.rs (~150 new lines)
- Superblock: add `journal_seq: u64`, bump version, shrink `_reserved`
- Constants: `FIRST_DATA_BLOCK_NR = 65`, `JOURNAL_BLOCKS = 64`,
  `ENTRIES_PER_BLOCK = 31`, `JOURNAL_CAPACITY = 1984`
- New `JournalEntry` struct (zerocopy, 128 bytes)
- New `Journal` struct with methods:
  - `new(file) -> Journal`
  - `append(&self, entry: &JournalEntry) -> Result<()>` (pwrite block)
  - `read_entry(&self, seq: u64) -> Result<Option<JournalEntry>>` (validate)
  - `scan_from(&self, start_seq: u64) -> Result<Option<JournalEntry>>`

### btree.rs (minimal)
- `initialize_with`: expected_root assertion adjusts to 65

### fs.rs (~80 lines changed)
- `Fs` gains `journal: Journal` and `next_journal_seq: u64` fields
- New `Fs::journal_commit(&mut self) -> Result<()>`
- Each write method calls `journal_commit()` after btree mutation
- `Fs::sync()` becomes checkpoint (write superblock + fdatasync)
- `Fs::open()` gains recovery scan before returning

### fuse.rs
- No changes (writes go through Fs methods)

### block_btree.rs
- No changes

## Testing

1. **Unit**: Journal append + read round-trip, CRC validation, ring wrap
2. **Integration**: write ops → reopen without checkpoint → verify data present
3. **Crash simulation**: write partial entry (bad CRC) → reopen → verify
   rollback to previous valid entry
