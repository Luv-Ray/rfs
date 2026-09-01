use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

pub const BLOCK_SIZE: usize = 4096;
pub const MAGIC_NUMBER: u32 = 0x39C5BB39;

/// Total key buffer in DiskEntry: logical bytes followed by 4-byte snap_id (BE).
pub const MAX_KEY_SIZE: usize = 32;
/// Maximum length of the logical (caller-visible) portion of a key.
/// snap_id occupies the next 4 bytes after the logical part, so
/// `MAX_LOGICAL_KEY_SIZE + 4 == MAX_KEY_SIZE`.
pub const MAX_LOGICAL_KEY_SIZE: usize = MAX_KEY_SIZE - 4;
pub const MAX_VALUE_SIZE: usize = 96;
pub const SNAP_ID_BYTES: usize = 4;

/// Snapshot id type. bcachefs convention: `u32::MAX` is the root, new ids are
/// allocated downward so a parent's id is always greater than its children's.
pub type SnapId = u32;
pub const ROOT_SNAP: SnapId = u32::MAX;

/// Maximum number of bsets (sorted runs) per node.
///
/// bcachefs uses `BSET_TREE_NR_MAX = 4`. When this limit is hit, the node is
/// compacted into a single fresh bset before the next write proceeds.
pub const BSET_TREE_NR_MAX: usize = 4;

/// Soft cap on the *latest* bset's entry count before opening a new bset.
///
/// Must satisfy `BSET_SOFT_LIMIT * BSET_TREE_NR_MAX < MAX_ENTRIES` so that
/// the worst-case "every bset filled to the soft limit" total stays below
/// the split threshold — otherwise the compaction-when-full code path is
/// unreachable (we'd always split before all four bsets fill). With
/// `MAX_ENTRIES = 29` and `BSET_TREE_NR_MAX = 4`, 7 is the largest legal
/// value (4 × 7 = 28 ≤ 28).
pub const BSET_SOFT_LIMIT: usize = 7;

/// `BsetHeader::flags` bit: this bset contains (or contained at some point)
/// at least one `Whiteout` entry. Kept as a "may contain whiteout" flag so a
/// snapshot-deletion pass can find whiteouts without scanning every entry of
/// every leaf; compaction recomputes it exactly from the surviving entries.
/// The flag may remain set after a whiteout is overwritten (over-approximation
/// is safe — it only causes an unnecessary whiteout-compaction pass).
pub const BSET_FLAG_NEEDS_WHITEOUT: u16 = 1 << 0;

/// Per-entry type tag. Stored in `DiskEntry::kind`.
///
/// - `Live`: normal key/value.
/// - `Deleted`: trivial tombstone (KEY_TYPE_deleted in bcachefs). Produced
///   when X deletes a key it itself wrote at snap_id == X. Like a whiteout it
///   must shadow ancestor versions until snap X is gone, so ordinary
///   compaction keeps it; snapshot deletion may drop it together with the
///   other keys at the dead snap_id.
/// - `Whiteout`: snapshot tombstone (KEY_TYPE_whiteout in bcachefs). Produced
///   when X deletes a key inherited from an ancestor snapshot. Must shadow
///   the ancestor's still-live key, so compaction keeps it until the relevant
///   snapshots are deleted.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryKind {
    Live = 0,
    Deleted = 1,
    Whiteout = 2,
}

impl EntryKind {
    pub fn from_u8(b: u8) -> Self {
        match b {
            0 => EntryKind::Live,
            1 => EntryKind::Deleted,
            2 => EntryKind::Whiteout,
            // Unknown tags fall back to Live so a corrupted byte doesn't
            // silently make data disappear; verify_node should catch it.
            _ => EntryKind::Live,
        }
    }

    pub fn is_tombstone(self) -> bool {
        matches!(self, EntryKind::Deleted | EntryKind::Whiteout)
    }
}

// ---------- Node header ----------

#[repr(C)]
#[derive(KnownLayout, Immutable, IntoBytes, FromBytes, Clone)]
pub struct NodeHeader {
    pub magic: u32,
    /// 0 = leaf, >0 = internal (value = distance from leaves).
    pub level: u8,
    /// Number of bsets currently packed in the body. 0..=BSET_TREE_NR_MAX.
    /// Newly-created nodes start with bset_count = 0 (empty).
    pub bset_count: u8,
    /// Reserved for future use (e.g. compression, checksum type).
    pub flags: u8,
    _pad0: u8,
    /// Incremented on each transaction commit; used for crash recovery.
    pub generation: u64,
    _pad1: [u8; 8],
    /// CRC-64 of the entire node (0 = not yet computed).
    pub checksum: u64,
    /// Pads header to 64 bytes; available for future fields.
    _reserved: [u8; 32],
}

const _: () = assert!(std::mem::size_of::<NodeHeader>() == 64);

// ---------- Bset header ----------

/// Per-bset header laid out at the start of each bset's region in body bytes.
///
/// bcachefs's `struct bset` carries seq + flags + version + size; this is
/// the minimal subset we need: a monotonic seq (so the merged iterator can
/// resolve ties between bsets storing the same key), the per-bset entry
/// count, and a flags field. Bit 0 is [`BSET_FLAG_NEEDS_WHITEOUT`].
#[repr(C)]
#[derive(KnownLayout, Immutable, IntoBytes, FromBytes, Clone)]
pub struct BsetHeader {
    /// Monotonically increasing per-write sequence number. When two bsets
    /// contain the same sortable key, the one with the higher seq wins.
    pub seq: u64,
    /// Number of valid DiskEntry slots immediately after this header.
    pub nkeys: u16,
    pub flags: u16,
    _pad: u32,
}

const _: () = assert!(std::mem::size_of::<BsetHeader>() == 16);

// ---------- Payload (leaf value or child pointer) ----------

/// Leaf nodes store values here; internal nodes store child block numbers
/// in the first 8 bytes.
#[repr(C, align(8))]
#[derive(KnownLayout, Immutable, IntoBytes, FromBytes, Clone, Copy)]
pub struct Payload([u8; MAX_VALUE_SIZE]);

impl Payload {
    pub fn child(&self) -> u64 {
        u64::from_ne_bytes(self.0[..8].try_into().unwrap())
    }

    pub fn set_child(&mut self, child: u64) {
        self.0[..8].copy_from_slice(&child.to_ne_bytes());
    }

    pub fn value(&self) -> &[u8] {
        &self.0
    }

    pub fn value_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

// ---------- Disk entry ----------

#[repr(C)]
#[derive(KnownLayout, Immutable, IntoBytes, FromBytes, Clone, Copy)]
pub struct DiskEntry {
    pub key: [u8; MAX_KEY_SIZE],
    /// Length of the logical portion of `key`. The 4 bytes immediately
    /// after that are the snap_id. So the sortable key length is `key_len + 4`.
    pub key_len: u8,
    /// Only meaningful in leaf nodes; ignored in internal nodes.
    pub value_len: u8,
    /// `EntryKind` discriminant. 0=Live, 1=Deleted, 2=Whiteout.
    pub kind: u8,
    /// Padding to keep `payload` 8-byte aligned.
    _pad: [u8; 5],
    pub payload: Payload,
}

const _: () =
    assert!(std::mem::size_of::<DiskEntry>() == MAX_KEY_SIZE + 1 + 1 + 1 + 5 + MAX_VALUE_SIZE); // 136

const ENTRY_SIZE: usize = std::mem::size_of::<DiskEntry>();
const HEADER_SIZE: usize = std::mem::size_of::<NodeHeader>();
const BSET_HEADER_SIZE: usize = std::mem::size_of::<BsetHeader>();
pub const BODY_SIZE: usize = BLOCK_SIZE - HEADER_SIZE; // 4032
/// Maximum total entries across all bsets, computed for the worst case
/// where all `BSET_TREE_NR_MAX` bsets coexist (each carries one header).
/// Same as the old single-bset limit (29) — multi-bset doesn't reduce
/// capacity because headers fit in slack space.
pub const MAX_ENTRIES: usize = (BODY_SIZE - BSET_TREE_NR_MAX * BSET_HEADER_SIZE) / ENTRY_SIZE; // 29
/// Internal nodes store nkeys+1 child pointers in entries[0..=nkeys].
/// The split-and-insert shift loop writes child[i+2], requiring
/// nkeys+2 <= MAX_ENTRIES, hence nkeys <= MAX_ENTRIES - 2.
pub const MAX_INTERNAL_KEYS: usize = MAX_ENTRIES - 2; // 27

// Compile-time check: every bset filled to the soft limit must still leave
// the node strictly under the split threshold, otherwise the
// 4-bsets-full → compact transition is unreachable (we'd always split first).
const _: () = assert!(BSET_SOFT_LIMIT * BSET_TREE_NR_MAX < MAX_ENTRIES);

// ---------- B-tree node (on-disk layout, one block) ----------
//
// Body layout (4032 bytes):
//   BsetHeader 0  (16B)
//   DiskEntry[0..bset0.nkeys]  (bset0.nkeys * 136B)
//   BsetHeader 1  (16B)
//   DiskEntry[0..bset1.nkeys]
//   ...
//   BsetHeader (bset_count - 1)
//   DiskEntry[...]
//   <unused tail>
//
// Each bset is independently sorted. Across bsets, the same sortable key
// may appear in multiple bsets; the highest-seq one wins (later writes
// shadow earlier ones).

#[repr(C, align(4096))]
#[derive(KnownLayout, Immutable, IntoBytes, FromBytes, Clone)]
pub struct BtreeNodeRaw {
    pub header: NodeHeader,
    pub body: [u8; BODY_SIZE],
}

const _: () = assert!(std::mem::size_of::<BtreeNodeRaw>() == BLOCK_SIZE);

// ---------- DiskEntry helpers ----------

impl DiskEntry {
    /// Logical key bytes only (no snap_id appended). This is what callers
    /// outside the btree see.
    pub fn logical_key_bytes(&self) -> &[u8] {
        &self.key[..self.key_len as usize]
    }

    /// Full sortable key bytes: logical || snap_id_be. This is what all
    /// internal comparisons use. Length is `key_len + 4`.
    pub fn key_bytes(&self) -> &[u8] {
        let n = self.key_len as usize;
        &self.key[..n + SNAP_ID_BYTES]
    }

    /// snap_id of this entry.
    pub fn snap_id(&self) -> SnapId {
        let n = self.key_len as usize;
        SnapId::from_be_bytes(self.key[n..n + SNAP_ID_BYTES].try_into().unwrap())
    }

    pub fn set_snap_id(&mut self, snap: SnapId) {
        let n = self.key_len as usize;
        self.key[n..n + SNAP_ID_BYTES].copy_from_slice(&snap.to_be_bytes());
    }

    pub fn kind_enum(&self) -> EntryKind {
        EntryKind::from_u8(self.kind)
    }

    pub fn set_kind(&mut self, k: EntryKind) {
        self.kind = k as u8;
    }

    /// Set the entry's key from a *sortable* byte slice: the last 4 bytes
    /// are taken as snap_id_be, everything before that is the logical key.
    /// This matches the slice returned by `key_bytes()` so `set_key(other.key_bytes())`
    /// copies a key faithfully between entries.
    pub fn set_key(&mut self, sortable: &[u8]) {
        assert!(
            sortable.len() >= SNAP_ID_BYTES,
            "sortable key must include {SNAP_ID_BYTES}-byte snap_id suffix; got {} bytes",
            sortable.len()
        );
        let total = sortable.len().min(MAX_KEY_SIZE);
        self.key[..total].copy_from_slice(&sortable[..total]);
        self.key[total..].fill(0);
        self.key_len = (total - SNAP_ID_BYTES) as u8;
    }

    /// Set the entry's key from an explicit (logical, snap_id) pair.
    /// Use this when constructing entries from caller-provided keys that
    /// don't include a snap_id suffix.
    pub fn set_key_with_snap(&mut self, logical: &[u8], snap: SnapId) {
        let n = logical.len().min(MAX_LOGICAL_KEY_SIZE);
        self.key[..n].copy_from_slice(&logical[..n]);
        self.key[n..n + SNAP_ID_BYTES].copy_from_slice(&snap.to_be_bytes());
        self.key[n + SNAP_ID_BYTES..].fill(0);
        self.key_len = n as u8;
    }

    pub fn empty() -> Self {
        DiskEntry {
            key: [0u8; MAX_KEY_SIZE],
            key_len: 0,
            value_len: 0,
            kind: EntryKind::Live as u8,
            _pad: [0u8; 5],
            payload: Payload([0u8; MAX_VALUE_SIZE]),
        }
    }
}

// ---------- Body navigation helpers ----------

/// Byte offset within `body` where bset `i`'s `BsetHeader` lives.
/// Walks preceding bsets' headers to compute the offset; O(bset_count).
fn bset_offset(node: &BtreeNodeRaw, target: usize) -> usize {
    debug_assert!(target < node.header.bset_count as usize);
    let mut off = 0usize;
    for _ in 0..target {
        let h = read_bset_header(node, off);
        off += BSET_HEADER_SIZE + (h.nkeys as usize) * ENTRY_SIZE;
    }
    off
}

fn read_bset_header(node: &BtreeNodeRaw, off: usize) -> BsetHeader {
    let bytes: &[u8; BSET_HEADER_SIZE] = node.body[off..off + BSET_HEADER_SIZE]
        .try_into()
        .expect("bset header slice in range");
    BsetHeader::read_from_bytes(bytes).expect("valid BsetHeader bytes")
}

fn write_bset_header(node: &mut BtreeNodeRaw, off: usize, header: &BsetHeader) {
    node.body[off..off + BSET_HEADER_SIZE].copy_from_slice(header.as_bytes());
}

/// View `body[off+16..off+16 + nkeys*136]` as a typed slice of DiskEntry.
fn entries_at(node: &BtreeNodeRaw, off: usize, nkeys: usize) -> &[DiskEntry] {
    let start = off + BSET_HEADER_SIZE;
    let end = start + nkeys * ENTRY_SIZE;
    <[DiskEntry]>::ref_from_bytes(&node.body[start..end]).expect("entries slice aligned and sized")
}

fn entries_at_mut(node: &mut BtreeNodeRaw, off: usize, nkeys: usize) -> &mut [DiskEntry] {
    let start = off + BSET_HEADER_SIZE;
    let end = start + nkeys * ENTRY_SIZE;
    <[DiskEntry]>::mut_from_bytes(&mut node.body[start..end])
        .expect("entries slice aligned and sized")
}

// ---------- BtreeNodeRaw: API ----------

impl BtreeNodeRaw {
    /// Construct a fresh empty node at the given level. Has zero bsets;
    /// the first insert will open bset 0 via `start_new_bset`.
    pub fn new(level: u8) -> Self {
        BtreeNodeRaw {
            header: NodeHeader {
                magic: MAGIC_NUMBER,
                level,
                bset_count: 0,
                flags: 0,
                _pad0: 0,
                generation: 0,
                _pad1: [0u8; 8],
                checksum: 0,
                _reserved: [0u8; 32],
            },
            body: [0u8; BODY_SIZE],
        }
    }

    pub fn level(&self) -> u8 {
        self.header.level
    }

    pub fn generation(&self) -> u64 {
        self.header.generation
    }

    pub fn set_generation(&mut self, new_gen: u64) {
        self.header.generation = new_gen;
    }

    /// Total number of stored DiskEntry slots across all bsets.
    /// For leaves this is the entry count. For internals (single-bset) this
    /// is `nkeys_separators + 1` because of the rightmost-child slot.
    pub fn total_stored(&self) -> usize {
        let mut sum = 0usize;
        for b in 0..self.bset_count() {
            sum += self.bset_header(b).nkeys as usize;
        }
        sum
    }

    /// Number of "search keys" by the bcachefs / classic-btree convention.
    /// - Leaves: same as `total_stored` (every entry is a key).
    /// - Internals: `total_stored - 1` (the last stored entry is the
    ///   rightmost-child slot and carries no separator).
    pub fn nkeys(&self) -> usize {
        let total = self.total_stored();
        if self.level() > 0 && total > 0 {
            total - 1
        } else {
            total
        }
    }

    pub fn bset_count(&self) -> usize {
        self.header.bset_count as usize
    }

    /// View bset `i`'s header.
    pub fn bset_header(&self, i: usize) -> BsetHeader {
        let off = bset_offset(self, i);
        read_bset_header(self, off)
    }

    /// Whether bset `i` has the "needs whiteout processing" flag set.
    pub fn bset_needs_whiteout(&self, i: usize) -> bool {
        self.bset_header(i).flags & BSET_FLAG_NEEDS_WHITEOUT != 0
    }

    /// Set or clear the "needs whiteout processing" flag on bset `i`.
    pub fn set_bset_needs_whiteout(&mut self, i: usize, needs: bool) {
        let off = bset_offset(self, i);
        let mut h = read_bset_header(self, off);
        if needs {
            h.flags |= BSET_FLAG_NEEDS_WHITEOUT;
        } else {
            h.flags &= !BSET_FLAG_NEEDS_WHITEOUT;
        }
        write_bset_header(self, off, &h);
    }

    /// Whether any bset in this node has the "needs whiteout processing" flag.
    pub fn any_bset_needs_whiteout(&self) -> bool {
        (0..self.bset_count()).any(|i| self.bset_needs_whiteout(i))
    }

    /// Sorted entries belonging to bset `i`.
    pub fn bset_entries(&self, i: usize) -> &[DiskEntry] {
        let off = bset_offset(self, i);
        let h = read_bset_header(self, off);
        entries_at(self, off, h.nkeys as usize)
    }

    /// Mutable view of bset `i`'s entries (used by sort-insert and compact).
    pub fn bset_entries_mut(&mut self, i: usize) -> &mut [DiskEntry] {
        let off = bset_offset(self, i);
        let h = read_bset_header(self, off);
        entries_at_mut(self, off, h.nkeys as usize)
    }

    /// Binary search bset `i` for `key`. Same semantics as the old
    /// single-array `search`: Ok(idx) = found, Err(idx) = insertion point.
    pub fn bset_search(&self, i: usize, key: &[u8]) -> std::result::Result<usize, usize> {
        self.bset_entries(i)
            .binary_search_by(|e| e.key_bytes().cmp(key))
    }

    /// Sequential scan over (bset_idx, entry_idx) for every valid entry,
    /// without sorting or dedup. Useful for compaction and verify.
    pub fn iter_all_entries(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.bset_count())
            .flat_map(move |b| (0..self.bset_entries(b).len()).map(move |i| (b, i)))
    }

    /// Resolve `(bset_idx, entry_idx)` to a `DiskEntry` reference.
    pub fn entry_at(&self, bset_idx: usize, entry_idx: usize) -> &DiskEntry {
        &self.bset_entries(bset_idx)[entry_idx]
    }

    /// Mutable variant of `entry_at`. Used for in-place edits to existing
    /// entries (overwrite a value/kind without inserting a new one).
    pub fn entry_at_mut(&mut self, bset_idx: usize, entry_idx: usize) -> &mut DiskEntry {
        &mut self.bset_entries_mut(bset_idx)[entry_idx]
    }

    /// Set the value at `(bset_idx, entry_idx)`. Leaf-only.
    pub fn set_value_at(&mut self, bset_idx: usize, entry_idx: usize, val: &[u8]) {
        debug_assert_eq!(self.level(), 0);
        let len = val.len().min(MAX_VALUE_SIZE);
        let entry = self.entry_at_mut(bset_idx, entry_idx);
        entry.payload.0[..len].copy_from_slice(&val[..len]);
        entry.payload.0[len..].fill(0);
        entry.value_len = len as u8;
    }

    /// Read value bytes at `(bset_idx, entry_idx)`. Leaf-only.
    pub fn value_bytes_at(&self, bset_idx: usize, entry_idx: usize) -> &[u8] {
        debug_assert_eq!(self.level(), 0);
        let entry = self.entry_at(bset_idx, entry_idx);
        let len = entry.value_len as usize;
        &entry.payload.0[..len]
    }

    /// Read child block number at `(bset_idx, entry_idx)`. Internal-only.
    pub fn child_block_at(&self, bset_idx: usize, entry_idx: usize) -> u64 {
        debug_assert!(self.level() > 0);
        self.entry_at(bset_idx, entry_idx).payload.child()
    }

    pub fn set_child_block_at(&mut self, bset_idx: usize, entry_idx: usize, nr: u64) {
        debug_assert!(self.level() > 0);
        self.entry_at_mut(bset_idx, entry_idx).payload.set_child(nr);
    }

    // ---------- Layout-mutating helpers ----------

    /// Append an empty bset with `seq` to the end of `body`. Caller writes
    /// entries via `append_to_last_bset`. Panics if `BSET_TREE_NR_MAX` is
    /// reached — the caller is expected to compact first.
    pub fn start_new_bset(&mut self, seq: u64) {
        let n = self.header.bset_count as usize;
        assert!(n < BSET_TREE_NR_MAX, "bset capacity already at max");
        // Compute the first free byte offset by walking existing bsets.
        let mut off = 0usize;
        for _ in 0..n {
            let h = read_bset_header(self, off);
            off += BSET_HEADER_SIZE + (h.nkeys as usize) * ENTRY_SIZE;
        }
        write_bset_header(
            self,
            off,
            &BsetHeader {
                seq,
                nkeys: 0,
                flags: 0,
                _pad: 0,
            },
        );
        self.header.bset_count = (n + 1) as u8;
    }

    /// Append `entry` to the latest bset in raw order (no sort enforcement
    /// on the caller's side — used by compact/split to build bsets that are
    /// already in sort order). Panics if there is no current bset or
    /// MAX_ENTRIES would be exceeded.
    pub fn append_to_last_bset(&mut self, entry: &DiskEntry) {
        let total = self.total_stored();
        assert!(total < MAX_ENTRIES, "node full: {total} entries");
        let bset_idx = self.bset_count();
        assert!(bset_idx > 0, "no current bset; call start_new_bset first");
        let bset_idx = bset_idx - 1;

        let off = bset_offset(self, bset_idx);
        let h = read_bset_header(self, off);
        let new_nkeys = (h.nkeys as usize) + 1;
        // Write the new entry just past the existing tail of the bset.
        let entry_off = off + BSET_HEADER_SIZE + (h.nkeys as usize) * ENTRY_SIZE;
        self.body[entry_off..entry_off + ENTRY_SIZE].copy_from_slice(entry.as_bytes());
        // Update bset header.
        write_bset_header(
            self,
            off,
            &BsetHeader {
                seq: h.seq,
                nkeys: new_nkeys as u16,
                flags: h.flags,
                _pad: 0,
            },
        );
    }

    /// Sort-insert `entry` into the latest bset. Returns:
    /// - `Ok(idx)` if a same-(logical, snap) entry existed and was overwritten;
    /// - `Err(idx)` if a new slot was opened at index `idx`.
    ///
    /// Used for leaf inserts. Internals don't go through this path because
    /// they're always single-bset and modifications rebuild the node.
    pub fn sort_insert_into_last_bset(
        &mut self,
        entry: &DiskEntry,
    ) -> std::result::Result<usize, usize> {
        let bset_cnt = self.bset_count();
        assert!(bset_cnt > 0, "no current bset; call start_new_bset first");
        let bset_idx = bset_cnt - 1;
        let off = bset_offset(self, bset_idx);
        let h = read_bset_header(self, off);

        match self.bset_search(bset_idx, entry.key_bytes()) {
            Ok(found) => {
                // Overwrite in place.
                let entries = entries_at_mut(self, off, h.nkeys as usize);
                entries[found] = *entry;
                Ok(found)
            }
            Err(insert_at) => {
                let total = self.total_stored();
                assert!(total < MAX_ENTRIES, "node full: {total} entries");
                let new_nkeys = (h.nkeys as usize) + 1;
                {
                    let entries = entries_at_mut(self, off, new_nkeys);
                    // Shift [insert_at .. new_nkeys - 1) right by one slot.
                    // copy_within uses ptr::copy (memmove semantics), safe for
                    // overlapping ranges — better than a reverse element-wise
                    // loop, which LLVM's loop-idiom pass doesn't reliably fold
                    // into a single memmove for non-trivial element types.
                    entries.copy_within(insert_at..new_nkeys - 1, insert_at + 1);
                    entries[insert_at] = *entry;
                }
                write_bset_header(
                    self,
                    off,
                    &BsetHeader {
                        seq: h.seq,
                        nkeys: new_nkeys as u16,
                        flags: h.flags,
                        _pad: 0,
                    },
                );
                Err(insert_at)
            }
        }
    }

    /// Search the *separator* keys of an internal node (single-bset).
    /// Internals store `nkeys + 1` entries in bset 0; the last entry has a
    /// degenerate key and only carries the rightmost child pointer. This
    /// helper restricts the binary search to the first `nkeys` entries.
    pub fn search_internal(&self, key: &[u8]) -> std::result::Result<usize, usize> {
        debug_assert!(self.level() > 0);
        debug_assert_eq!(self.bset_count(), 1);
        let entries = self.bset_entries(0);
        let n = entries.len().saturating_sub(1); // skip rightmost-child slot
        entries[..n].binary_search_by(|e| e.key_bytes().cmp(key))
    }

    // ---------- Single-bset convenience accessors ----------
    //
    // Two classes of node accesses are *structurally* single-bset and don't
    // need to think in (bset_idx, entry_idx) pairs:
    //
    //   1. Internal nodes. Every modification rebuilds them from scratch
    //      (clone-then-patch in `promote_to_parent` / `insert_internal`),
    //      so they always carry exactly one bset by construction.
    //   2. Freshly-built leaves out of `split_*` / compaction. They start
    //      life with one sorted bset; only the next insert may grow them
    //      into a multi-bset node.
    //
    // For both, callers are clearer when they don't have to thread a `0`
    // through every access. These wrappers forward `entry(i)` / `search(key)`
    // / `child_block(i)` to their `_at(0, ...)` counterparts and assert the
    // single-bset precondition in debug builds.
    //
    // Code that may operate on a multi-bset leaf must use the explicit
    // `_at` accessors or the merged iterator instead.

    /// Returns the i-th entry. Single-bset nodes only — see section comment.
    pub fn entry(&self, i: usize) -> &DiskEntry {
        debug_assert_eq!(
            self.bset_count(),
            1,
            "entry(i) only valid on single-bset nodes; got bset_count={}",
            self.bset_count()
        );
        self.entry_at(0, i)
    }

    /// Mutable variant of `entry`.
    pub fn entry_mut(&mut self, i: usize) -> &mut DiskEntry {
        debug_assert_eq!(self.bset_count(), 1);
        self.entry_at_mut(0, i)
    }

    /// Binary-search a single-bset node. Internals route to `search_internal`
    /// (skipping the rightmost-child slot); leaves run `bset_search` on bset 0.
    pub fn search(&self, key: &[u8]) -> std::result::Result<usize, usize> {
        debug_assert_eq!(
            self.bset_count(),
            1,
            "search() only valid on single-bset nodes; got bset_count={}",
            self.bset_count()
        );
        if self.level() > 0 {
            self.search_internal(key)
        } else {
            self.bset_search(0, key)
        }
    }

    pub fn child_block(&self, i: usize) -> u64 {
        debug_assert_eq!(self.bset_count(), 1);
        self.child_block_at(0, i)
    }

    pub fn set_child_block(&mut self, i: usize, nr: u64) {
        debug_assert_eq!(self.bset_count(), 1);
        self.set_child_block_at(0, i, nr);
    }

    pub fn value_bytes(&self, i: usize) -> &[u8] {
        debug_assert_eq!(self.bset_count(), 1);
        self.value_bytes_at(0, i)
    }

    pub fn set_value(&mut self, i: usize, val: &[u8]) {
        debug_assert_eq!(self.bset_count(), 1);
        self.set_value_at(0, i, val);
    }

    /// Commit the entry count for a freshly-built single-bset node.
    /// For leaves this is the live entry count; for internals the underlying
    /// bset stores `n + 1` entries (extra rightmost-child slot). Used by
    /// build-from-scratch paths (split / promote) that pre-write entries
    /// into the body and then publish the count in one shot.
    pub fn set_nkeys(&mut self, n: usize) {
        debug_assert_eq!(self.bset_count(), 1, "set_nkeys requires bset_count==1");
        let stored = if self.level() > 0 { n + 1 } else { n };
        let off = 0usize;
        let h = read_bset_header(self, off);
        write_bset_header(
            self,
            off,
            &BsetHeader {
                seq: h.seq,
                nkeys: stored as u16,
                flags: h.flags,
                _pad: 0,
            },
        );
    }
}

// ---------- Merged search / iteration ----------
//
// Multi-bset reads need to consider every bset and pick the highest-seq
// entry for any (sortable) key that appears more than once.

/// Result of a cross-bset search.
#[derive(Debug, Clone, Copy)]
pub struct MergedHit {
    pub bset_idx: usize,
    pub entry_idx: usize,
    pub seq: u64,
}

/// Find the highest-seq entry whose sortable key equals `key`, across all
/// bsets. Returns `None` if no bset contains the key.
pub fn merged_find(node: &BtreeNodeRaw, key: &[u8]) -> Option<MergedHit> {
    let mut best: Option<MergedHit> = None;
    for b in 0..node.bset_count() {
        if let Ok(idx) = node.bset_search(b, key) {
            let seq = node.bset_header(b).seq;
            match best {
                Some(h) if h.seq >= seq => {}
                _ => {
                    best = Some(MergedHit {
                        bset_idx: b,
                        entry_idx: idx,
                        seq,
                    })
                }
            }
        }
    }
    best
}

/// Per-bset lower-bound positions for `key` — used to seed range-scan
/// cursors in the merged iterator.
fn merged_lower_bound_cursors(node: &BtreeNodeRaw, key: &[u8]) -> [usize; BSET_TREE_NR_MAX] {
    let mut out = [0usize; BSET_TREE_NR_MAX];
    for (b, slot) in out.iter_mut().enumerate().take(node.bset_count()) {
        *slot = match node.bset_search(b, key) {
            Ok(idx) => idx,
            Err(idx) => idx,
        };
    }
    out
}

/// k-way merged iterator across all bsets of a node. Emits `(bset_idx,
/// entry_idx)` pairs in ascending sortable-key order. When the same key
/// appears in multiple bsets, only the highest-seq instance is emitted;
/// the others are silently skipped.
pub struct MergedIter<'a> {
    node: &'a BtreeNodeRaw,
    cursors: [usize; BSET_TREE_NR_MAX],
    lens: [usize; BSET_TREE_NR_MAX],
    seqs: [u64; BSET_TREE_NR_MAX],
    bset_count: usize,
}

impl<'a> MergedIter<'a> {
    pub fn new(node: &'a BtreeNodeRaw) -> Self {
        let mut lens = [0usize; BSET_TREE_NR_MAX];
        let mut seqs = [0u64; BSET_TREE_NR_MAX];
        for b in 0..node.bset_count() {
            let h = node.bset_header(b);
            lens[b] = h.nkeys as usize;
            seqs[b] = h.seq;
        }
        MergedIter {
            node,
            cursors: [0; BSET_TREE_NR_MAX],
            lens,
            seqs,
            bset_count: node.bset_count(),
        }
    }

    pub fn with_lower_bound(node: &'a BtreeNodeRaw, key: &[u8]) -> Self {
        let mut iter = Self::new(node);
        iter.cursors = merged_lower_bound_cursors(node, key);
        iter
    }
}

impl<'a> Iterator for MergedIter<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        // Find the smallest-key cursor (tie-break: higher seq wins).
        // Both `best_key` and per-iteration `key` borrow from `self.node`,
        // not `self`, so we can hold them across `self.cursors[b] += 1`
        // mutations later.
        let mut best: Option<(usize, u64)> = None;
        let mut best_key: Option<&'a [u8]> = None;
        for b in 0..self.bset_count {
            if self.cursors[b] >= self.lens[b] {
                continue;
            }
            let key: &'a [u8] = self.node.bset_entries(b)[self.cursors[b]].key_bytes();
            match best_key {
                None => {
                    best = Some((b, self.seqs[b]));
                    best_key = Some(key);
                }
                Some(bk) if key < bk => {
                    best = Some((b, self.seqs[b]));
                    best_key = Some(key);
                }
                Some(bk) if key == bk && self.seqs[b] > best.unwrap().1 => {
                    best = Some((b, self.seqs[b]));
                    best_key = Some(key);
                }
                _ => {}
            }
        }
        let (best_b, _) = best?;
        let best_idx = self.cursors[best_b];
        let bkey: &'a [u8] = best_key.expect("best implies best_key set");

        // Advance every cursor whose current key equals `bkey`. Lower-seq
        // duplicates are silently dropped.
        for b in 0..self.bset_count {
            if self.cursors[b] >= self.lens[b] {
                continue;
            }
            let key: &'a [u8] = self.node.bset_entries(b)[self.cursors[b]].key_bytes();
            if key == bkey {
                self.cursors[b] += 1;
            }
        }
        Some((best_b, best_idx))
    }
}
