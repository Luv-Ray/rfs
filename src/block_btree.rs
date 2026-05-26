use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

const BLOCK_SIZE: usize = 4096;
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

/// Per-entry type tag. Stored in `DiskEntry::kind`.
///
/// - `Live`: normal key/value.
/// - `Deleted`: trivial tombstone (KEY_TYPE_deleted in bcachefs). Produced
///   when X deletes a key it itself wrote at snap_id == X. Compaction may
///   drop these unconditionally.
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
    _pad0: [u8; 4],
    /// Incremented on each transaction commit; used for crash recovery.
    pub generation: u64,
    pub nkeys: u16,
    /// 0 = leaf, >0 = internal (value = distance from leaves).
    pub level: u8,
    /// Reserved for future use (e.g. compression, checksum type).
    pub flags: u8,
    _pad1: [u8; 4],
    /// CRC-64 of the entire node (0 = not yet computed).
    pub checksum: u64,
    /// Pads header to 64 bytes; available for future fields.
    _reserved: [u8; 32],
}

const _: () = assert!(std::mem::size_of::<NodeHeader>() == 64);

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
//
// Layout of `key` buffer:
//   bytes 0..key_len           = logical key (variable length, up to MAX_LOGICAL_KEY_SIZE)
//   bytes key_len..key_len+4   = snap_id, big-endian (always present)
//   bytes key_len+4..MAX_KEY_SIZE = unused, must be 0
//
// The "sortable key" returned by `key_bytes()` is the contiguous slice
// `&key[..key_len + 4]` — logical bytes followed immediately by snap_id_be.
// Comparing two sortable keys lexicographically gives the (logical, snap_id)
// ordering: smaller logical first, ties broken by smaller snap_id.

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
const BODY_SPACE: usize = BLOCK_SIZE - HEADER_SIZE;
pub const MAX_ENTRIES: usize = BODY_SPACE / ENTRY_SIZE; // 29
/// Internal nodes store nkeys+1 child pointers in entries[0..=nkeys].
/// The split-and-insert shift loop writes child[i+2], requiring
/// nkeys+2 <= MAX_ENTRIES, hence nkeys <= MAX_ENTRIES - 2.
pub const MAX_INTERNAL_KEYS: usize = MAX_ENTRIES - 2; // 27
const REMAINING: usize = BODY_SPACE - (MAX_ENTRIES * ENTRY_SIZE);

// ---------- B-tree node (on-disk layout, one block) ----------

#[repr(C, align(4096))]
#[derive(KnownLayout, Immutable, IntoBytes, FromBytes, Clone)]
pub struct BtreeNodeRaw {
    pub header: NodeHeader,
    /// Only entries[0..header.nkeys] are valid.
    pub entries: [DiskEntry; MAX_ENTRIES],
    pub _padding: [u8; REMAINING],
}

const _: () = assert!(std::mem::size_of::<BtreeNodeRaw>() == BLOCK_SIZE);

// ---------- Methods ----------

impl DiskEntry {
    /// Logical key bytes only (no snap_id appended). This is what callers
    /// outside the btree see.
    pub fn logical_key_bytes(&self) -> &[u8] {
        &self.key[..self.key_len as usize]
    }

    /// Full sortable key bytes: logical || snap_id_be. This is what
    /// `BtreeNodeRaw::search` and all internal comparisons use.
    /// Length is `key_len + 4`.
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

impl BtreeNodeRaw {
    pub fn new(level: u8) -> Self {
        BtreeNodeRaw {
            header: NodeHeader {
                magic: MAGIC_NUMBER,
                _pad0: Default::default(),
                generation: 0,
                nkeys: 0,
                level,
                flags: 0,
                _pad1: Default::default(),
                checksum: 0,
                _reserved: Default::default(),
            },
            entries: std::array::from_fn(|_| DiskEntry::empty()),
            _padding: [0u8; REMAINING],
        }
    }

    pub fn nkeys(&self) -> usize {
        self.header.nkeys as usize
    }

    pub fn set_nkeys(&mut self, n: usize) {
        self.header.nkeys = n as u16;
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

    pub fn entry(&self, i: usize) -> &DiskEntry {
        &self.entries[i]
    }

    pub fn entry_mut(&mut self, i: usize) -> &mut DiskEntry {
        &mut self.entries[i]
    }

    /// Child block pointer at position i (internal nodes have nkeys+1 children).
    pub fn child_block(&self, i: usize) -> u64 {
        debug_assert!(self.header.level > 0);
        self.entries[i].payload.child()
    }

    pub fn set_child_block(&mut self, i: usize, nr: u64) {
        debug_assert!(self.header.level > 0);
        self.entries[i].payload.set_child(nr);
    }

    /// Value bytes at position i, truncated to value_len.
    pub fn value_bytes(&self, i: usize) -> &[u8] {
        debug_assert!(self.header.level == 0);
        let len = self.entries[i].value_len as usize;
        &self.entries[i].payload.0[..len]
    }

    pub fn set_value(&mut self, i: usize, val: &[u8]) {
        debug_assert!(self.header.level == 0);
        let len = val.len().min(MAX_VALUE_SIZE);
        self.entries[i].payload.0[..len].copy_from_slice(&val[..len]);
        self.entries[i].payload.0[len..].fill(0);
        self.entries[i].value_len = len as u8;
    }

    /// Binary search among entries[0..nkeys]. Ok(idx) = found, Err(idx) = insertion point.
    pub fn search(&self, key: &[u8]) -> Result<usize, usize> {
        let n = self.nkeys();
        self.entries[..n].binary_search_by(|e| e.key_bytes().cmp(key))
    }
}
