use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

const BLOCK_SIZE: usize = 4096;
pub const MAGIC_NUMBER: u32 = 0x39C5BB39;

pub const MAX_KEY_SIZE: usize = 32;
pub const MAX_VALUE_SIZE: usize = 96;

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

#[repr(C)]
#[derive(KnownLayout, Immutable, IntoBytes, FromBytes, Clone, Copy)]
pub struct DiskEntry {
    pub key: [u8; MAX_KEY_SIZE],
    pub key_len: u8,
    /// Only meaningful in leaf nodes; ignored in internal nodes.
    pub value_len: u8,
    /// Aligns `payload` to 8 bytes so the child field is naturally aligned.
    _pad: [u8; 6],
    pub payload: Payload,
}

const _: () = assert!(std::mem::size_of::<DiskEntry>() == MAX_KEY_SIZE + 1 + 7 + MAX_VALUE_SIZE); // 136

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
    pub fn key_bytes(&self) -> &[u8] {
        &self.key[..self.key_len as usize]
    }

    pub fn set_key(&mut self, bytes: &[u8]) {
        let len = bytes.len().min(MAX_KEY_SIZE);
        self.key[..len].copy_from_slice(&bytes[..len]);
        self.key[len..].fill(0);
        self.key_len = len as u8;
    }

    pub fn empty() -> Self {
        DiskEntry {
            key: [0u8; MAX_KEY_SIZE],
            key_len: 0,
            value_len: 0,
            _pad: [0u8; 6],
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
