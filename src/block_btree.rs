// Block size constant
const BLOCK_SIZE: usize = 4096;

// Max key/value sizes (this version inlines all data)
const MAX_KEY_SIZE: usize = 32;
const MAX_VALUE_SIZE: usize = 96;

// ---------- Node header ----------
#[repr(C)]
pub struct NodeHeader {
    /// Magic number for quick block type validation
    pub magic: u32,
    /// Generation number, incremented on each transaction commit, used for crash recovery
    pub generation: u64,
    /// Number of valid entries in this node
    pub nkeys: u16,
    /// Tree level: 0 -> leaf node, >0 -> internal node
    pub level: u8,
    /// Flags (reserved)
    pub flags: u8,
    /// Checksum of the entire node (0 means not yet computed)
    pub checksum: u64,
    // Padding to 64 bytes for future extensibility
    _reserved: [u8; 32],
}

// Compile-time assertion on header size to ensure manual padding is correct
const _: () = assert!(std::mem::size_of::<NodeHeader>() == 64);

// ---------- Entry payload (leaf value or child node pointer) ----------
#[repr(C)]
pub union EntryPayload {
    /// Leaf node data value (only valid when level == 0)
    value: [u8; MAX_VALUE_SIZE],
    /// Internal node child block number (only valid when level > 0)
    child: u64,
}

// ---------- Disk entry ----------
#[repr(C)]
pub struct DiskEntry {
    /// Raw bytes of the key (up to MAX_KEY_SIZE bytes)
    pub key: [u8; MAX_KEY_SIZE],
    /// Actual byte length of the key (<= MAX_KEY_SIZE)
    pub key_len: u8,
    /// Alignment padding to ensure payload's child field is 8-byte aligned
    _pad: [u8; 7],
    /// Entry payload (interpreted as value or child pointer based on node level)
    pub payload: EntryPayload,
}

// Ensure DiskEntry size matches our expectations
const _: () = assert!(std::mem::size_of::<DiskEntry>() == MAX_KEY_SIZE + 1 + 7 + MAX_VALUE_SIZE); // 32+1+7+96 = 136

// ---------- B+ tree node (disk layout) ----------
const ENTRY_SIZE: usize = std::mem::size_of::<DiskEntry>(); // 136
const HEADER_SIZE: usize = std::mem::size_of::<NodeHeader>(); // 64
const BODY_SPACE: usize = BLOCK_SIZE - HEADER_SIZE; // 4096 - 64 = 4032
const MAX_ENTRIES: usize = BODY_SPACE / ENTRY_SIZE; // 4032 / 136 = 29
const REMAINING: usize = BODY_SPACE - (MAX_ENTRIES * ENTRY_SIZE); // 4032 - 29*136 = 4032 - 3944 = 88

#[repr(C, align(4096))]
pub struct BtreeNodeRaw {
    /// Node header (contains level, generation, checksum, etc.)
    pub header: NodeHeader,
    /// Fixed-size entry array; only the first header.nkeys entries are valid
    pub entries: [DiskEntry; MAX_ENTRIES],
    /// Tail padding so the entire struct size equals BLOCK_SIZE exactly
    pub _padding: [u8; REMAINING],
}

// Critical assertion: confirm BtreeNodeRaw size matches block size exactly
const _: () = assert!(std::mem::size_of::<BtreeNodeRaw>() == BLOCK_SIZE);

// Optional: if using bytemuck for zero-copy read/write, safe traits must be implemented.
// Because this struct contains a union, a custom unsafe impl is typically needed.
// Placeholder provided below.
//
// unsafe impl bytemuck::Pod for BtreeNodeRaw {}
// unsafe impl bytemuck::Zeroable for BtreeNodeRaw {}
