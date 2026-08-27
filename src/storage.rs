//! Block storage layer: backs btree nodes and file data blocks with either
//! pure RAM or a single image file (pread/pwrite).
//!
//! Design notes:
//! - Cache is a **dirty-tracked mutable map** (`RwLock<HashMap<u64, Box<..>>>`).
//!   The node cache is accessed only through the `with_node` closure (read)
//!   entry point, never handed out as a long-lived borrow — so the lock is
//!   released as soon as the closure returns and callers recurse without
//!   holding it. See docs/node-cache-rewrite-plan.md.
//!   Note: COW-once (a given `block_nr` is written at most once after
//!   allocation) is still upheld today, but it is now a *convention* enforced
//!   by funnelling all mutation through `write_node`, not a static guarantee
//!   of the cache type.
//! - Allocator is unified: btree nodes and file data blocks share one
//!   monotonically-increasing `next_block_nr`. Block 0 is reserved for the
//!   superblock; blocks 1..64 are the journal ring; node blocks have a
//!   `MAGIC_NUMBER` + CRC at the head and data blocks are raw 4 KB payloads.
//! - Journal: append-only ring buffer providing crash recovery — see journal.rs.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

use zerocopy::{FromBytes, IntoBytes, KnownLayout};

use crate::block_btree::{BLOCK_SIZE, BtreeNodeRaw, MAGIC_NUMBER};
use crate::btree::{Error, Result};

/// Block 0 is always the superblock.
pub const SUPERBLOCK_BLOCK_NR: u64 = 0;

/// Journal occupies blocks 1..64 (inclusive). Each 4 KB block holds
/// `ENTRIES_PER_BLOCK` fixed-size 256-byte frames (16 × 256 = 4096, no
/// padding). A frame is either a logged operation or a commit-end record;
/// see [`JournalFrame`].
pub const JOURNAL_MAGIC: u32 = 0x524A_4E4C; // "RJNL"
pub const JOURNAL_BLOCKS: u64 = 64;
pub const FIRST_JOURNAL_BLOCK: u64 = 1;
pub const ENTRIES_PER_BLOCK: usize = 16;
pub const JOURNAL_CAPACITY: u64 = JOURNAL_BLOCKS * ENTRIES_PER_BLOCK as u64; // 1024

/// First block number that may hold a node or data block. Block 0 is the
/// superblock, blocks 1..64 are the journal ring buffer.
pub const FIRST_DATA_BLOCK_NR: u64 = FIRST_JOURNAL_BLOCK + JOURNAL_BLOCKS; // 65

/// Magic number stamped at the head of the superblock.
pub const SUPERBLOCK_MAGIC: u32 = 0x5246_5342; // "RFSB"
/// On-disk format version; bumped on incompatible layout changes.
/// v4: added `free_head` (on-disk free-list chain).
pub const SUPERBLOCK_VERSION: u32 = 4;

/// Superblock — the single source of truth for "where the live tree is".
///
/// Written at block 0. CRC covers every byte except `checksum` itself.
/// On open we verify magic + version + checksum; if any fails we refuse to
/// mount rather than silently load a corrupt root.
///
/// Layout (64 bytes of named fields, no implicit padding):
///   magic:4  version:4  root_block:8  next_block_nr:8  next_bset_seq:8
///   next_ino:8  journal_seq:8  next_snap_id:4  next_subvol_id:4
///   current_subvol:4  checksum:4  _reserved:4032
#[repr(C)]
#[derive(KnownLayout, zerocopy::Immutable, IntoBytes, FromBytes, Clone, Copy)]
pub struct Superblock {
    pub magic: u32,
    pub version: u32,
    /// btree root block at the time of the last sync.
    pub root_block: u64,
    /// First unused block number; allocator hands these out monotonically.
    pub next_block_nr: u64,
    /// Next bset seq the btree will hand out to a freshly opened bset.
    pub next_bset_seq: u64,
    /// First inode number not yet handed out by `Fs::alloc_ino`.
    pub next_ino: u64,
    /// Sequence number of the last journal entry that was checkpointed into
    /// this superblock. 0 means no journal entries have been checkpointed.
    pub journal_seq: u64,
    /// Smallest snap_id allocated so far minus one (snap ids count down).
    pub next_snap_id: u32,
    /// Smallest subvol id not yet used (count up).
    pub next_subvol_id: u32,
    /// Currently active subvolume id.
    pub current_subvol: u32,
    /// CRC32 over the rest of the superblock (computed with this field == 0).
    pub checksum: u32,
    /// Block number of the head of the on-disk free-list chain, or 0 if empty.
    pub free_head: u64,
    pub _reserved: [u8; BLOCK_SIZE - 72],
}

const _: () = assert!(std::mem::size_of::<Superblock>() == BLOCK_SIZE);

impl Superblock {
    /// Build a fresh superblock for a brand-new image.
    pub fn fresh(root_block: u64, next_block_nr: u64) -> Self {
        Superblock {
            magic: SUPERBLOCK_MAGIC,
            version: SUPERBLOCK_VERSION,
            root_block,
            next_block_nr,
            next_bset_seq: 1,
            next_ino: 0,
            journal_seq: 0,
            next_snap_id: u32::MAX - 1,
            next_subvol_id: 1,
            current_subvol: 0,
            checksum: 0,
            free_head: 0,
            _reserved: [0; BLOCK_SIZE - 72],
        }
    }

    /// Compute the CRC over every field except `checksum`.
    fn compute_checksum(&self) -> u32 {
        let mut copy = *self;
        copy.checksum = 0;
        crc32fast::hash(copy.as_bytes())
    }

    /// Verify magic, version, and CRC. Returns the parsed superblock on
    /// success, or a structured error pointing at block 0.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        // `read_from_bytes` copies into an owned value, so it has no alignment
        // requirement on `bytes`. `ref_from_bytes` would return a reference
        // *into* the buffer and thus require 8-byte alignment (Superblock has
        // u64 fields); callers hand us a `Vec<u8>` / `&[u8]` with no alignment
        // guarantee, so we must copy. (Miri flags the ref_from_bytes form.)
        let sb = Superblock::read_from_bytes(bytes)
            .map_err(|_| Error::Io(io::Error::other("superblock size mismatch")))?;
        if sb.magic != SUPERBLOCK_MAGIC {
            return Err(Error::BadMagic {
                block: SUPERBLOCK_BLOCK_NR,
                got: sb.magic,
                expected: SUPERBLOCK_MAGIC,
            });
        }
        if sb.version != SUPERBLOCK_VERSION {
            return Err(Error::Io(io::Error::other(format!(
                "unsupported superblock version {} (expected {})",
                sb.version, SUPERBLOCK_VERSION
            ))));
        }
        let want = sb.compute_checksum();
        if sb.checksum != want {
            return Err(Error::ChecksumMismatch {
                block: SUPERBLOCK_BLOCK_NR,
            });
        }
        Ok(sb)
    }

    /// Render the superblock to its 4 KB on-disk form, stamping the CRC.
    pub fn to_bytes(mut self) -> [u8; BLOCK_SIZE] {
        self.checksum = 0;
        self.checksum = self.compute_checksum();
        let mut out = [0u8; BLOCK_SIZE];
        out.copy_from_slice(self.as_bytes());
        out
    }
}

/// Size of one on-disk journal frame. 16 frames fill a 4 KB block exactly.
pub const JOURNAL_FRAME_SIZE: usize = 256;

/// Bytes of a frame available for a logged-operation payload, after the
/// fixed 24-byte header + 48 bytes of commit-end state fields
/// (24 + 48 + 184 = 256).
pub const JOURNAL_OP_CAPACITY: usize = 184;

/// What a [`JournalFrame`] carries.
///
/// The journal is a sequence of frames grouped into atomic *commit groups*.
/// A group is zero or more `LoggedOp` frames followed by exactly one
/// `CommitEnd` frame. Replay only applies a group once it has seen the
/// group's `CommitEnd`; a trailing group with no `CommitEnd` (a crash mid
/// commit) is discarded.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameKind {
    /// One logged high-level operation; payload in `op_kind` + `op_data`.
    LoggedOp = 1,
    /// Closes a commit group and records the resulting fs state scalars.
    CommitEnd = 2,
}

impl FrameKind {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(FrameKind::LoggedOp),
            2 => Some(FrameKind::CommitEnd),
            _ => None,
        }
    }
}

/// A single fixed-size journal frame (256 bytes). `magic` + `checksum` guard
/// against torn writes; `seq` is a monotonically-increasing per-frame counter
/// the scanner uses to walk the ring and detect wraparound.
///
/// Layout: 32-byte header (magic, checksum, seq, frame_kind, op_kind, op_len)
/// + 48 bytes of commit-end state scalars + `JOURNAL_OP_CAPACITY` op payload.
#[repr(C)]
#[derive(KnownLayout, zerocopy::Immutable, IntoBytes, FromBytes, Clone, Copy)]
pub struct JournalFrame {
    pub magic: u32,
    pub checksum: u32,
    pub seq: u64,
    /// `FrameKind` discriminant.
    pub frame_kind: u8,
    /// For `LoggedOp`: the logged-operation opcode (see fs.rs). 0 otherwise.
    pub op_kind: u8,
    /// For `LoggedOp`: valid byte length of `op_data`. 0 otherwise.
    pub op_len: u16,
    _pad0: [u8; 4],
    // ---- commit-end state (meaningful only when frame_kind == CommitEnd) ----
    pub root_block: u64,
    pub next_block_nr: u64,
    pub next_bset_seq: u64,
    pub next_ino: u64,
    pub next_snap_id: u32,
    pub next_subvol_id: u32,
    pub current_subvol: u32,
    _pad1: u32,
    // ---- logged-op payload (meaningful only when frame_kind == LoggedOp) ----
    pub op_data: [u8; JOURNAL_OP_CAPACITY],
}

const _: () = assert!(std::mem::size_of::<JournalFrame>() == JOURNAL_FRAME_SIZE);
const _: () = assert!(JOURNAL_FRAME_SIZE * ENTRIES_PER_BLOCK == BLOCK_SIZE);

impl JournalFrame {
    /// A zeroed frame with magic set and the given seq + kind.
    fn new(seq: u64, kind: FrameKind) -> Self {
        JournalFrame {
            magic: JOURNAL_MAGIC,
            checksum: 0,
            seq,
            frame_kind: kind as u8,
            op_kind: 0,
            op_len: 0,
            _pad0: [0; 4],
            root_block: 0,
            next_block_nr: 0,
            next_bset_seq: 0,
            next_ino: 0,
            next_snap_id: 0,
            next_subvol_id: 0,
            current_subvol: 0,
            _pad1: 0,
            op_data: [0; JOURNAL_OP_CAPACITY],
        }
    }

    /// Build a `LoggedOp` frame carrying `op_kind` + `data` (must fit in
    /// `JOURNAL_OP_CAPACITY`).
    pub fn logged_op(seq: u64, op_kind: u8, data: &[u8]) -> Self {
        assert!(
            data.len() <= JOURNAL_OP_CAPACITY,
            "logged op payload {} > {JOURNAL_OP_CAPACITY}",
            data.len()
        );
        let mut f = Self::new(seq, FrameKind::LoggedOp);
        f.op_kind = op_kind;
        f.op_len = data.len() as u16;
        f.op_data[..data.len()].copy_from_slice(data);
        f
    }

    /// Build a `CommitEnd` frame recording the fs state scalars.
    #[allow(clippy::too_many_arguments)]
    pub fn commit_end(
        seq: u64,
        root_block: u64,
        next_block_nr: u64,
        next_bset_seq: u64,
        next_ino: u64,
        next_snap_id: u32,
        next_subvol_id: u32,
        current_subvol: u32,
    ) -> Self {
        let mut f = Self::new(seq, FrameKind::CommitEnd);
        f.root_block = root_block;
        f.next_block_nr = next_block_nr;
        f.next_bset_seq = next_bset_seq;
        f.next_ino = next_ino;
        f.next_snap_id = next_snap_id;
        f.next_subvol_id = next_subvol_id;
        f.current_subvol = current_subvol;
        f
    }

    pub fn kind(&self) -> Option<FrameKind> {
        FrameKind::from_u8(self.frame_kind)
    }

    /// The logged-op payload slice (only meaningful for `LoggedOp` frames).
    pub fn op_payload(&self) -> &[u8] {
        &self.op_data[..self.op_len as usize]
    }

    /// Compute a CRC32 over the frame with `checksum` treated as zero.
    pub fn compute_checksum(&self) -> u32 {
        let mut copy = *self;
        copy.checksum = 0;
        crc32fast::hash(copy.as_bytes())
    }

    /// Return `true` iff magic, seq, kind, and checksum all pass.
    pub fn is_valid(&self, expected_seq: u64) -> bool {
        self.magic == JOURNAL_MAGIC
            && self.seq == expected_seq
            && self.kind().is_some()
            && self.checksum == self.compute_checksum()
    }
}

/// Aligned 4 KB raw payload for non-node blocks (file data, future
/// on-disk structures that don't need a magic+CRC header). Same alignment
/// as `BtreeNodeRaw` so the cache always hands out properly-aligned bytes.
#[repr(C, align(4096))]
#[derive(KnownLayout, zerocopy::Immutable, IntoBytes, FromBytes, Clone, Copy)]
pub struct DataBlock(pub [u8; BLOCK_SIZE]);

impl DataBlock {
    pub fn zeroed() -> Self {
        DataBlock([0u8; BLOCK_SIZE])
    }
}

const _: () = assert!(std::mem::size_of::<DataBlock>() == BLOCK_SIZE);
const _: () = assert!(std::mem::align_of::<DataBlock>() == BLOCK_SIZE);

/// A fixed-offset block device: the single IO abstraction under `BlockStore`
/// and `Journal`. Both the real image file ([`FileDevice`]) and the in-RAM
/// simulation ([`MemDevice`]) implement it, so every read/write path is
/// identical regardless of backing — there is no `match` on the backing kind.
///
/// Offsets are absolute byte offsets into the device. Implementations model a
/// zeroed, sparse address space: a `read_at` of a region never written returns
/// zeros rather than erroring (`FileDevice` relies on the file being
/// `set_len`-extended and zero-filled to the same effect). Correct COW callers
/// never read a block before writing it, so the two devices are
/// indistinguishable in normal operation.
pub trait BlockDevice: Send + Sync {
    /// Fill `buf` from the device starting at `offset`.
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()>;
    /// Write all of `buf` to the device starting at `offset`.
    fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<()>;
    /// Flush durably (data + as needed metadata). No-op for volatile devices.
    fn sync(&self) -> io::Result<()>;
    /// Resize the device, zero-filling any growth.
    fn set_len(&self, len: u64) -> io::Result<()>;
}

/// Image-file device: pread/pwrite/fdatasync against a single backing file.
pub struct FileDevice {
    file: File,
}

impl FileDevice {
    pub fn new(file: File) -> Self {
        FileDevice { file }
    }
}

impl BlockDevice for FileDevice {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        self.file.read_exact_at(buf, offset)
    }
    fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<()> {
        self.file.write_all_at(buf, offset)
    }
    fn sync(&self) -> io::Result<()> {
        self.file.sync_data()
    }
    fn set_len(&self, len: u64) -> io::Result<()> {
        self.file.set_len(len)
    }
}

/// In-RAM simulated device: a single growable, zero-filled byte vector. Used by
/// pure-RAM filesystems and tests so they exercise the same IO paths as an
/// image without touching disk. `sync` is a no-op (volatile); `read_at` past
/// the written region returns zeros; `write_at` auto-grows the backing vector.
#[derive(Default)]
pub struct MemDevice {
    data: RwLock<Vec<u8>>,
}

impl MemDevice {
    pub fn new() -> Self {
        MemDevice::default()
    }
}

impl BlockDevice for MemDevice {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        // Zeroed sparse semantics: fill with zeros, then overlay whatever bytes
        // have actually been written into this range.
        buf.fill(0);
        let data = self.data.read().unwrap();
        let start = offset as usize;
        if start < data.len() {
            let end = (start + buf.len()).min(data.len());
            buf[..end - start].copy_from_slice(&data[start..end]);
        }
        Ok(())
    }
    fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<()> {
        let mut data = self.data.write().unwrap();
        let end = offset as usize + buf.len();
        if data.len() < end {
            data.resize(end, 0);
        }
        data[offset as usize..end].copy_from_slice(buf);
        Ok(())
    }
    fn sync(&self) -> io::Result<()> {
        Ok(())
    }
    fn set_len(&self, len: u64) -> io::Result<()> {
        self.data.write().unwrap().resize(len as usize, 0);
        Ok(())
    }
}

/// A cached btree node plus its dirty bit.
///
/// `dirty` is set when the in-cache copy is newer than what is on disk. In
/// the current write-through mode `write_node` clears it immediately (it
/// pwrites before returning), but the field is in place for the upcoming
/// write-back / incremental-flush step.
struct CachedNode {
    node: BtreeNodeRaw,
    dirty: bool,
}

/// Block cache + optional image-file backing.
///
/// Mutation is funnelled through:
/// - `AtomicU64` for the allocator counter
/// - `RwLock<HashMap>` for the caches
/// - `File`'s own `&self` `read_at` / `write_at` (Unix `FileExt`)
///
/// The node cache is never handed out as a borrow: reads go through
/// `with_node(nr, |node| ...)`, which takes the read lock, runs the closure,
/// and drops the lock before returning — so recursive tree walks release the
/// lock between levels (they extract the child block number inside the
/// closure and recurse outside it).
///
/// Two cache lanes are kept side by side because btree nodes need
/// 4 KB alignment to be cast back to `&BtreeNodeRaw` (zerocopy), while
/// raw data blocks are addressed as plain bytes. A given block_nr is
/// always read/written through one lane consistently — caller picks via
/// `with_node`/`write_node` vs `read_data`/`write_data`. They share
/// `next_block_nr`, so there is one global block-number space.
pub struct BlockStore {
    node_cache: RwLock<HashMap<u64, Box<CachedNode>>>,
    data_cache: RwLock<HashMap<u64, Box<DataBlock>>>,
    next_block_nr: AtomicU64,
    free_list: Mutex<Vec<u64>>,
    /// Chain blocks written by the last `persist_free_list` call. Recycled
    /// at the start of the next persist so they don't leak across syncs.
    last_chain_blocks: Mutex<Vec<u64>>,
    /// The block device every read/write goes through — a real image file
    /// ([`FileDevice`]) or the in-RAM simulation ([`MemDevice`]).
    device: Arc<dyn BlockDevice>,
    /// Whether this store is durable. RAM-backed stores set this false so
    /// `Fs` can skip sync/gc/checkpoint work that only matters on disk, even
    /// though the IO path itself is device-uniform.
    persistent: bool,
    /// Test-only counter of node persists (`write_node` calls). Used to assert
    /// that in-place append collapses many hot-leaf writes into a handful of
    /// node persists per checkpoint instead of one COW rewrite per op.
    #[cfg(test)]
    node_writes: AtomicU64,
}

impl BlockStore {
    /// Assemble a store over an already-constructed device.
    fn with_device(device: Arc<dyn BlockDevice>, persistent: bool) -> Self {
        BlockStore {
            node_cache: RwLock::new(HashMap::new()),
            data_cache: RwLock::new(HashMap::new()),
            next_block_nr: AtomicU64::new(FIRST_DATA_BLOCK_NR),
            free_list: Mutex::new(Vec::new()),
            last_chain_blocks: Mutex::new(Vec::new()),
            device,
            persistent,
            #[cfg(test)]
            node_writes: AtomicU64::new(0),
        }
    }

    /// Pure-RAM store backed by a [`MemDevice`]. Marked non-persistent so `Fs`
    /// skips disk-only work (sync/gc/checkpoint), but every IO still flows
    /// through the same device path as an image.
    pub fn in_memory() -> Self {
        Self::with_device(Arc::new(MemDevice::new()), false)
    }

    /// Create a brand-new image file at `path` and return a store backed by
    /// it. Fails if the file already exists.
    pub fn create_image(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)?;
        Ok(Self::with_device(Arc::new(FileDevice::new(file)), true))
    }

    /// Open an existing image file. Caller is responsible for then loading
    /// the superblock and seeding `next_block_nr`.
    pub fn open_image(path: &Path) -> Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        Ok(Self::with_device(Arc::new(FileDevice::new(file)), true))
    }

    /// Whether this store is durable (image-backed). RAM stores return false.
    pub fn is_persistent(&self) -> bool {
        self.persistent
    }

    /// Clone the underlying device handle (a cheap `Arc` bump). Used by
    /// `Journal`, which shares the same device as the store.
    pub fn device(&self) -> Arc<dyn BlockDevice> {
        Arc::clone(&self.device)
    }

    /// Allocate a block number. Reuses a previously freed block if available,
    /// otherwise bumps the high-water mark.
    pub fn alloc(&self) -> u64 {
        if let Some(nr) = self.free_list.lock().unwrap().pop() {
            return nr;
        }
        self.next_block_nr.fetch_add(1, Ordering::Relaxed)
    }

    /// Return a block to the free list for future reuse. Evicts the block
    /// from both caches so stale data is never served.
    pub fn free(&self, nr: u64) {
        self.node_cache.write().unwrap().remove(&nr);
        self.data_cache.write().unwrap().remove(&nr);
        self.free_list.lock().unwrap().push(nr);
    }

    /// Number of blocks on the free list (for statfs reporting).
    pub fn free_count(&self) -> u64 {
        self.free_list.lock().unwrap().len() as u64
    }

    /// Snapshot of the block numbers currently on the free list. Used by GC to
    /// mark already-free blocks as live so the sweep doesn't double-free them.
    pub fn free_list_snapshot(&self) -> Vec<u64> {
        self.free_list.lock().unwrap().clone()
    }

    /// Block numbers holding the on-disk free-list chain written by the last
    /// `persist_free_list`. These are allocated-and-accounted-for containers,
    /// not orphans: GC must mark them live, or the sweep would free them while
    /// the next `persist_free_list` (which recycles them via `last_chain_blocks`)
    /// re-adds them — duplicating a block on the free list.
    pub fn chain_blocks_snapshot(&self) -> Vec<u64> {
        self.last_chain_blocks.lock().unwrap().clone()
    }

    /// Current value of the allocator counter (for snapshotting into
    /// the superblock at sync time).
    pub fn next_block_nr(&self) -> u64 {
        self.next_block_nr.load(Ordering::Relaxed)
    }

    /// Override the allocator after loading a superblock.
    pub fn set_next_block_nr(&self, n: u64) {
        self.next_block_nr.store(n, Ordering::Relaxed);
    }

    // ---------- Btree node IO ----------

    /// Access a btree node under the cache read lock, running `f` against it
    /// and returning the result. The lock is held only for the duration of
    /// `f`, so callers must extract whatever they need (a child block number,
    /// a copied-out value) inside the closure and recurse *outside* it — do
    /// NOT call another `with_node` from within `f` (it would re-enter the
    /// lock and can deadlock).
    ///
    /// On a cache miss the node is faulted in from the device (magic + CRC
    /// verified) and published before `f` runs. A block never written reads
    /// back as zeros and fails the magic check — the device equivalent of
    /// "not found" for both image and RAM backings.
    pub fn with_node<R>(&self, nr: u64, f: impl FnOnce(&BtreeNodeRaw) -> R) -> Result<R> {
        // Fast path: already cached.
        {
            let cache = self.node_cache.read().unwrap();
            if let Some(c) = cache.get(&nr) {
                return Ok(f(&c.node));
            }
        }
        // Miss: fault in from the device, then publish and run under the lock.
        // Allocate an aligned BtreeNodeRaw box, fill it, verify magic + CRC,
        // then publish to the cache.
        let mut node = unsafe {
            // SAFETY: BtreeNodeRaw is a POD `#[derive(FromBytes)]` type and
            // Box::new_zeroed gives us properly aligned, zero-initialized
            // memory. We immediately read_at into it, fully overwriting before
            // any read.
            Box::<BtreeNodeRaw>::new_zeroed().assume_init()
        };
        self.device
            .read_at(node.as_mut_bytes(), nr * BLOCK_SIZE as u64)?;
        verify_node_in_place(nr, &node)?;
        let mut cache = self.node_cache.write().unwrap();
        // Another thread may have raced us in; either copy is valid.
        let entry = cache.entry(nr).or_insert_with(|| {
            Box::new(CachedNode {
                node: *node,
                dirty: false,
            })
        });
        Ok(f(&entry.node))
    }

    /// Read a full owned copy of a btree node. Only for test assertions —
    /// production read paths use `with_node` to avoid the 4 KB copy.
    #[cfg(test)]
    pub fn read_node_copy(&self, nr: u64) -> Result<Box<BtreeNodeRaw>> {
        self.with_node(nr, |n| Box::new(n.clone()))
    }

    /// Mutate a cached btree node in place, marking it dirty. **Does not
    /// pwrite** — the change lives only in the cache until the next
    /// `checkpoint_flush` / `flush_all`. This is the write-back entry point
    /// that backs bcachefs-style in-place bset append: between two
    /// checkpoints, a hot leaf is appended to at a stable `block_nr` without
    /// COWing the root→leaf path.
    ///
    /// Same locking discipline as `with_node`: the write lock is held only for
    /// the duration of `f`. Do NOT call another `with_node*` from within `f`
    /// (it would re-enter the lock and deadlock).
    ///
    /// On a cache miss the node is faulted in from the device (magic + CRC
    /// verified) before `f` runs, so an in-place edit of a node that was
    /// flushed in a prior checkpoint works transparently.
    ///
    /// Safety of "dirty but not persisted": between checkpoints the on-disk
    /// bytes of a node are never read by recovery — `Fs::open` reopens at the
    /// last checkpoint root and replays the WAL forward — so a node that is
    /// dirty-in-cache-only is fully covered by its journal frames.
    pub fn with_node_mut<R>(&self, nr: u64, f: impl FnOnce(&mut BtreeNodeRaw) -> R) -> Result<R> {
        // Fast path: already cached — take the write lock and mutate.
        {
            let mut cache = self.node_cache.write().unwrap();
            if let Some(c) = cache.get_mut(&nr) {
                let r = f(&mut c.node);
                c.dirty = true;
                return Ok(r);
            }
        }
        // Miss: fault in from the device, publish, then mutate under the lock.
        let mut node = unsafe {
            // SAFETY: see `with_node` — POD FromBytes type, zeroed + aligned by
            // new_zeroed, fully overwritten by read_at.
            Box::<BtreeNodeRaw>::new_zeroed().assume_init()
        };
        self.device
            .read_at(node.as_mut_bytes(), nr * BLOCK_SIZE as u64)?;
        verify_node_in_place(nr, &node)?;
        let mut cache = self.node_cache.write().unwrap();
        let entry = cache.entry(nr).or_insert_with(|| {
            Box::new(CachedNode {
                node: *node,
                dirty: false,
            })
        });
        let r = f(&mut entry.node);
        entry.dirty = true;
        Ok(r)
    }

    /// Peek a node **only if it is already cached**, running `f(node, dirty)`.
    /// Returns `None` without faulting anything in when the block is not
    /// resident. `checkpoint_flush` uses this to walk exactly the subtree that
    /// was touched this interval: a child absent from cache was never read or
    /// written since the last checkpoint, so its on-disk subtree is clean and
    /// can be kept as-is without reading it.
    ///
    /// Correctness relies on: (a) dirty nodes are never evicted, and (b) an
    /// in-place edit leaves the whole root→leaf path resident (the descent
    /// faulted every ancestor in). Both hold today (unbounded cache). A future
    /// bounded/LRU cache must preserve them (flush-before-evict, and never
    /// evict a node with a dirty descendant).
    pub fn with_cached_node<R>(
        &self,
        nr: u64,
        f: impl FnOnce(&BtreeNodeRaw, bool) -> R,
    ) -> Option<R> {
        let cache = self.node_cache.read().unwrap();
        cache.get(&nr).map(|c| f(&c.node, c.dirty))
    }

    /// Write a btree node, stamping CRC into its header before persisting.
    ///
    /// Write-through: the block is written to the device immediately and the
    /// cache entry is marked clean. Used by the paths that produce a node at a
    /// *fresh* block — split / promote, and `checkpoint_flush`'s COW-relocate —
    /// where an immediate persist to a never-before-written block is always
    /// safe. Hot in-place edits go through `with_node_mut` instead (dirty, not
    /// pwritten until the next checkpoint).
    pub fn write_node(&self, nr: u64, node: &BtreeNodeRaw) -> Result<()> {
        #[cfg(test)]
        self.node_writes.fetch_add(1, Ordering::Relaxed);
        let mut copy = node.clone();
        copy.header.checksum = 0;
        let crc = crc32fast::hash(copy.as_bytes()) as u64;
        copy.header.checksum = crc;
        self.device
            .write_at(copy.as_bytes(), nr * BLOCK_SIZE as u64)?;
        self.node_cache.write().unwrap().insert(
            nr,
            Box::new(CachedNode {
                node: copy,
                dirty: false,
            }),
        );
        Ok(())
    }

    /// Test-only: number of `write_node` persists so far.
    #[cfg(test)]
    pub fn node_writes(&self) -> u64 {
        self.node_writes.load(Ordering::Relaxed)
    }

    /// Flush a single dirty node to disk **in place** (CRC stamped, then pwrite
    /// to the node's own block). No-op if the node is absent or already clean.
    ///
    /// Not used on the checkpoint path: `checkpoint_flush` relocates dirty
    /// nodes onto *fresh* blocks so a crash can't corrupt the still-live
    /// previous checkpoint. Writing a dirty node back to its own block_nr would
    /// overwrite committed state, so this must NOT be called for checkpointing.
    ///
    /// Kept as the entry point for a future bounded cache with LRU eviction:
    /// evicting a dirty node requires flushing it first (flush-before-evict).
    /// A block written between checkpoints is re-derivable from the WAL, so an
    /// in-place flush of it is safe there — it is only unsafe as a substitute
    /// for the checkpoint's COW-relocate. Currently unused (unbounded cache).
    pub fn flush_node(&self, nr: u64) -> Result<()> {
        let mut cache = self.node_cache.write().unwrap();
        let Some(c) = cache.get_mut(&nr) else {
            return Ok(());
        };
        if !c.dirty {
            return Ok(());
        }
        c.node.header.checksum = 0;
        let crc = crc32fast::hash(c.node.as_bytes()) as u64;
        c.node.header.checksum = crc;
        self.device
            .write_at(c.node.as_bytes(), nr * BLOCK_SIZE as u64)?;
        c.dirty = false;
        Ok(())
    }

    /// Flush every dirty node in place. Same caveat as [`Self::flush_node`]:
    /// this is the future LRU-eviction primitive, not the checkpoint path
    /// (which uses `checkpoint_flush`'s COW-relocate). Currently unused.
    pub fn flush_all(&self) -> Result<()> {
        let dirty: Vec<u64> = {
            let cache = self.node_cache.read().unwrap();
            cache
                .iter()
                .filter(|(_, c)| c.dirty)
                .map(|(&nr, _)| nr)
                .collect()
        };
        for nr in dirty {
            self.flush_node(nr)?;
        }
        Ok(())
    }

    // ---------- Data block IO ----------

    /// Read a 4 KB data block (file content). No magic / no CRC: data
    /// blocks are raw payload. Caller must have allocated the block first.
    ///
    /// Returns an owned copy: data blocks are only ever read then immediately
    /// copied by callers, and they are never mutated in place, so there's no
    /// need for a closure/borrow API here.
    pub fn read_data(&self, nr: u64) -> Result<[u8; BLOCK_SIZE]> {
        {
            let cache = self.data_cache.read().unwrap();
            if let Some(b) = cache.get(&nr) {
                return Ok(b.0);
            }
        }
        let mut block = Box::new(DataBlock::zeroed());
        self.device.read_at(&mut block.0, nr * BLOCK_SIZE as u64)?;
        let bytes = block.0;
        self.data_cache.write().unwrap().insert(nr, block);
        Ok(bytes)
    }

    /// Write a 4 KB data block.
    pub fn write_data(&self, nr: u64, bytes: &[u8; BLOCK_SIZE]) -> Result<()> {
        self.device.write_at(bytes, nr * BLOCK_SIZE as u64)?;
        self.data_cache
            .write()
            .unwrap()
            .insert(nr, Box::new(DataBlock(*bytes)));
        Ok(())
    }

    // ---------- Superblock IO ----------

    /// Write the superblock at block 0 (with CRC stamped). Does not fsync.
    pub fn write_superblock(&self, sb: &Superblock) -> Result<()> {
        let bytes = sb.to_bytes();
        self.device
            .write_at(&bytes, SUPERBLOCK_BLOCK_NR * BLOCK_SIZE as u64)?;
        // Don't cache the superblock alongside node/data — its layout differs.
        Ok(())
    }

    /// Read and verify the superblock from block 0. A store whose superblock
    /// was never written reads back zeros and fails `parse` with `BadMagic`.
    pub fn read_superblock(&self) -> Result<Superblock> {
        let mut buf = vec![0u8; BLOCK_SIZE];
        self.device
            .read_at(&mut buf, SUPERBLOCK_BLOCK_NR * BLOCK_SIZE as u64)?;
        Superblock::parse(&buf)
    }

    /// Flush the device durably. No-op for volatile (RAM) devices.
    pub fn fsync(&self) -> Result<()> {
        self.device.sync()?;
        Ok(())
    }

    // ---------- Free-list persistence ----------

    const FREE_CHAIN_ENTRIES: usize = (BLOCK_SIZE - 16) / 8; // 510

    /// Persist the in-memory free list as a chain of blocks on disk. Each
    /// chain block layout: `[next_block: u64, n_entries: u64, entries…]`.
    /// Returns the head block number (or 0 if the list is empty).
    /// Consumes fresh blocks from the bump allocator for the chain itself.
    pub fn persist_free_list(&self) -> Result<u64> {
        {
            let old_chain = std::mem::take(&mut *self.last_chain_blocks.lock().unwrap());
            let mut fl = self.free_list.lock().unwrap();
            fl.extend(old_chain);
        }
        let list = self.free_list.lock().unwrap().clone();
        if list.is_empty() {
            return Ok(0);
        }
        let mut head: u64 = 0;
        let mut chain_blocks = Vec::new();
        for chunk in list.chunks(Self::FREE_CHAIN_ENTRIES) {
            let block_nr = self.next_block_nr.fetch_add(1, Ordering::Relaxed);
            let mut buf = [0u8; BLOCK_SIZE];
            buf[..8].copy_from_slice(&head.to_le_bytes());
            let n = chunk.len() as u64;
            buf[8..16].copy_from_slice(&n.to_le_bytes());
            for (i, &entry) in chunk.iter().enumerate() {
                let off = 16 + i * 8;
                buf[off..off + 8].copy_from_slice(&entry.to_le_bytes());
            }
            self.write_data(block_nr, &buf)?;
            head = block_nr;
            chain_blocks.push(block_nr);
        }
        *self.last_chain_blocks.lock().unwrap() = chain_blocks;
        Ok(head)
    }

    /// Load the on-disk free-list chain starting at `head` into the in-memory
    /// free list. Chain blocks themselves are also added to the free list
    /// (they are reclaimable after this checkpoint is superseded).
    pub fn load_free_list(&self, head: u64) -> Result<()> {
        let mut cur = head;
        let mut list = Vec::new();
        let mut chain_blocks = Vec::new();
        while cur != 0 {
            chain_blocks.push(cur);
            let buf = self.read_data(cur)?;
            let next = u64::from_le_bytes(buf[..8].try_into().unwrap());
            let n = u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize;
            for i in 0..n.min(Self::FREE_CHAIN_ENTRIES) {
                let off = 16 + i * 8;
                let entry = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
                list.push(entry);
            }
            cur = next;
        }
        // The chain blocks themselves are now free (they won't be referenced
        // after the next checkpoint writes a new chain).
        list.extend(chain_blocks);
        *self.free_list.lock().unwrap() = list;
        Ok(())
    }
}

/// Verify a node already in our `BtreeNodeRaw` representation: magic at
/// the head + CRC over the whole 4 KB (with `checksum` field treated as 0
/// during compute).
fn verify_node_in_place(nr: u64, node: &BtreeNodeRaw) -> Result<()> {
    if node.header.magic != MAGIC_NUMBER {
        return Err(Error::BadMagic {
            block: nr,
            got: node.header.magic,
            expected: MAGIC_NUMBER,
        });
    }
    let stored = node.header.checksum;
    let mut copy = node.clone();
    copy.header.checksum = 0;
    let want = crc32fast::hash(copy.as_bytes()) as u64;
    if stored != want {
        return Err(Error::ChecksumMismatch { block: nr });
    }
    Ok(())
}

#[cfg(test)]
mod journal_tests {
    use super::*;

    #[test]
    fn commit_end_frame_crc_roundtrip() {
        let mut f = JournalFrame::commit_end(42, 100, 200, 10, 50, u32::MAX - 5, 3, 1);
        f.checksum = f.compute_checksum();
        assert!(f.is_valid(42));
        assert!(!f.is_valid(43)); // wrong seq
        assert_eq!(f.kind(), Some(FrameKind::CommitEnd));
        assert_eq!(f.root_block, 100);
    }

    #[test]
    fn commit_end_frame_detects_corruption() {
        let mut f = JournalFrame::commit_end(1, 65, 66, 1, 2, u32::MAX - 1, 1, 1);
        f.checksum = f.compute_checksum();
        f.root_block = 999; // corrupt
        assert!(!f.is_valid(1));
    }

    #[test]
    fn logged_op_frame_roundtrip() {
        let payload = b"unlink:parent=5,name=foo";
        let mut f = JournalFrame::logged_op(7, 3, payload);
        f.checksum = f.compute_checksum();
        assert!(f.is_valid(7));
        assert_eq!(f.kind(), Some(FrameKind::LoggedOp));
        assert_eq!(f.op_kind, 3);
        assert_eq!(f.op_payload(), payload);
    }

    #[test]
    fn free_list_persist_and_load_roundtrip() {
        let store = BlockStore::in_memory();
        // Burn a few blocks so the bump allocator is past FIRST_DATA_BLOCK_NR.
        let a = store.alloc();
        let b = store.alloc();
        let c = store.alloc();
        store.free(b);
        store.free(a);

        let head = store.persist_free_list().unwrap();
        assert_ne!(head, 0);

        // Create a second store and load from the chain.
        let store2 = BlockStore::in_memory();
        // Seed the same data block into store2's cache so load can read it.
        let buf = store.read_data(head).unwrap();
        store2.write_data(head, &buf).unwrap();
        store2.load_free_list(head).unwrap();

        // After loading, allocs should yield the freed blocks (+ the chain block).
        let got1 = store2.alloc();
        let got2 = store2.alloc();
        let got3 = store2.alloc();
        let mut got = [got1, got2, got3];
        got.sort();
        assert!(got.contains(&a));
        assert!(got.contains(&b));
        assert!(got.contains(&head)); // chain block itself is recycled
        // c was never freed, must not appear.
        assert!(!got.contains(&c));
    }
}
