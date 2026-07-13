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
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

use zerocopy::{FromBytes, IntoBytes, KnownLayout};

use crate::block_btree::{BLOCK_SIZE, BtreeNodeRaw, MAGIC_NUMBER};
use crate::btree::{Error, Result};

/// Block 0 is always the superblock.
pub const SUPERBLOCK_BLOCK_NR: u64 = 0;

/// Journal occupies blocks 1..64 (inclusive).
pub const JOURNAL_MAGIC: u32 = 0x524A_4E4C; // "RJNL"
pub const JOURNAL_BLOCKS: u64 = 64;
pub const FIRST_JOURNAL_BLOCK: u64 = 1;
pub const ENTRIES_PER_BLOCK: usize = 31;
pub const JOURNAL_CAPACITY: u64 = JOURNAL_BLOCKS * ENTRIES_PER_BLOCK as u64; // 1984

/// First block number that may hold a node or data block. Block 0 is the
/// superblock, blocks 1..64 are the journal ring buffer.
pub const FIRST_DATA_BLOCK_NR: u64 = FIRST_JOURNAL_BLOCK + JOURNAL_BLOCKS; // 65

/// Magic number stamped at the head of the superblock.
pub const SUPERBLOCK_MAGIC: u32 = 0x5246_5342; // "RFSB"
/// On-disk format version; bumped on incompatible layout changes.
pub const SUPERBLOCK_VERSION: u32 = 2;

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
    pub _reserved: [u8; BLOCK_SIZE - 64],
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
            _reserved: [0; BLOCK_SIZE - 64],
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
        let sb = Superblock::ref_from_bytes(bytes)
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
        Ok(*sb)
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

/// A single journal entry, written to a journal block. Each block holds
/// `ENTRIES_PER_BLOCK` (31) entries tightly packed; the block's remaining
/// 128 bytes are unused padding.
///
/// `magic` and `checksum` guard against torn/partial writes. `seq` is a
/// monotonically-increasing counter; the replay scanner uses it to find the
/// highest valid entry.
#[repr(C)]
#[derive(KnownLayout, zerocopy::Immutable, IntoBytes, FromBytes, Clone, Copy)]
pub struct JournalEntry {
    pub magic: u32,
    pub checksum: u32,
    pub seq: u64,
    pub root_block: u64,
    pub next_block_nr: u64,
    pub next_bset_seq: u64,
    pub next_ino: u64,
    pub next_snap_id: u32,
    pub next_subvol_id: u32,
    pub current_subvol: u32,
    pub _reserved: [u8; 68],
}

const _: () = assert!(std::mem::size_of::<JournalEntry>() == 128);

impl JournalEntry {
    /// Compute a CRC32 over the entry with `checksum` treated as zero.
    pub fn compute_checksum(&self) -> u32 {
        let mut copy = *self;
        copy.checksum = 0;
        crc32fast::hash(copy.as_bytes())
    }

    /// Return `true` iff magic, seq, and checksum all pass.
    pub fn is_valid(&self, expected_seq: u64) -> bool {
        self.magic == JOURNAL_MAGIC
            && self.seq == expected_seq
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

/// Backing for a `BlockStore`.
///
/// `Memory` keeps everything in the cache forever — used by the existing
/// pure-RAM tests so they don't need to touch the filesystem. `Image` adds
/// pread/pwrite against a single backing file.
enum Backing {
    Memory,
    Image { file: File },
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
    backing: Backing,
}

impl BlockStore {
    /// Pure-RAM store: no image file. Compatible with the previous
    /// `HashMap<u64, BtreeNodeRaw>` mode used by every in-process test.
    pub fn in_memory() -> Self {
        BlockStore {
            node_cache: RwLock::new(HashMap::new()),
            data_cache: RwLock::new(HashMap::new()),
            next_block_nr: AtomicU64::new(FIRST_DATA_BLOCK_NR),
            backing: Backing::Memory,
        }
    }

    /// Create a brand-new image file at `path` and return a store backed by
    /// it. Fails if the file already exists.
    pub fn create_image(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)?;
        Ok(BlockStore {
            node_cache: RwLock::new(HashMap::new()),
            data_cache: RwLock::new(HashMap::new()),
            next_block_nr: AtomicU64::new(FIRST_DATA_BLOCK_NR),
            backing: Backing::Image { file },
        })
    }

    /// Open an existing image file. Caller is responsible for then loading
    /// the superblock and seeding `next_block_nr`.
    pub fn open_image(path: &Path) -> Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        Ok(BlockStore {
            node_cache: RwLock::new(HashMap::new()),
            data_cache: RwLock::new(HashMap::new()),
            next_block_nr: AtomicU64::new(FIRST_DATA_BLOCK_NR),
            backing: Backing::Image { file },
        })
    }

    /// Whether this store is backed by an image file.
    pub fn is_persistent(&self) -> bool {
        matches!(self.backing, Backing::Image { .. })
    }

    /// Hand out the next free block number. Bumps the counter.
    pub fn alloc(&self) -> u64 {
        self.next_block_nr.fetch_add(1, Ordering::Relaxed)
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
    /// On a cache miss with an image backing, the node is faulted in from
    /// disk (magic + CRC verified) and published before `f` runs.
    pub fn with_node<R>(&self, nr: u64, f: impl FnOnce(&BtreeNodeRaw) -> R) -> Result<R> {
        // Fast path: already cached.
        {
            let cache = self.node_cache.read().unwrap();
            if let Some(c) = cache.get(&nr) {
                return Ok(f(&c.node));
            }
        }
        // Miss: fault in (image only), then publish and run under the lock.
        match &self.backing {
            Backing::Memory => Err(Error::BlockNotFound(nr)),
            Backing::Image { file } => {
                // Allocate an aligned BtreeNodeRaw box, fill it from disk,
                // verify magic + CRC, then publish to the cache.
                let mut node = unsafe {
                    // SAFETY: BtreeNodeRaw is a POD `#[derive(FromBytes)]`
                    // type and Box::new_zeroed gives us properly aligned,
                    // zero-initialized memory. We immediately read_at into
                    // it, fully overwriting before any read.
                    Box::<BtreeNodeRaw>::new_zeroed().assume_init()
                };
                file.read_exact_at(node.as_mut_bytes(), nr * BLOCK_SIZE as u64)?;
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
        }
    }

    /// Read a full owned copy of a btree node. Only for test assertions —
    /// production read paths use `with_node` to avoid the 4 KB copy.
    #[cfg(test)]
    pub fn read_node_copy(&self, nr: u64) -> Result<Box<BtreeNodeRaw>> {
        self.with_node(nr, |n| Box::new(n.clone()))
    }

    /// Write a btree node, stamping CRC into its header before persisting.
    ///
    /// Write-through: the block is pwritten immediately (image backing) and
    /// the cache entry is marked clean. The `dirty` bookkeeping is in place
    /// for the future write-back step.
    pub fn write_node(&self, nr: u64, node: &BtreeNodeRaw) -> Result<()> {
        let mut copy = node.clone();
        copy.header.checksum = 0;
        let crc = crc32fast::hash(copy.as_bytes()) as u64;
        copy.header.checksum = crc;
        if let Backing::Image { file } = &self.backing {
            file.write_all_at(copy.as_bytes(), nr * BLOCK_SIZE as u64)?;
        }
        self.node_cache.write().unwrap().insert(
            nr,
            Box::new(CachedNode {
                node: copy,
                dirty: false,
            }),
        );
        Ok(())
    }

    /// Flush a single dirty node to disk (CRC stamped, then pwrite). No-op if
    /// the node is absent or already clean. Under write-through nodes are
    /// never left dirty, so this is currently only exercised by the future
    /// write-back path; kept here so `sync` can call it unconditionally.
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
        if let Backing::Image { file } = &self.backing {
            file.write_all_at(c.node.as_bytes(), nr * BLOCK_SIZE as u64)?;
        }
        c.dirty = false;
        Ok(())
    }

    /// Flush every dirty node. No-op under write-through (nothing is dirty).
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
        match &self.backing {
            Backing::Memory => Err(Error::BlockNotFound(nr)),
            Backing::Image { file } => {
                let mut block = Box::new(DataBlock::zeroed());
                file.read_exact_at(&mut block.0, nr * BLOCK_SIZE as u64)?;
                let bytes = block.0;
                self.data_cache.write().unwrap().insert(nr, block);
                Ok(bytes)
            }
        }
    }

    /// Write a 4 KB data block.
    pub fn write_data(&self, nr: u64, bytes: &[u8; BLOCK_SIZE]) -> Result<()> {
        if let Backing::Image { file } = &self.backing {
            file.write_all_at(bytes, nr * BLOCK_SIZE as u64)?;
        }
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
        if let Backing::Image { file } = &self.backing {
            file.write_all_at(&bytes, SUPERBLOCK_BLOCK_NR * BLOCK_SIZE as u64)?;
        }
        // Don't cache the superblock alongside node/data — its layout differs.
        Ok(())
    }

    /// Read and verify the superblock from block 0. Memory-only stores
    /// have no superblock and return `BlockNotFound`.
    pub fn read_superblock(&self) -> Result<Superblock> {
        match &self.backing {
            Backing::Memory => Err(Error::BlockNotFound(SUPERBLOCK_BLOCK_NR)),
            Backing::Image { file } => {
                let mut buf = vec![0u8; BLOCK_SIZE];
                file.read_exact_at(&mut buf, SUPERBLOCK_BLOCK_NR * BLOCK_SIZE as u64)?;
                Superblock::parse(&buf)
            }
        }
    }

    /// fdatasync the backing file. No-op on memory-only stores.
    pub fn fsync(&self) -> Result<()> {
        if let Backing::Image { file } = &self.backing {
            file.sync_data()?;
        }
        Ok(())
    }

    /// Clone the backing file handle. Returns an error for memory-only stores.
    pub fn try_clone_file(&self) -> Result<std::fs::File> {
        match &self.backing {
            Backing::Memory => Err(Error::Io(std::io::Error::other("no backing file"))),
            Backing::Image { file } => Ok(file.try_clone()?),
        }
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
    fn journal_entry_crc_roundtrip() {
        let mut entry = JournalEntry {
            magic: JOURNAL_MAGIC,
            checksum: 0,
            seq: 42,
            root_block: 100,
            next_block_nr: 200,
            next_bset_seq: 10,
            next_ino: 50,
            next_snap_id: u32::MAX - 5,
            next_subvol_id: 3,
            current_subvol: 1,
            _reserved: [0; 68],
        };
        entry.checksum = entry.compute_checksum();
        assert!(entry.is_valid(42));
        assert!(!entry.is_valid(43)); // wrong seq
    }

    #[test]
    fn journal_entry_detects_corruption() {
        let mut entry = JournalEntry {
            magic: JOURNAL_MAGIC,
            checksum: 0,
            seq: 1,
            root_block: 65,
            next_block_nr: 66,
            next_bset_seq: 1,
            next_ino: 2,
            next_snap_id: u32::MAX - 1,
            next_subvol_id: 1,
            current_subvol: 1,
            _reserved: [0; 68],
        };
        entry.checksum = entry.compute_checksum();
        entry.root_block = 999; // corrupt
        assert!(!entry.is_valid(1));
    }
}
