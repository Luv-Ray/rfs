use std::sync::Arc;

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::block_btree::{BLOCK_SIZE, MAX_LOGICAL_KEY_SIZE, MAX_VALUE_SIZE, ROOT_SNAP, SnapId};
use crate::btree::{Btree, Result};
use crate::storage::BlockStore;

pub const ROOT_INO: u64 = 1;

pub const FILE_KIND_REGULAR: u8 = 1;
pub const FILE_KIND_DIR: u8 = 2;

// ---------- Key kinds ----------
//
// One physical Btree stores multiple logical trees distinguished by a kind
// byte prefix — the bcachefs pattern. Because inode / parent_ino / offset are
// encoded big-endian, lexicographic order on the raw key matches the natural
// ordering users expect (ino 1 < ino 2, offset 0 < offset 4096, etc.).

const KIND_INODE: u8 = 1;
const KIND_DIRENT: u8 = 2;
const KIND_EXTENT: u8 = 3;
/// Snapshot tree: maps a snap_id to its parent_id (and other metadata).
/// Stored at logical key `[KIND_SNAPSHOT, snap_id_be]` with the entry's
/// own snap_id always set to ROOT_SNAP — snapshot metadata is global, not
/// versioned by snapshot.
const KIND_SNAPSHOT: u8 = 0xF0;
/// Subvolume tree: maps a subvol_id to its current snap_id and root inode.
/// Stored at logical key `[KIND_SUBVOL, subvol_id_be]` with snap=ROOT_SNAP.
const KIND_SUBVOL: u8 = 0xF1;

const INODE_KEY_LEN: usize = 1 + 8;
const DIRENT_PREFIX_LEN: usize = 1 + 8;
const EXTENT_KEY_LEN: usize = 1 + 8 + 8;
const SNAPSHOT_KEY_LEN: usize = 1 + 4;
const SUBVOL_KEY_LEN: usize = 1 + 4;

/// Sentinel meaning "no parent" in the snapshot tree. Real snap_ids are
/// allocated downward from `ROOT_SNAP = u32::MAX`, so 0 will never collide.
pub const NO_PARENT_SNAP: SnapId = 0;
/// Default top-level subvolume id. Created by `Fs::new()`.
pub const ROOT_SUBVOL: SubvolId = 1;
pub type SubvolId = u32;

/// Longest dirent name that still fits in the logical-key portion of an entry.
pub const MAX_NAME_LEN: usize = MAX_LOGICAL_KEY_SIZE - DIRENT_PREFIX_LEN;

fn inode_key(ino: u64) -> [u8; INODE_KEY_LEN] {
    let mut k = [0u8; INODE_KEY_LEN];
    k[0] = KIND_INODE;
    k[1..].copy_from_slice(&ino.to_be_bytes());
    k
}

fn dirent_key(parent_ino: u64, name: &[u8]) -> Vec<u8> {
    assert!(
        name.len() <= MAX_NAME_LEN,
        "dirent name too long: {} > {MAX_NAME_LEN}",
        name.len()
    );
    let mut k = Vec::with_capacity(DIRENT_PREFIX_LEN + name.len());
    k.push(KIND_DIRENT);
    k.extend_from_slice(&parent_ino.to_be_bytes());
    k.extend_from_slice(name);
    k
}

fn dirent_range(parent_ino: u64) -> ([u8; DIRENT_PREFIX_LEN], [u8; DIRENT_PREFIX_LEN]) {
    let mut start = [0u8; DIRENT_PREFIX_LEN];
    start[0] = KIND_DIRENT;
    start[1..].copy_from_slice(&parent_ino.to_be_bytes());
    let mut end = [0u8; DIRENT_PREFIX_LEN];
    end[0] = KIND_DIRENT;
    end[1..].copy_from_slice(&parent_ino.saturating_add(1).to_be_bytes());
    (start, end)
}

fn extent_key(ino: u64, offset: u64) -> [u8; EXTENT_KEY_LEN] {
    let mut k = [0u8; EXTENT_KEY_LEN];
    k[0] = KIND_EXTENT;
    k[1..9].copy_from_slice(&ino.to_be_bytes());
    k[9..].copy_from_slice(&offset.to_be_bytes());
    k
}

fn extent_range(ino: u64) -> ([u8; EXTENT_KEY_LEN], [u8; EXTENT_KEY_LEN]) {
    (extent_key(ino, 0), extent_key(ino.saturating_add(1), 0))
}

fn extent_offset_from_key(key: &[u8]) -> u64 {
    u64::from_be_bytes(key[9..17].try_into().unwrap())
}

fn dirent_name_from_key(key: &[u8]) -> &[u8] {
    &key[DIRENT_PREFIX_LEN..]
}

fn snapshot_key(snap: SnapId) -> [u8; SNAPSHOT_KEY_LEN] {
    let mut k = [0u8; SNAPSHOT_KEY_LEN];
    k[0] = KIND_SNAPSHOT;
    k[1..].copy_from_slice(&snap.to_be_bytes());
    k
}

fn subvol_key(subvol: SubvolId) -> [u8; SUBVOL_KEY_LEN] {
    let mut k = [0u8; SUBVOL_KEY_LEN];
    k[0] = KIND_SUBVOL;
    k[1..].copy_from_slice(&subvol.to_be_bytes());
    k
}

// ---------- Value structs ----------

#[repr(C)]
#[derive(KnownLayout, Immutable, IntoBytes, FromBytes, Clone, Copy, Debug, PartialEq, Eq)]
pub struct InodeV1 {
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub nlink: u32,
    pub size: u64,
    pub atime: u64,
    pub mtime: u64,
    pub ctime: u64,
    /// Parent directory inode. For the root directory this self-references
    /// `ROOT_INO` so `cd ..` from `/` stays put. Hardlinks are not supported,
    /// so each inode has exactly one parent.
    pub parent_ino: u64,
}

const _: () = assert!(std::mem::size_of::<InodeV1>() == 56);
const _: () = assert!(std::mem::size_of::<InodeV1>() <= MAX_VALUE_SIZE);

#[repr(C)]
#[derive(KnownLayout, Immutable, IntoBytes, FromBytes, Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirentV1 {
    pub target_ino: u64,
    pub kind: u8,
    _pad: [u8; 7],
}

const _: () = assert!(std::mem::size_of::<DirentV1>() == 16);

impl DirentV1 {
    pub fn new(target_ino: u64, kind: u8) -> Self {
        DirentV1 {
            target_ino,
            kind,
            _pad: [0; 7],
        }
    }
}

#[repr(C)]
#[derive(KnownLayout, Immutable, IntoBytes, FromBytes, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtentV1 {
    pub len: u32,
    _pad: [u8; 4],
    pub data_block: u64,
}

const _: () = assert!(std::mem::size_of::<ExtentV1>() == 16);

/// Per-snapshot metadata (one entry per snap_id in the snapshot tree).
/// `parent_id == NO_PARENT_SNAP` means the snapshot has no ancestor (root of
/// a snapshot tree). bcachefs convention: parent ids are always larger than
/// child ids, since new snap_ids are allocated downward from `ROOT_SNAP`.
#[repr(C)]
#[derive(KnownLayout, Immutable, IntoBytes, FromBytes, Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapshotV1 {
    pub parent_id: SnapId,
    /// bit 0 = readonly snapshot view (set on the snapshot side after a
    /// `snapshot_subvol` call; the original subvolume keeps a writable id).
    pub flags: u32,
    _reserved: [u8; 16],
}

const _: () = assert!(std::mem::size_of::<SnapshotV1>() == 24);

/// Per-subvolume metadata. The `snap_id` is the subvolume's *current*
/// active snap_id — every fs op performed inside this subvolume reads and
/// writes at this snap_id.
#[repr(C)]
#[derive(KnownLayout, Immutable, IntoBytes, FromBytes, Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubvolV1 {
    pub snap_id: SnapId,
    pub flags: u32,
    pub root_inode: u64,
    pub parent_subvol: SubvolId,
    _reserved: [u8; 12],
}

const _: () = assert!(std::mem::size_of::<SubvolV1>() == 32);

pub const SUBVOL_FLAG_READONLY: u32 = 1 << 0;

// ---------- Fs error type ----------

/// High-level filesystem errors. The lower-level Btree calls return
/// `btree::Result<T>`; high-level ops (unlink, rmdir, rename, ...) return
/// `FsResult<T>` so they can also signal POSIX-ish conditions like ENOENT.
/// `fuse.rs` maps these to libc errnos.
#[derive(Debug)]
pub enum FsError {
    /// Underlying btree returned an error (block missing / I/O / corruption).
    Btree(crate::btree::Error),
    /// Path component not found (ENOENT).
    NotFound,
    /// Operation expected a directory, found a regular file (ENOTDIR).
    NotADirectory,
    /// rmdir on a non-empty directory (ENOTEMPTY).
    NotEmpty,
    /// rename / create where the target already exists (EEXIST).
    AlreadyExists,
}

impl std::fmt::Display for FsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FsError::Btree(e) => write!(f, "btree: {e}"),
            FsError::NotFound => f.write_str("not found"),
            FsError::NotADirectory => f.write_str("not a directory"),
            FsError::NotEmpty => f.write_str("directory not empty"),
            FsError::AlreadyExists => f.write_str("already exists"),
        }
    }
}

impl std::error::Error for FsError {}

impl From<crate::btree::Error> for FsError {
    fn from(e: crate::btree::Error) -> Self {
        FsError::Btree(e)
    }
}

pub type FsResult<T> = std::result::Result<T, FsError>;

// ---------- Fs ----------

pub struct Fs {
    tree: Btree,
    /// Shared block storage. Same `Arc` as `tree.store`; held here too so
    /// `Fs::sync` can reach it without borrowing through the btree.
    store: Arc<BlockStore>,
    next_ino: u64,
    /// Smallest snap_id allocated so far minus one. New snapshots take ids
    /// counting down from `next_snap_id`. bcachefs: parent always > child.
    #[allow(dead_code)] // wired up in Phase 8 (snapshot_subvol)
    next_snap_id: SnapId,
    /// Smallest subvol_id not yet used. Subvol ids count up from ROOT_SUBVOL.
    #[allow(dead_code)] // wired up in Phase 8 (snapshot_subvol)
    next_subvol_id: SubvolId,
    /// The currently active subvolume — every fs op reads/writes at this
    /// subvol's snap_id. v1 always uses ROOT_SUBVOL; multi-subvol switching
    /// is a future feature.
    current_subvol: SubvolId,
}

impl Fs {
    pub fn new() -> Self {
        let tree = Btree::new();
        let store = tree.store.clone();
        Self::seed(tree, store, ROOT_INO)
    }

    /// Build a fresh image-backed filesystem at `path`. Fails if the file
    /// exists. The superblock is written by [`Fs::sync`]; this constructor
    /// only initializes in-memory state.
    pub fn create(path: &std::path::Path) -> Result<Self> {
        let store = Arc::new(BlockStore::create_image(path)?);
        let tree = Btree::create_in(store.clone());
        Ok(Self::seed(tree, store, ROOT_INO))
    }

    /// Reopen a filesystem from an existing image. Reads + verifies the
    /// superblock (magic / version / CRC), then reattaches the btree at the
    /// stored root.
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let store = Arc::new(BlockStore::open_image(path)?);
        let sb = store.read_superblock()?;
        store.set_next_block_nr(sb.next_block_nr);
        let tree = Btree::reopen(store.clone(), sb.root_block, sb.next_bset_seq);
        Ok(Fs {
            tree,
            store,
            next_ino: sb.next_ino,
            next_snap_id: sb.next_snap_id,
            next_subvol_id: sb.next_subvol_id,
            current_subvol: sb.current_subvol,
        })
    }

    /// Persist all live state (`fsync` data + write a fresh superblock).
    /// Called on `close` and any explicit `Fs::sync` checkpoint.
    pub fn sync(&self) -> Result<()> {
        // First flush data blocks + node blocks already written via
        // BlockStore (their pwrite is done at write time, but we want the
        // kernel to push them to the device before we publish the new
        // superblock).
        self.store.fsync()?;
        let sb = crate::storage::Superblock {
            magic: crate::storage::SUPERBLOCK_MAGIC,
            version: crate::storage::SUPERBLOCK_VERSION,
            root_block: self.tree.root_block,
            next_block_nr: self.store.next_block_nr(),
            next_bset_seq: self.tree.next_bset_seq(),
            next_ino: self.next_ino,
            next_snap_id: self.next_snap_id,
            next_subvol_id: self.next_subvol_id,
            current_subvol: self.current_subvol,
            _pad: 0,
            checksum: 0,
            _reserved: [0; BLOCK_SIZE - 60],
        };
        self.store.write_superblock(&sb)?;
        // Second fsync makes the new root visible after a crash.
        self.store.fsync()?;
        Ok(())
    }

    fn seed(tree: Btree, store: Arc<BlockStore>, root_ino: u64) -> Self {
        let mut fs = Fs {
            tree,
            store,
            next_ino: root_ino,
            // First child snapshot gets id = ROOT_SNAP - 1.
            next_snap_id: ROOT_SNAP - 1,
            next_subvol_id: ROOT_SUBVOL + 1,
            current_subvol: ROOT_SUBVOL,
        };
        // Seed the snapshot tree with the root snapshot. parent=NO_PARENT_SNAP
        // marks it as the top of its tree. Stored under snap=ROOT_SNAP itself
        // so the metadata is reachable from any snapshot view.
        fs.put_snapshot(
            ROOT_SNAP,
            &SnapshotV1 {
                parent_id: NO_PARENT_SNAP,
                flags: 0,
                _reserved: [0; 16],
            },
        )
        .expect("seed root snapshot");
        // Seed the default subvolume.
        fs.put_subvol(
            ROOT_SUBVOL,
            &SubvolV1 {
                snap_id: ROOT_SNAP,
                flags: 0,
                root_inode: ROOT_INO,
                parent_subvol: 0,
                _reserved: [0; 12],
            },
        )
        .expect("seed root subvolume");
        fs
    }

    /// snap_id active in the currently selected subvolume. Used by every
    /// fs op as the snap_id to read at and write under.
    pub fn current_snap(&self) -> SnapId {
        self.get_subvol(self.current_subvol)
            .expect("subvol lookup")
            .expect("current subvolume is missing")
            .snap_id
    }

    pub fn current_subvol(&self) -> SubvolId {
        self.current_subvol
    }

    pub fn alloc_ino(&mut self) -> u64 {
        let ino = self.next_ino;
        self.next_ino += 1;
        ino
    }

    /// Allocate a fresh 4 KB data block in the shared store. Returns the
    /// block number; the block is not zeroed on disk (ftruncate-style
    /// sparse, but with our sequential allocator that's fine — the block
    /// is always written before being read).
    fn alloc_data_block(&self) -> u64 {
        self.store.alloc()
    }

    // -- Snapshot tree --
    //
    // Snapshot metadata lives in the same physical Btree at prefix
    // KIND_SNAPSHOT. Every entry is stored at snap=ROOT_SNAP so the
    // ancestor walk for snapshot lookup itself doesn't recurse — snapshot
    // metadata is global, not versioned.

    pub fn put_snapshot(&mut self, snap: SnapId, sv: &SnapshotV1) -> Result<()> {
        self.tree
            .insert_at(&snapshot_key(snap), ROOT_SNAP, sv.as_bytes())
    }

    pub fn get_snapshot(&self, snap: SnapId) -> Result<Option<SnapshotV1>> {
        let bytes = self.tree.find_at(&snapshot_key(snap), ROOT_SNAP)?;
        Ok(bytes.map(|b| SnapshotV1::read_from_bytes(&b).expect("snapshot value size mismatch")))
    }

    /// Walk the snapshot ancestor chain starting at `snap`. Yields
    /// `[snap, parent(snap), grandparent(snap), ...]`, stopping at the first
    /// snapshot whose `parent_id == NO_PARENT_SNAP` (a tree root) or whose
    /// metadata is missing (defensive: shouldn't happen in a valid tree).
    pub fn ancestor_chain(&self, snap: SnapId) -> Result<Vec<SnapId>> {
        let mut chain = vec![snap];
        let mut cur = snap;
        while let Some(meta) = self.get_snapshot(cur)? {
            if meta.parent_id == NO_PARENT_SNAP {
                break;
            }
            // Defensive: in a well-formed tree parent_id > cur. A self-loop
            // or downward link would imply corruption; bail out.
            if meta.parent_id <= cur {
                break;
            }
            chain.push(meta.parent_id);
            cur = meta.parent_id;
        }
        Ok(chain)
    }

    /// Is `ancestor` on the ancestor chain of `of` (inclusive)?
    pub fn is_ancestor(&self, ancestor: SnapId, of: SnapId) -> Result<bool> {
        // Optimization vs the chain walker: ancestors always have id >=
        // descendant, so if ancestor < of we can short-circuit.
        if ancestor < of {
            return Ok(false);
        }
        let mut cur = of;
        loop {
            if cur == ancestor {
                return Ok(true);
            }
            let Some(meta) = self.get_snapshot(cur)? else {
                return Ok(false);
            };
            if meta.parent_id == NO_PARENT_SNAP || meta.parent_id <= cur {
                return Ok(false);
            }
            cur = meta.parent_id;
        }
    }

    // -- Subvolume tree --

    pub fn put_subvol(&mut self, id: SubvolId, sv: &SubvolV1) -> Result<()> {
        self.tree
            .insert_at(&subvol_key(id), ROOT_SNAP, sv.as_bytes())
    }

    pub fn get_subvol(&self, id: SubvolId) -> Result<Option<SubvolV1>> {
        let bytes = self.tree.find_at(&subvol_key(id), ROOT_SNAP)?;
        Ok(bytes.map(|b| SubvolV1::read_from_bytes(&b).expect("subvol value size mismatch")))
    }

    // -- Inode --

    pub fn put_inode(&mut self, ino: u64, inode: &InodeV1) -> Result<()> {
        let snap = self.current_snap();
        self.tree.insert_at(&inode_key(ino), snap, inode.as_bytes())
    }

    pub fn get_inode(&self, ino: u64) -> Result<Option<InodeV1>> {
        let chain = self.current_chain()?;
        let bytes = self.tree.find_visible(&inode_key(ino), &chain)?;
        Ok(bytes.map(|b| InodeV1::read_from_bytes(&b).expect("inode value size mismatch")))
    }

    pub fn delete_inode(&mut self, ino: u64) -> Result<bool> {
        let snap = self.current_snap();
        let chain = self.current_chain()?;
        self.tree.delete_at(&inode_key(ino), snap, &chain)
    }

    // -- Dirent --

    pub fn put_dirent(&mut self, parent: u64, name: &[u8], d: &DirentV1) -> Result<()> {
        let snap = self.current_snap();
        self.tree
            .insert_at(&dirent_key(parent, name), snap, d.as_bytes())
    }

    pub fn lookup_dirent(&self, parent: u64, name: &[u8]) -> Result<Option<DirentV1>> {
        let chain = self.current_chain()?;
        let bytes = self.tree.find_visible(&dirent_key(parent, name), &chain)?;
        Ok(bytes.map(|b| DirentV1::read_from_bytes(&b).expect("dirent value size mismatch")))
    }

    pub fn delete_dirent(&mut self, parent: u64, name: &[u8]) -> Result<bool> {
        let snap = self.current_snap();
        let chain = self.current_chain()?;
        self.tree.delete_at(&dirent_key(parent, name), snap, &chain)
    }

    pub fn list_dirents(&self, parent: u64) -> Result<Vec<(Vec<u8>, DirentV1)>> {
        let chain = self.current_chain()?;
        let (start, end) = dirent_range(parent);
        let entries = self.tree.range_scan_visible(&start, &end, &chain)?;
        Ok(entries
            .into_iter()
            .map(|(k, v)| {
                let name = dirent_name_from_key(&k).to_vec();
                let d = DirentV1::read_from_bytes(&v).expect("dirent value size mismatch");
                (name, d)
            })
            .collect())
    }

    // -- Extent --

    pub fn put_extent(&mut self, ino: u64, offset: u64, data: &[u8]) -> Result<()> {
        assert!(
            data.len() <= BLOCK_SIZE,
            "extent data too large: {} > {BLOCK_SIZE}",
            data.len()
        );
        let snap = self.current_snap();
        let block_nr = self.alloc_data_block();
        let mut block = [0u8; BLOCK_SIZE];
        block[..data.len()].copy_from_slice(data);
        self.store.write_data(block_nr, &block)?;
        let extent = ExtentV1 {
            len: data.len() as u32,
            _pad: [0; 4],
            data_block: block_nr,
        };
        self.tree
            .insert_at(&extent_key(ino, offset), snap, extent.as_bytes())
    }

    pub fn get_extent(&self, ino: u64, offset: u64) -> Result<Option<ExtentV1>> {
        let chain = self.current_chain()?;
        let bytes = self.tree.find_visible(&extent_key(ino, offset), &chain)?;
        Ok(bytes.map(|b| ExtentV1::read_from_bytes(&b).expect("extent value size mismatch")))
    }

    pub fn delete_extent(&mut self, ino: u64, offset: u64) -> Result<bool> {
        let snap = self.current_snap();
        let chain = self.current_chain()?;
        self.tree.delete_at(&extent_key(ino, offset), snap, &chain)
    }

    pub fn read_data_block(&self, block_nr: u64) -> Result<&[u8; BLOCK_SIZE]> {
        self.store.read_data(block_nr)
    }

    pub fn list_extents(&self, ino: u64) -> Result<Vec<(u64, ExtentV1)>> {
        let chain = self.current_chain()?;
        let (start, end) = extent_range(ino);
        let entries = self.tree.range_scan_visible(&start, &end, &chain)?;
        Ok(entries
            .into_iter()
            .map(|(k, v)| {
                let offset = extent_offset_from_key(&k);
                let extent = ExtentV1::read_from_bytes(&v).expect("extent value size mismatch");
                (offset, extent)
            })
            .collect())
    }

    /// The ancestor chain of the currently active subvolume's snap_id.
    /// Used by every read path to resolve visibility.
    fn current_chain(&self) -> Result<Vec<SnapId>> {
        self.ancestor_chain(self.current_snap())
    }

    // -- High-level POSIX-ish ops (Phase 7) --
    //
    // These wrap multi-key changes in a single Btree::transaction so the
    // outside view sees a single root swap (atomic).

    /// Remove a regular file. Decrements nlink; on the last link, removes
    /// the inode and all of its extents (data blocks remain in `data_blocks`
    /// for now — block reclamation is a TODO since snapshots can still hold
    /// references to them via inherited extent entries).
    pub fn unlink(&mut self, parent: u64, name: &[u8]) -> FsResult<()> {
        let snap = self.current_snap();
        let chain = self.current_chain()?;

        let dirent = self.lookup_dirent(parent, name)?.ok_or(FsError::NotFound)?;
        let target_ino = dirent.target_ino;
        let inode = self.get_inode(target_ino)?.ok_or(FsError::NotFound)?;

        // Last-link case: pre-collect extent offsets to delete in tx. Done
        // here (not inside the closure) because we only have &mut Tx in the
        // closure, not access to Fs::list_extents.
        let extents: Vec<u64> = if inode.nlink <= 1 {
            self.list_extents(target_ino)?
                .into_iter()
                .map(|(off, _)| off)
                .collect()
        } else {
            Vec::new()
        };

        let dirent_k = dirent_key(parent, name);
        let inode_k = inode_key(target_ino);

        self.tree.transaction(|tx| {
            tx.delete_at(&dirent_k, snap, &chain)?;
            if inode.nlink <= 1 {
                for offset in &extents {
                    tx.delete_at(&extent_key(target_ino, *offset), snap, &chain)?;
                }
                tx.delete_at(&inode_k, snap, &chain)?;
            } else {
                let mut updated = inode;
                updated.nlink -= 1;
                tx.insert(&inode_k, snap, updated.as_bytes())?;
            }
            Ok(())
        })?;
        Ok(())
    }

    /// Remove an empty directory. Returns ENOTDIR if the target isn't a
    /// directory and ENOTEMPTY if it has any visible children.
    pub fn rmdir(&mut self, parent: u64, name: &[u8]) -> FsResult<()> {
        let snap = self.current_snap();
        let chain = self.current_chain()?;

        let dirent = self.lookup_dirent(parent, name)?.ok_or(FsError::NotFound)?;
        if dirent.kind != FILE_KIND_DIR {
            return Err(FsError::NotADirectory);
        }
        let target_ino = dirent.target_ino;

        // list_dirents already filters tombstones via range_scan_visible,
        // so any non-empty result means a live child still exists.
        if !self.list_dirents(target_ino)?.is_empty() {
            return Err(FsError::NotEmpty);
        }

        let dirent_k = dirent_key(parent, name);
        let inode_k = inode_key(target_ino);

        self.tree.transaction(|tx| {
            tx.delete_at(&dirent_k, snap, &chain)?;
            tx.delete_at(&inode_k, snap, &chain)?;
            Ok(())
        })?;
        Ok(())
    }

    /// Move a dirent from `(old_parent, old_name)` to `(new_parent, new_name)`.
    /// v1 refuses when the destination exists (EEXIST); overwrite-rename
    /// can be added later.
    pub fn rename(
        &mut self,
        old_parent: u64,
        old_name: &[u8],
        new_parent: u64,
        new_name: &[u8],
    ) -> FsResult<()> {
        let snap = self.current_snap();
        let chain = self.current_chain()?;

        let src_dirent = self
            .lookup_dirent(old_parent, old_name)?
            .ok_or(FsError::NotFound)?;
        if self.lookup_dirent(new_parent, new_name)?.is_some() {
            return Err(FsError::AlreadyExists);
        }

        let old_k = dirent_key(old_parent, old_name);
        let new_k = dirent_key(new_parent, new_name);

        self.tree.transaction(|tx| {
            tx.insert(&new_k, snap, src_dirent.as_bytes())?;
            tx.delete_at(&old_k, snap, &chain)?;
            Ok(())
        })?;
        Ok(())
    }

    // -- Snapshot create (Phase 8) --

    /// Take a writable snapshot of `src` subvolume.
    ///
    /// bcachefs flow:
    /// 1. The src subvol is currently at snap_id S.
    /// 2. Allocate two new ids `S_w` and `S_ro` (both `< S`).
    /// 3. Both become children of S in the snapshot tree.
    /// 4. The src subvol's snap_id flips to `S_w` (it keeps writing here).
    /// 5. A fresh subvol is created pointing at `S_ro`, marked readonly.
    ///
    /// The result: existing data (written at S) is visible from BOTH the
    /// updated src subvol (via S_w → S in chain) and the new snapshot
    /// (via S_ro → S in chain). New writes diverge.
    ///
    /// Returns the new readonly subvolume's id.
    pub fn snapshot_subvol(&mut self, src: SubvolId) -> FsResult<SubvolId> {
        let src_subvol = self.get_subvol(src)?.ok_or(FsError::NotFound)?;
        let parent_snap = src_subvol.snap_id;

        // Allocate ids. bcachefs allocates snap_ids decreasing so that
        // parent.id > child.id always holds — assertion enforced in
        // ancestor_chain.
        let s_w = self.next_snap_id;
        let s_ro = self.next_snap_id - 1;
        self.next_snap_id = self.next_snap_id.saturating_sub(2);
        let new_subvol_id = self.next_subvol_id;
        self.next_subvol_id = self.next_subvol_id.saturating_add(1);

        let snap_w = SnapshotV1 {
            parent_id: parent_snap,
            flags: 0,
            _reserved: [0; 16],
        };
        let snap_ro = SnapshotV1 {
            parent_id: parent_snap,
            flags: 1, // readonly view; not enforced yet but recorded.
            _reserved: [0; 16],
        };
        let mut updated_src = src_subvol;
        updated_src.snap_id = s_w;
        let dst_subvol = SubvolV1 {
            snap_id: s_ro,
            flags: SUBVOL_FLAG_READONLY,
            root_inode: src_subvol.root_inode,
            parent_subvol: src,
            _reserved: [0; 12],
        };

        let s_w_key = snapshot_key(s_w);
        let s_ro_key = snapshot_key(s_ro);
        let src_subvol_key = subvol_key(src);
        let dst_subvol_key = subvol_key(new_subvol_id);

        self.tree.transaction(|tx| {
            tx.insert(&s_w_key, ROOT_SNAP, snap_w.as_bytes())?;
            tx.insert(&s_ro_key, ROOT_SNAP, snap_ro.as_bytes())?;
            // Overwrite src subvol with bumped snap_id.
            tx.insert(&src_subvol_key, ROOT_SNAP, updated_src.as_bytes())?;
            // Insert the new readonly subvol.
            tx.insert(&dst_subvol_key, ROOT_SNAP, dst_subvol.as_bytes())?;
            Ok(())
        })?;

        Ok(new_subvol_id)
    }

    /// Switch the active subvolume. All subsequent fs ops read and write
    /// at the target subvol's snap_id.
    pub fn switch_subvol(&mut self, id: SubvolId) -> FsResult<()> {
        if self.get_subvol(id)?.is_none() {
            return Err(FsError::NotFound);
        }
        self.current_subvol = id;
        Ok(())
    }
}

impl Default for Fs {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_inode(size: u64) -> InodeV1 {
        InodeV1 {
            mode: 0o100644,
            uid: 1000,
            gid: 1000,
            nlink: 1,
            size,
            atime: 10,
            mtime: 20,
            ctime: 30,
            parent_ino: ROOT_INO,
        }
    }

    #[test]
    fn inode_round_trip() {
        let mut fs = Fs::new();
        let inode = sample_inode(4242);
        fs.put_inode(5, &inode).unwrap();
        assert_eq!(fs.get_inode(5).unwrap(), Some(inode));
        assert_eq!(fs.get_inode(6).unwrap(), None);
    }

    #[test]
    fn dirent_lookup_hit_and_miss() {
        let mut fs = Fs::new();
        let d = DirentV1::new(42, FILE_KIND_REGULAR);
        fs.put_dirent(1, b"a", &d).unwrap();
        assert_eq!(fs.lookup_dirent(1, b"a").unwrap(), Some(d));
        assert_eq!(fs.lookup_dirent(1, b"b").unwrap(), None);
    }

    #[test]
    fn list_dirents_returns_sorted_names() {
        let mut fs = Fs::new();
        for (name, target) in [(&b"b"[..], 11), (b"a", 10), (b"c", 12)] {
            fs.put_dirent(1, name, &DirentV1::new(target, FILE_KIND_REGULAR))
                .unwrap();
        }
        let got = fs.list_dirents(1).unwrap();
        let names: Vec<&[u8]> = got.iter().map(|(n, _)| n.as_slice()).collect();
        assert_eq!(names, vec![&b"a"[..], b"b", b"c"]);
        let inos: Vec<u64> = got.iter().map(|(_, d)| d.target_ino).collect();
        assert_eq!(inos, vec![10, 11, 12]);
    }

    #[test]
    fn dirent_isolated_across_parents() {
        let mut fs = Fs::new();
        fs.put_dirent(1, b"x", &DirentV1::new(100, FILE_KIND_REGULAR))
            .unwrap();
        fs.put_dirent(2, b"x", &DirentV1::new(200, FILE_KIND_REGULAR))
            .unwrap();
        assert_eq!(fs.lookup_dirent(1, b"x").unwrap().unwrap().target_ino, 100);
        assert_eq!(fs.lookup_dirent(2, b"x").unwrap().unwrap().target_ino, 200);
        assert_eq!(fs.list_dirents(1).unwrap().len(), 1);
        assert_eq!(fs.list_dirents(2).unwrap().len(), 1);
    }

    #[test]
    fn extent_round_trip() {
        let mut fs = Fs::new();
        fs.put_extent(5, 0, b"hello").unwrap();
        let ext = fs.get_extent(5, 0).unwrap().unwrap();
        assert_eq!(ext.len, 5);
        let block = fs
            .read_data_block(ext.data_block)
            .expect("data block missing");
        assert_eq!(&block[..ext.len as usize], b"hello");
    }

    #[test]
    fn list_extents_sorted_by_offset() {
        let mut fs = Fs::new();
        fs.put_extent(5, 8192, b"three").unwrap();
        fs.put_extent(5, 0, b"one").unwrap();
        fs.put_extent(5, 4096, b"two").unwrap();
        let got = fs.list_extents(5).unwrap();
        let offsets: Vec<u64> = got.iter().map(|(o, _)| *o).collect();
        assert_eq!(offsets, vec![0, 4096, 8192]);
    }

    #[test]
    fn extent_isolated_across_inos() {
        let mut fs = Fs::new();
        fs.put_extent(5, 0, b"aaa").unwrap();
        fs.put_extent(6, 0, b"bbb").unwrap();
        let ino5 = fs.list_extents(5).unwrap();
        let ino6 = fs.list_extents(6).unwrap();
        assert_eq!(ino5.len(), 1);
        assert_eq!(ino6.len(), 1);
        assert_eq!(
            &fs.read_data_block(ino5[0].1.data_block)
                .expect("data block missing")[..3],
            b"aaa"
        );
        assert_eq!(
            &fs.read_data_block(ino6[0].1.data_block)
                .expect("data block missing")[..3],
            b"bbb"
        );
    }

    #[test]
    fn kinds_isolated_within_same_id() {
        // ino=5 has an inode, a dirent subtree (parent=5), and extents (ino=5).
        // None of the three range scans should see keys from the others.
        let mut fs = Fs::new();
        fs.put_inode(5, &sample_inode(100)).unwrap();
        fs.put_dirent(5, b"child", &DirentV1::new(9, FILE_KIND_REGULAR))
            .unwrap();
        fs.put_extent(5, 0, b"data").unwrap();

        assert!(fs.get_inode(5).unwrap().is_some());
        let dirents = fs.list_dirents(5).unwrap();
        assert_eq!(dirents.len(), 1);
        assert_eq!(dirents[0].0, b"child");
        let extents = fs.list_extents(5).unwrap();
        assert_eq!(extents.len(), 1);
        assert_eq!(extents[0].0, 0);
    }

    #[test]
    fn max_length_dirent_name_works() {
        let mut fs = Fs::new();
        let name = vec![b'x'; MAX_NAME_LEN];
        let d = DirentV1::new(7, FILE_KIND_DIR);
        fs.put_dirent(1, &name, &d).unwrap();
        assert_eq!(fs.lookup_dirent(1, &name).unwrap(), Some(d));
    }

    #[test]
    #[should_panic(expected = "dirent name too long")]
    fn oversized_dirent_name_panics() {
        let mut fs = Fs::new();
        let name = vec![b'x'; MAX_NAME_LEN + 1];
        let _ = fs.put_dirent(1, &name, &DirentV1::new(7, FILE_KIND_DIR));
    }

    #[test]
    fn alloc_ino_monotonic_from_root() {
        let mut fs = Fs::new();
        assert_eq!(fs.alloc_ino(), ROOT_INO);
        assert_eq!(fs.alloc_ino(), ROOT_INO + 1);
        assert_eq!(fs.alloc_ino(), ROOT_INO + 2);
    }

    // ---------- Phase 3: snapshot + subvolume tree ----------

    #[test]
    fn fs_new_seeds_root_snapshot_and_subvol() {
        let fs = Fs::new();
        // The ROOT_SNAP entry exists with no parent.
        let s = fs.get_snapshot(ROOT_SNAP).unwrap().unwrap();
        assert_eq!(s.parent_id, NO_PARENT_SNAP);
        // The default subvolume points at ROOT_SNAP and is the active one.
        let v = fs.get_subvol(ROOT_SUBVOL).unwrap().unwrap();
        assert_eq!(v.snap_id, ROOT_SNAP);
        assert_eq!(v.root_inode, ROOT_INO);
        assert_eq!(fs.current_subvol(), ROOT_SUBVOL);
        assert_eq!(fs.current_snap(), ROOT_SNAP);
    }

    #[test]
    fn ancestor_chain_walks_to_root() {
        // Manually build a snapshot tree:
        //   ROOT_SNAP
        //      |
        //     100
        //      |
        //     50
        let mut fs = Fs::new();
        fs.put_snapshot(
            100,
            &SnapshotV1 {
                parent_id: ROOT_SNAP,
                flags: 0,
                _reserved: [0; 16],
            },
        )
        .unwrap();
        fs.put_snapshot(
            50,
            &SnapshotV1 {
                parent_id: 100,
                flags: 0,
                _reserved: [0; 16],
            },
        )
        .unwrap();

        assert_eq!(fs.ancestor_chain(50).unwrap(), vec![50, 100, ROOT_SNAP]);
        assert_eq!(fs.ancestor_chain(100).unwrap(), vec![100, ROOT_SNAP]);
        assert_eq!(fs.ancestor_chain(ROOT_SNAP).unwrap(), vec![ROOT_SNAP]);
    }

    #[test]
    fn is_ancestor_inclusive_and_directional() {
        let mut fs = Fs::new();
        fs.put_snapshot(
            100,
            &SnapshotV1 {
                parent_id: ROOT_SNAP,
                flags: 0,
                _reserved: [0; 16],
            },
        )
        .unwrap();
        fs.put_snapshot(
            50,
            &SnapshotV1 {
                parent_id: 100,
                flags: 0,
                _reserved: [0; 16],
            },
        )
        .unwrap();

        // Self is an ancestor of self (inclusive).
        assert!(fs.is_ancestor(50, 50).unwrap());
        // Direct parent and grandparent.
        assert!(fs.is_ancestor(100, 50).unwrap());
        assert!(fs.is_ancestor(ROOT_SNAP, 50).unwrap());
        // Reverse direction is false.
        assert!(!fs.is_ancestor(50, 100).unwrap());
        assert!(!fs.is_ancestor(50, ROOT_SNAP).unwrap());
        // Snapshots on different branches: nothing in between.
        fs.put_snapshot(
            80,
            &SnapshotV1 {
                parent_id: ROOT_SNAP,
                flags: 0,
                _reserved: [0; 16],
            },
        )
        .unwrap();
        // 80 and 50 are both descendants of ROOT_SNAP but unrelated.
        assert!(!fs.is_ancestor(80, 50).unwrap());
        assert!(!fs.is_ancestor(50, 80).unwrap());
        // Their common ancestor IS an ancestor of both.
        assert!(fs.is_ancestor(ROOT_SNAP, 80).unwrap());
        assert!(fs.is_ancestor(ROOT_SNAP, 50).unwrap());
    }

    #[test]
    fn snapshot_metadata_does_not_leak_into_user_keyspace() {
        // Snapshot/subvol entries live in the same physical Btree but with
        // dedicated kind-byte prefixes, so a list_dirents under the user's
        // root inode must not see them.
        let mut fs = Fs::new();
        fs.put_dirent(ROOT_INO, b"hello", &DirentV1::new(2, FILE_KIND_REGULAR))
            .unwrap();
        let dirents = fs.list_dirents(ROOT_INO).unwrap();
        assert_eq!(dirents.len(), 1);
        assert_eq!(dirents[0].0, b"hello");
    }

    // ---------- Phase 7: unlink / rmdir / rename ----------

    fn make_root(fs: &mut Fs) {
        // alloc_ino() returns ROOT_INO=1 first (matching fuse.rs convention).
        let r = fs.alloc_ino();
        debug_assert_eq!(r, ROOT_INO);
        fs.put_inode(ROOT_INO, &sample_inode(0)).unwrap();
    }

    fn make_file(fs: &mut Fs, parent: u64, name: &[u8]) -> u64 {
        let ino = fs.alloc_ino();
        fs.put_inode(ino, &sample_inode(0)).unwrap();
        fs.put_dirent(parent, name, &DirentV1::new(ino, FILE_KIND_REGULAR))
            .unwrap();
        ino
    }

    fn make_dir(fs: &mut Fs, parent: u64, name: &[u8]) -> u64 {
        let ino = fs.alloc_ino();
        let mut ind = sample_inode(0);
        ind.mode = 0o040755;
        fs.put_inode(ino, &ind).unwrap();
        fs.put_dirent(parent, name, &DirentV1::new(ino, FILE_KIND_DIR))
            .unwrap();
        ino
    }

    #[test]
    fn unlink_removes_dirent_and_inode_for_last_link() {
        let mut fs = Fs::new();
        make_root(&mut fs);
        let ino = make_file(&mut fs, ROOT_INO, b"f");
        fs.put_extent(ino, 0, b"hello").unwrap();
        fs.put_extent(ino, 4096, b"world").unwrap();

        fs.unlink(ROOT_INO, b"f").unwrap();
        // Dirent gone.
        assert_eq!(fs.lookup_dirent(ROOT_INO, b"f").unwrap(), None);
        // Inode gone.
        assert_eq!(fs.get_inode(ino).unwrap(), None);
        // Extents gone (visible scan returns nothing).
        assert_eq!(fs.list_extents(ino).unwrap().len(), 0);
    }

    #[test]
    fn unlink_nonexistent_returns_not_found() {
        let mut fs = Fs::new();
        match fs.unlink(ROOT_INO, b"missing") {
            Err(FsError::NotFound) => {}
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn rmdir_empty_dir_succeeds() {
        let mut fs = Fs::new();
        make_root(&mut fs);
        let dir_ino = make_dir(&mut fs, ROOT_INO, b"d");
        fs.rmdir(ROOT_INO, b"d").unwrap();
        assert_eq!(fs.lookup_dirent(ROOT_INO, b"d").unwrap(), None);
        assert_eq!(fs.get_inode(dir_ino).unwrap(), None);
    }

    #[test]
    fn rmdir_nonempty_returns_not_empty() {
        let mut fs = Fs::new();
        make_root(&mut fs);
        let dir_ino = make_dir(&mut fs, ROOT_INO, b"d");
        let _ = make_file(&mut fs, dir_ino, b"child");
        match fs.rmdir(ROOT_INO, b"d") {
            Err(FsError::NotEmpty) => {}
            other => panic!("expected NotEmpty, got {other:?}"),
        }
    }

    #[test]
    fn rmdir_on_regular_file_returns_not_a_directory() {
        let mut fs = Fs::new();
        make_root(&mut fs);
        let _ = make_file(&mut fs, ROOT_INO, b"f");
        match fs.rmdir(ROOT_INO, b"f") {
            Err(FsError::NotADirectory) => {}
            other => panic!("expected NotADirectory, got {other:?}"),
        }
    }

    #[test]
    fn rename_moves_dirent_atomically() {
        let mut fs = Fs::new();
        make_root(&mut fs);
        let ino = make_file(&mut fs, ROOT_INO, b"old");
        fs.rename(ROOT_INO, b"old", ROOT_INO, b"new").unwrap();
        assert_eq!(fs.lookup_dirent(ROOT_INO, b"old").unwrap(), None);
        let d = fs.lookup_dirent(ROOT_INO, b"new").unwrap().unwrap();
        assert_eq!(d.target_ino, ino);
    }

    #[test]
    fn rename_into_other_dir_works() {
        let mut fs = Fs::new();
        make_root(&mut fs);
        let ino = make_file(&mut fs, ROOT_INO, b"f");
        let other_dir = make_dir(&mut fs, ROOT_INO, b"d");
        fs.rename(ROOT_INO, b"f", other_dir, b"f").unwrap();
        assert_eq!(fs.lookup_dirent(ROOT_INO, b"f").unwrap(), None);
        let d = fs.lookup_dirent(other_dir, b"f").unwrap().unwrap();
        assert_eq!(d.target_ino, ino);
    }

    #[test]
    fn rename_refuses_when_dst_exists() {
        let mut fs = Fs::new();
        make_root(&mut fs);
        let _ = make_file(&mut fs, ROOT_INO, b"src");
        let _ = make_file(&mut fs, ROOT_INO, b"dst");
        match fs.rename(ROOT_INO, b"src", ROOT_INO, b"dst") {
            Err(FsError::AlreadyExists) => {}
            other => panic!("expected AlreadyExists, got {other:?}"),
        }
    }

    #[test]
    fn unlink_then_recreate_with_same_name() {
        // Tombstone semantics: after delete + insert, same logical key has a
        // single Live entry visible.
        let mut fs = Fs::new();
        make_root(&mut fs);
        let _ = make_file(&mut fs, ROOT_INO, b"f");
        fs.unlink(ROOT_INO, b"f").unwrap();
        assert_eq!(fs.lookup_dirent(ROOT_INO, b"f").unwrap(), None);
        let new_ino = make_file(&mut fs, ROOT_INO, b"f");
        let d = fs.lookup_dirent(ROOT_INO, b"f").unwrap().unwrap();
        assert_eq!(d.target_ino, new_ino);
    }

    // ---------- Phase 8: snapshot create ----------

    #[test]
    fn snapshot_subvol_returns_new_subvol_with_readonly_view() {
        let mut fs = Fs::new();
        make_root(&mut fs);
        let _ = make_file(&mut fs, ROOT_INO, b"a");

        let snap_subvol = fs.snapshot_subvol(ROOT_SUBVOL).unwrap();
        // The new subvol exists, is readonly, and points at a new snap_id.
        let dst = fs.get_subvol(snap_subvol).unwrap().unwrap();
        assert_eq!(dst.parent_subvol, ROOT_SUBVOL);
        assert!(dst.flags & SUBVOL_FLAG_READONLY != 0);
        // src subvol's snap_id flipped to a fresh writable id.
        let src = fs.get_subvol(ROOT_SUBVOL).unwrap().unwrap();
        assert_ne!(src.snap_id, ROOT_SNAP);
        assert_ne!(src.snap_id, dst.snap_id);
        // Both new ids are children of the original ROOT_SNAP.
        let s_meta = fs.get_snapshot(src.snap_id).unwrap().unwrap();
        let d_meta = fs.get_snapshot(dst.snap_id).unwrap().unwrap();
        assert_eq!(s_meta.parent_id, ROOT_SNAP);
        assert_eq!(d_meta.parent_id, ROOT_SNAP);
    }

    #[test]
    fn snapshot_preserves_view_of_pre_snap_data() {
        // Both src and snap subvol see the file that existed before the snap.
        let mut fs = Fs::new();
        make_root(&mut fs);
        let ino = make_file(&mut fs, ROOT_INO, b"shared");

        let snap_subvol = fs.snapshot_subvol(ROOT_SUBVOL).unwrap();

        // src subvol still sees it.
        let d = fs.lookup_dirent(ROOT_INO, b"shared").unwrap().unwrap();
        assert_eq!(d.target_ino, ino);

        // Switch to the snapshot subvol — same view.
        fs.switch_subvol(snap_subvol).unwrap();
        let d = fs.lookup_dirent(ROOT_INO, b"shared").unwrap().unwrap();
        assert_eq!(d.target_ino, ino);
    }

    #[test]
    fn writes_after_snap_are_invisible_to_snapshot() {
        let mut fs = Fs::new();
        make_root(&mut fs);
        let snap_subvol = fs.snapshot_subvol(ROOT_SUBVOL).unwrap();

        // Write in src after taking the snap.
        let _ = make_file(&mut fs, ROOT_INO, b"new_file");

        // src sees it.
        assert!(fs.lookup_dirent(ROOT_INO, b"new_file").unwrap().is_some());
        // snapshot does NOT see it.
        fs.switch_subvol(snap_subvol).unwrap();
        assert!(fs.lookup_dirent(ROOT_INO, b"new_file").unwrap().is_none());
    }

    #[test]
    fn delete_after_snap_keeps_snapshot_view() {
        // Create file at ROOT_SNAP, take snap, then delete in src.
        // Snapshot must still show the file (Whiteout in src shadows, but
        // the snapshot's chain doesn't include src's new snap_id).
        let mut fs = Fs::new();
        make_root(&mut fs);
        let _ = make_file(&mut fs, ROOT_INO, b"victim");

        let snap_subvol = fs.snapshot_subvol(ROOT_SUBVOL).unwrap();

        // Delete in src.
        fs.unlink(ROOT_INO, b"victim").unwrap();
        assert!(fs.lookup_dirent(ROOT_INO, b"victim").unwrap().is_none());

        // Switch to snapshot — file still visible.
        fs.switch_subvol(snap_subvol).unwrap();
        assert!(fs.lookup_dirent(ROOT_INO, b"victim").unwrap().is_some());
    }

    #[test]
    fn switch_subvol_rejects_unknown_id() {
        let mut fs = Fs::new();
        match fs.switch_subvol(9999) {
            Err(FsError::NotFound) => {}
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    // ---------- Image-backed persistence ----------

    /// Helper: build a unique image path under the test tmpdir. Cleaned up
    /// at the end of the test (Drop on the returned PathGuard).
    struct PathGuard(std::path::PathBuf);
    impl Drop for PathGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    fn tmp_image_path(label: &str) -> PathGuard {
        let pid = std::process::id();
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let mut p = std::env::temp_dir();
        p.push(format!("rfs-{label}-{pid}-{now_ns}.img"));
        PathGuard(p)
    }

    #[test]
    fn image_round_trip_basic() {
        // Create a fresh image, write a directory + file + extents,
        // close, reopen, and verify everything is still there.
        let img = tmp_image_path("rt-basic");
        // ---- create + populate
        {
            let mut fs = Fs::create(&img.0).unwrap();
            // Bootstrap a root inode (FuseFs would do this, but Fs alone
            // doesn't — alloc_ino()=1 is the root).
            let root_ino = fs.alloc_ino();
            fs.put_inode(root_ino, &sample_inode(0)).unwrap();
            // Add a file inode + dirent + extent.
            let file_ino = fs.alloc_ino();
            fs.put_inode(file_ino, &sample_inode(11)).unwrap();
            fs.put_dirent(
                root_ino,
                b"hello.txt",
                &DirentV1 {
                    target_ino: file_ino,
                    kind: FILE_KIND_REGULAR,
                    _pad: [0; 7],
                },
            )
            .unwrap();
            fs.put_extent(file_ino, 0, b"hello world").unwrap();
            fs.sync().unwrap();
        }
        // ---- reopen + verify
        {
            let fs = Fs::open(&img.0).unwrap();
            // Inodes survive.
            let inode = fs.get_inode(2).unwrap().expect("file inode missing");
            assert_eq!(inode.size, 11);
            // Dirent survives + points at the file.
            let dirent = fs
                .lookup_dirent(1, b"hello.txt")
                .unwrap()
                .expect("dirent missing");
            assert_eq!(dirent.target_ino, 2);
            assert_eq!(dirent.kind, FILE_KIND_REGULAR);
            // Extent survives + the data block reads back.
            let ext = fs.get_extent(2, 0).unwrap().expect("extent missing");
            assert_eq!(ext.len, 11);
            let block = fs.read_data_block(ext.data_block).unwrap();
            assert_eq!(&block[..ext.len as usize], b"hello world");
        }
    }

    #[test]
    fn image_round_trip_preserves_allocator_counters() {
        // Allocate enough inodes to bump next_ino past ROOT, sync, reopen,
        // and check the next alloc continues from where it left off.
        let img = tmp_image_path("rt-alloc");
        {
            let mut fs = Fs::create(&img.0).unwrap();
            for _ in 0..10 {
                let _ = fs.alloc_ino();
            }
            fs.sync().unwrap();
        }
        let mut fs = Fs::open(&img.0).unwrap();
        let next = fs.alloc_ino();
        assert_eq!(
            next, 11,
            "alloc_ino must resume after persisted counter (10 alloc + ROOT_INO=1)"
        );
    }

    #[test]
    fn open_rejects_bad_magic() {
        // Create + sync, then corrupt the superblock magic and try to open.
        let img = tmp_image_path("bad-magic");
        {
            let fs = Fs::create(&img.0).unwrap();
            fs.sync().unwrap();
        }
        // Stomp on the magic word.
        {
            use std::io::{Seek, SeekFrom, Write};
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .open(&img.0)
                .unwrap();
            f.seek(SeekFrom::Start(0)).unwrap();
            f.write_all(&[0xff; 4]).unwrap();
        }
        match Fs::open(&img.0) {
            Err(crate::btree::Error::BadMagic { block: 0, .. }) => {}
            Err(other) => panic!("expected BadMagic at block 0, got {other:?}"),
            Ok(_) => panic!("expected BadMagic, got Ok(Fs)"),
        }
    }

    #[test]
    fn open_rejects_checksum_mismatch_on_node() {
        // Create + sync, flip a byte inside a node block (the persisted
        // root_block), then re-read via BlockStore — should surface
        // ChecksumMismatch with the right block number. We don't go
        // through Fs here because most Fs ops use `expect()` on a
        // Result and would panic instead of returning the typed error.
        let img = tmp_image_path("bad-crc");
        let root_block;
        {
            let mut fs = Fs::create(&img.0).unwrap();
            fs.put_inode(1, &sample_inode(0)).unwrap();
            fs.sync().unwrap();
            root_block = {
                let store = crate::storage::BlockStore::open_image(&img.0).unwrap();
                store.read_superblock().unwrap().root_block
            };
        }
        // Corrupt one byte deep in the body so magic still passes but CRC
        // does not.
        {
            use std::io::{Seek, SeekFrom, Write};
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .open(&img.0)
                .unwrap();
            f.seek(SeekFrom::Start(root_block * 4096 + 1024)).unwrap();
            f.write_all(&[0xaa]).unwrap();
        }
        // Reopen the store and read the corrupted node.
        let store = crate::storage::BlockStore::open_image(&img.0).unwrap();
        match store.read_node(root_block) {
            Err(crate::btree::Error::ChecksumMismatch { block }) => {
                assert_eq!(block, root_block);
            }
            other => panic!(
                "expected ChecksumMismatch on block {root_block}, got {:?}",
                other.map(|_| "Ok(node)")
            ),
        }
    }
}
