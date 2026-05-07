use std::collections::HashMap;

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::block_btree::{MAX_KEY_SIZE, MAX_VALUE_SIZE};
use crate::btree::{Btree, Result};

const BLOCK_SIZE: usize = 4096;

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

const INODE_KEY_LEN: usize = 1 + 8;
const DIRENT_PREFIX_LEN: usize = 1 + 8;
const EXTENT_KEY_LEN: usize = 1 + 8 + 8;

/// Longest dirent name that still fits in MAX_KEY_SIZE.
pub const MAX_NAME_LEN: usize = MAX_KEY_SIZE - DIRENT_PREFIX_LEN;

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
    (
        extent_key(ino, 0),
        extent_key(ino.saturating_add(1), 0),
    )
}

fn extent_offset_from_key(key: &[u8]) -> u64 {
    u64::from_be_bytes(key[9..17].try_into().unwrap())
}

fn dirent_name_from_key(key: &[u8]) -> &[u8] {
    &key[DIRENT_PREFIX_LEN..]
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
        DirentV1 { target_ino, kind, _pad: [0; 7] }
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

// ---------- Fs ----------

pub struct Fs {
    tree: Btree,
    data_blocks: HashMap<u64, [u8; BLOCK_SIZE]>,
    next_ino: u64,
    next_data_block: u64,
}

impl Fs {
    pub fn new() -> Self {
        Fs {
            tree: Btree::new(),
            data_blocks: HashMap::new(),
            next_ino: ROOT_INO,
            next_data_block: 0,
        }
    }

    pub fn alloc_ino(&mut self) -> u64 {
        let ino = self.next_ino;
        self.next_ino += 1;
        ino
    }

    fn alloc_data_block(&mut self) -> u64 {
        let nr = self.next_data_block;
        self.next_data_block += 1;
        self.data_blocks.insert(nr, [0u8; BLOCK_SIZE]);
        nr
    }

    // -- Inode --

    pub fn put_inode(&mut self, ino: u64, inode: &InodeV1) -> Result<()> {
        self.tree.insert(&inode_key(ino), inode.as_bytes())
    }

    pub fn get_inode(&self, ino: u64) -> Result<Option<InodeV1>> {
        let bytes = self.tree.find(&inode_key(ino))?;
        Ok(bytes.map(|b| {
            InodeV1::read_from_bytes(&b).expect("inode value size mismatch")
        }))
    }

    // -- Dirent --

    pub fn put_dirent(&mut self, parent: u64, name: &[u8], d: &DirentV1) -> Result<()> {
        self.tree.insert(&dirent_key(parent, name), d.as_bytes())
    }

    pub fn lookup_dirent(&self, parent: u64, name: &[u8]) -> Result<Option<DirentV1>> {
        let bytes = self.tree.find(&dirent_key(parent, name))?;
        Ok(bytes.map(|b| {
            DirentV1::read_from_bytes(&b).expect("dirent value size mismatch")
        }))
    }

    pub fn list_dirents(&self, parent: u64) -> Result<Vec<(Vec<u8>, DirentV1)>> {
        let (start, end) = dirent_range(parent);
        let entries = self.tree.range_scan(&start, &end)?;
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
        let block_nr = self.alloc_data_block();
        let block = self.data_blocks.get_mut(&block_nr).unwrap();
        block[..data.len()].copy_from_slice(data);
        let extent = ExtentV1 {
            len: data.len() as u32,
            _pad: [0; 4],
            data_block: block_nr,
        };
        self.tree.insert(&extent_key(ino, offset), extent.as_bytes())
    }

    pub fn get_extent(&self, ino: u64, offset: u64) -> Result<Option<ExtentV1>> {
        let bytes = self.tree.find(&extent_key(ino, offset))?;
        Ok(bytes.map(|b| {
            ExtentV1::read_from_bytes(&b).expect("extent value size mismatch")
        }))
    }

    pub fn read_data_block(&self, block_nr: u64) -> &[u8; BLOCK_SIZE] {
        &self.data_blocks[&block_nr]
    }

    pub fn list_extents(&self, ino: u64) -> Result<Vec<(u64, ExtentV1)>> {
        let (start, end) = extent_range(ino);
        let entries = self.tree.range_scan(&start, &end)?;
        Ok(entries
            .into_iter()
            .map(|(k, v)| {
                let offset = extent_offset_from_key(&k);
                let extent = ExtentV1::read_from_bytes(&v).expect("extent value size mismatch");
                (offset, extent)
            })
            .collect())
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
        fs.put_dirent(1, b"x", &DirentV1::new(100, FILE_KIND_REGULAR)).unwrap();
        fs.put_dirent(2, b"x", &DirentV1::new(200, FILE_KIND_REGULAR)).unwrap();
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
        let block = fs.read_data_block(ext.data_block);
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
        assert_eq!(&fs.read_data_block(ino5[0].1.data_block)[..3], b"aaa");
        assert_eq!(&fs.read_data_block(ino6[0].1.data_block)[..3], b"bbb");
    }

    #[test]
    fn kinds_isolated_within_same_id() {
        // ino=5 has an inode, a dirent subtree (parent=5), and extents (ino=5).
        // None of the three range scans should see keys from the others.
        let mut fs = Fs::new();
        fs.put_inode(5, &sample_inode(100)).unwrap();
        fs.put_dirent(5, b"child", &DirentV1::new(9, FILE_KIND_REGULAR)).unwrap();
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
}
