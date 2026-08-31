// libc's S_IF* constants are u32 on Linux but u16 on macOS; keep the casts
// explicit so this compiles the same way on either.
#![allow(clippy::unnecessary_cast)]

use std::ffi::OsStr;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use fuser::{
    Errno, FileAttr, FileHandle, FileType, Filesystem, FopenFlags, Generation, INodeNo, LockOwner,
    OpenFlags, RenameFlags, ReplyAttr, ReplyCreate, ReplyData, ReplyDirectory, ReplyEmpty,
    ReplyEntry, ReplyOpen, ReplyStatfs, ReplyWrite, Request, TimeOrNow, WriteFlags,
};
use libc::{S_IFDIR, S_IFLNK, S_IFMT, S_IFREG};

use crate::btree;
use crate::fs::{
    DirentV1, FILE_KIND_DIR, FILE_KIND_REGULAR, FILE_KIND_SYMLINK, Fs, FsError, InodeV1,
    MAX_NAME_LEN, ROOT_INO,
};

const BLOCK_SIZE: u64 = 4096;
const BLOCK_SIZE_USIZE: usize = BLOCK_SIZE as usize;
const TTL: Duration = Duration::from_secs(1);
const GENERATION: Generation = Generation(0);

/// Wraps `Fs` with interior mutability. The fuser 0.17 `Filesystem` trait is
/// `Send + Sync + 'static` with `&self` methods; a single `Mutex` is enough for
/// our demo since the event loop is single-threaded by default.
pub struct FuseFs {
    fs: Mutex<Fs>,
}

impl FuseFs {
    /// Pure-RAM filesystem. Bootstraps a root inode.
    pub fn new() -> Self {
        let fs = Fs::new();
        Self::bootstrap_root(fs)
    }

    /// Create a brand-new image file at `path` and mount on top of it.
    /// Bootstraps a root inode just like `new`.
    pub fn create_image(path: &std::path::Path) -> btree::Result<Self> {
        let fs = Fs::create(path)?;
        Ok(Self::bootstrap_root(fs))
    }

    /// Mount on top of an existing image. The root inode is already there;
    /// we do not bootstrap it.
    pub fn open_image(path: &std::path::Path) -> btree::Result<Self> {
        let fs = Fs::open(path)?;
        Ok(FuseFs { fs: Mutex::new(fs) })
    }

    fn bootstrap_root(mut fs: Fs) -> Self {
        // First alloc returns ROOT_INO=1 by construction; use it for the root.
        let root_ino = fs.alloc_ino();
        assert_eq!(root_ino, ROOT_INO);
        let now = now_secs();
        let uid = unsafe { libc::getuid() };
        let gid = unsafe { libc::getgid() };
        let root = InodeV1 {
            mode: S_IFDIR as u32 | 0o755,
            uid,
            gid,
            nlink: 2,
            size: 0,
            atime: now,
            mtime: now,
            ctime: now,
            parent_ino: ROOT_INO,
        };
        fs.put_inode(ROOT_INO, &root).expect("bootstrap root inode");
        FuseFs { fs: Mutex::new(fs) }
    }
}

impl Default for FuseFs {
    fn default() -> Self {
        Self::new()
    }
}

// ---------- Pure helpers (unit-tested below) ----------

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn to_attr(ino: u64, inode: &InodeV1) -> FileAttr {
    let kind = match inode.mode & S_IFMT as u32 {
        x if x == S_IFDIR as u32 => FileType::Directory,
        x if x == S_IFLNK as u32 => FileType::Symlink,
        _ => FileType::RegularFile,
    };
    let to_st = |s: u64| UNIX_EPOCH + Duration::from_secs(s);
    FileAttr {
        ino: INodeNo(ino),
        size: inode.size,
        blocks: inode.size.div_ceil(BLOCK_SIZE),
        atime: to_st(inode.atime),
        mtime: to_st(inode.mtime),
        ctime: to_st(inode.ctime),
        crtime: UNIX_EPOCH,
        kind,
        perm: (inode.mode & 0o7777) as u16,
        nlink: inode.nlink,
        uid: inode.uid,
        gid: inode.gid,
        rdev: 0,
        blksize: BLOCK_SIZE as u32,
        flags: 0,
    }
}

/// Read `[offset, offset+size)` from `ino`, clipped to inode.size, zero-filling
/// gaps between extents.
///
/// `pub` only so the differential fuzz target can drive it directly; not a
/// stable API.
#[doc(hidden)]
pub fn do_read(fs: &Fs, ino: u64, offset: u64, size: u32) -> btree::Result<Vec<u8>> {
    let Some(inode) = fs.get_inode(ino)? else {
        return Ok(Vec::new());
    };
    if offset >= inode.size {
        return Ok(Vec::new());
    }
    let end = (offset + size as u64).min(inode.size);
    let out_len = (end - offset) as usize;
    let mut out = vec![0u8; out_len];
    // Scan only the extents overlapping [offset, end) instead of the whole
    // inode. Extents are keyed on their block boundary, so the one covering
    // `offset` has key `offset & !(BLOCK_SIZE-1)`; start the window there so it
    // isn't clipped by the lower bound.
    let scan_start = offset & !(BLOCK_SIZE - 1);
    for (ext_off, ext) in fs.extents_in_range(ino, scan_start, end)? {
        let ext_end = ext_off + ext.len as u64;
        if ext_end <= offset || ext_off >= end {
            continue;
        }
        let copy_start = offset.max(ext_off);
        let copy_end = end.min(ext_end);
        let src_block = fs
            .read_data_block(ext.data_block)
            .expect("data block missing");
        let src = &src_block[(copy_start - ext_off) as usize..(copy_end - ext_off) as usize];
        let dst_off = (copy_start - offset) as usize;
        out[dst_off..dst_off + src.len()].copy_from_slice(src);
    }
    Ok(out)
}

/// Write `data` at `offset` into `ino`, split into BLOCK_SIZE-aligned chunks.
/// Each chunk is read-modify-write against any existing extent at that
/// block offset so partial-block writes preserve surrounding bytes. Returns
/// the number of bytes written (== data.len()).
fn do_write(fs: &mut Fs, ino: u64, offset: u64, data: &[u8]) -> btree::Result<usize> {
    let total = data.len();
    let mut cursor = offset;
    let mut remaining = data;
    while !remaining.is_empty() {
        let block_off = cursor & !(BLOCK_SIZE - 1);
        let in_block = (cursor - block_off) as usize;
        let take = (BLOCK_SIZE_USIZE - in_block).min(remaining.len());

        let mut buf = [0u8; BLOCK_SIZE_USIZE];
        let mut existing_len = 0;
        // `block_off` is block-aligned, so this exact-match lookup hits the
        // extent covering the block (if any) for a read-modify-write. See
        // `Fs::get_extent` for why this depends on the one-extent-per-block
        // keying and what changes under variable-length extents.
        if let Some(ext) = fs.get_extent(ino, block_off)? {
            existing_len = ext.len as usize;
            let src = fs
                .read_data_block(ext.data_block)
                .expect("data block missing");
            buf[..existing_len].copy_from_slice(&src[..existing_len]);
        }
        buf[in_block..in_block + take].copy_from_slice(&remaining[..take]);
        let new_len = (in_block + take).max(existing_len);
        fs.put_extent(ino, block_off, &buf[..new_len])?;

        cursor += take as u64;
        remaining = &remaining[take..];
    }
    Ok(total)
}

/// Reject names that can't be stored as a dirent key or are reserved.
fn validate_name(name: &OsStr) -> Result<&[u8], Errno> {
    let bytes = name.as_encoded_bytes();
    if bytes.is_empty() || bytes == b"." || bytes == b".." {
        return Err(Errno::EINVAL);
    }
    if bytes.contains(&b'/') || bytes.contains(&0) {
        return Err(Errno::EINVAL);
    }
    if bytes.len() > MAX_NAME_LEN {
        return Err(Errno::ENAMETOOLONG);
    }
    Ok(bytes)
}

/// Truncate file extents beyond `new_size`. Removes whole blocks past the
/// boundary and zero-fills the partial tail block.
fn truncate_extents(fs: &mut Fs, ino: u64, new_size: u64) -> btree::Result<()> {
    let boundary_block = new_size & !(BLOCK_SIZE - 1);
    let tail_offset = (new_size % BLOCK_SIZE) as usize;

    for (ext_off, ext) in fs.list_extents(ino)? {
        if ext_off >= new_size {
            fs.delete_extent(ino, ext_off)?;
            fs.free_data_block(ext.data_block);
        } else if ext_off == boundary_block
            && tail_offset > 0
            && (ext.len as u64) > new_size - ext_off
        {
            let mut buf = fs.read_data_block(ext.data_block)?;
            for b in &mut buf[tail_offset..ext.len as usize] {
                *b = 0;
            }
            fs.put_extent(ino, ext_off, &buf[..tail_offset])?;
        }
    }
    Ok(())
}

/// Map an FsError to a POSIX errno for FUSE replies.
fn errno_for_fs_error(e: FsError) -> Errno {
    match e {
        FsError::NotFound => Errno::ENOENT,
        FsError::NotADirectory => Errno::ENOTDIR,
        FsError::NotEmpty => Errno::ENOTEMPTY,
        FsError::AlreadyExists => Errno::EEXIST,
        FsError::Exhausted => Errno::ENOSPC,
        FsError::Invalid => Errno::EINVAL,
        FsError::Busy => Errno::EBUSY,
        FsError::Btree(_) => Errno::EIO,
    }
}

// ---------- Filesystem trait impl ----------

impl Filesystem for FuseFs {
    fn lookup(&self, _req: &Request, parent: INodeNo, name: &OsStr, reply: ReplyEntry) {
        let parent = parent.0;
        let fs = self.fs.lock().unwrap();
        let raw = name.as_encoded_bytes();
        if raw == b"." || raw == b".." {
            let parent_inode = match fs.get_inode(parent) {
                Ok(Some(i)) => i,
                Ok(None) => {
                    reply.error(Errno::ENOENT);
                    return;
                }
                Err(_) => {
                    reply.error(Errno::EIO);
                    return;
                }
            };
            let (target_ino, target_inode) = if raw == b"." {
                (parent, parent_inode)
            } else {
                let pp = parent_inode.parent_ino;
                match fs.get_inode(pp) {
                    Ok(Some(i)) => (pp, i),
                    Ok(None) => {
                        reply.error(Errno::ENOENT);
                        return;
                    }
                    Err(_) => {
                        reply.error(Errno::EIO);
                        return;
                    }
                }
            };
            reply.entry(&TTL, &to_attr(target_ino, &target_inode), GENERATION);
            return;
        }
        let bytes = match validate_name(name) {
            Ok(b) => b,
            Err(e) => {
                reply.error(e);
                return;
            }
        };
        match fs.lookup_dirent(parent, bytes) {
            Ok(Some(d)) => match fs.get_inode(d.target_ino) {
                Ok(Some(inode)) => reply.entry(&TTL, &to_attr(d.target_ino, &inode), GENERATION),
                Ok(None) => reply.error(Errno::EIO),
                Err(_) => reply.error(Errno::EIO),
            },
            Ok(None) => reply.error(Errno::ENOENT),
            Err(_) => reply.error(Errno::EIO),
        }
    }

    fn getattr(&self, _req: &Request, ino: INodeNo, _fh: Option<FileHandle>, reply: ReplyAttr) {
        let ino = ino.0;
        let fs = self.fs.lock().unwrap();
        match fs.get_inode(ino) {
            Ok(Some(inode)) => reply.attr(&TTL, &to_attr(ino, &inode)),
            Ok(None) => reply.error(Errno::ENOENT),
            Err(_) => reply.error(Errno::EIO),
        }
    }

    fn readdir(
        &self,
        _req: &Request,
        ino: INodeNo,
        _fh: FileHandle,
        offset: u64,
        mut reply: ReplyDirectory,
    ) {
        let ino = ino.0;
        let fs = self.fs.lock().unwrap();
        let inode = match fs.get_inode(ino) {
            Ok(Some(i)) => i,
            Ok(None) => {
                reply.error(Errno::ENOENT);
                return;
            }
            Err(_) => {
                reply.error(Errno::EIO);
                return;
            }
        };
        if inode.mode & S_IFMT as u32 != S_IFDIR as u32 {
            reply.error(Errno::ENOTDIR);
            return;
        }
        let dirents = match fs.list_dirents(ino) {
            Ok(v) => v,
            Err(_) => {
                reply.error(Errno::EIO);
                return;
            }
        };

        let mut entries: Vec<(u64, FileType, Vec<u8>)> = Vec::with_capacity(2 + dirents.len());
        entries.push((ino, FileType::Directory, b".".to_vec()));
        entries.push((inode.parent_ino, FileType::Directory, b"..".to_vec()));
        for (name, d) in dirents {
            let kind = match d.kind {
                FILE_KIND_DIR => FileType::Directory,
                FILE_KIND_SYMLINK => FileType::Symlink,
                _ => FileType::RegularFile,
            };
            entries.push((d.target_ino, kind, name));
        }

        for (i, (child_ino, kind, name)) in entries.into_iter().enumerate().skip(offset as usize) {
            let name_os = OsStr::from_bytes(&name);
            if reply.add(INodeNo(child_ino), (i + 1) as u64, kind, name_os) {
                break;
            }
        }
        reply.ok();
    }

    fn read(
        &self,
        _req: &Request,
        ino: INodeNo,
        _fh: FileHandle,
        offset: u64,
        size: u32,
        _flags: OpenFlags,
        _lock: Option<LockOwner>,
        reply: ReplyData,
    ) {
        let fs = self.fs.lock().unwrap();
        match do_read(&fs, ino.0, offset, size) {
            Ok(buf) => reply.data(&buf),
            Err(_) => reply.error(Errno::EIO),
        }
    }

    fn write(
        &self,
        _req: &Request,
        ino: INodeNo,
        _fh: FileHandle,
        offset: u64,
        data: &[u8],
        _write_flags: WriteFlags,
        _flags: OpenFlags,
        _lock_owner: Option<LockOwner>,
        reply: ReplyWrite,
    ) {
        let ino = ino.0;
        let mut fs = self.fs.lock().unwrap();
        let written = match do_write(&mut fs, ino, offset, data) {
            Ok(n) => n,
            Err(_) => {
                reply.error(Errno::EIO);
                return;
            }
        };
        let mut inode = match fs.get_inode(ino) {
            Ok(Some(i)) => i,
            Ok(None) => {
                reply.error(Errno::ENOENT);
                return;
            }
            Err(_) => {
                reply.error(Errno::EIO);
                return;
            }
        };
        inode.size = inode.size.max(offset + data.len() as u64);
        let now = now_secs();
        inode.mtime = now;
        inode.ctime = now;
        if fs.put_inode(ino, &inode).is_err() {
            reply.error(Errno::EIO);
            return;
        }
        if fs.journal_commit().is_err() {
            reply.error(Errno::EIO);
            return;
        }
        reply.written(written as u32);
    }

    fn create(
        &self,
        req: &Request,
        parent: INodeNo,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        _flags: i32,
        reply: ReplyCreate,
    ) {
        let parent = parent.0;
        let mut fs = self.fs.lock().unwrap();
        let name_bytes = match validate_name(name) {
            Ok(b) => b,
            Err(e) => {
                reply.error(e);
                return;
            }
        };
        match fs.lookup_dirent(parent, name_bytes) {
            Ok(Some(_)) => {
                reply.error(Errno::EEXIST);
                return;
            }
            Ok(None) => {}
            Err(_) => {
                reply.error(Errno::EIO);
                return;
            }
        }
        let new_ino = fs.alloc_ino();
        let now = now_secs();
        let inode = InodeV1 {
            mode: S_IFREG as u32 | (mode & 0o7777),
            uid: req.uid(),
            gid: req.gid(),
            nlink: 1,
            size: 0,
            atime: now,
            mtime: now,
            ctime: now,
            parent_ino: parent,
        };
        if fs.put_inode(new_ino, &inode).is_err() {
            reply.error(Errno::EIO);
            return;
        }
        let d = DirentV1::new(new_ino, FILE_KIND_REGULAR);
        if fs.put_dirent(parent, name_bytes, &d).is_err() {
            reply.error(Errno::EIO);
            return;
        }
        if fs.journal_commit().is_err() {
            reply.error(Errno::EIO);
            return;
        }
        reply.created(
            &TTL,
            &to_attr(new_ino, &inode),
            GENERATION,
            FileHandle(0),
            FopenFlags::empty(),
        );
    }

    fn mkdir(
        &self,
        req: &Request,
        parent: INodeNo,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        reply: ReplyEntry,
    ) {
        let parent = parent.0;
        let mut fs = self.fs.lock().unwrap();
        let name_bytes = match validate_name(name) {
            Ok(b) => b,
            Err(e) => {
                reply.error(e);
                return;
            }
        };
        match fs.lookup_dirent(parent, name_bytes) {
            Ok(Some(_)) => {
                reply.error(Errno::EEXIST);
                return;
            }
            Ok(None) => {}
            Err(_) => {
                reply.error(Errno::EIO);
                return;
            }
        }
        let new_ino = fs.alloc_ino();
        let now = now_secs();
        let inode = InodeV1 {
            mode: S_IFDIR as u32 | (mode & 0o7777),
            uid: req.uid(),
            gid: req.gid(),
            nlink: 2,
            size: 0,
            atime: now,
            mtime: now,
            ctime: now,
            parent_ino: parent,
        };
        if fs.put_inode(new_ino, &inode).is_err() {
            reply.error(Errno::EIO);
            return;
        }
        let d = DirentV1::new(new_ino, FILE_KIND_DIR);
        if fs.put_dirent(parent, name_bytes, &d).is_err() {
            reply.error(Errno::EIO);
            return;
        }
        if fs.journal_commit().is_err() {
            reply.error(Errno::EIO);
            return;
        }
        reply.entry(&TTL, &to_attr(new_ino, &inode), GENERATION);
    }

    fn unlink(&self, _req: &Request, parent: INodeNo, name: &OsStr, reply: ReplyEmpty) {
        let parent = parent.0;
        let mut fs = self.fs.lock().unwrap();
        let name_bytes = match validate_name(name) {
            Ok(b) => b,
            Err(e) => {
                reply.error(e);
                return;
            }
        };
        match fs.unlink(parent, name_bytes) {
            Ok(()) => {
                if fs.journal_commit().is_err() {
                    reply.error(Errno::EIO);
                    return;
                }
                reply.ok();
            }
            Err(e) => reply.error(errno_for_fs_error(e)),
        }
    }

    fn rmdir(&self, _req: &Request, parent: INodeNo, name: &OsStr, reply: ReplyEmpty) {
        let parent = parent.0;
        let mut fs = self.fs.lock().unwrap();
        let name_bytes = match validate_name(name) {
            Ok(b) => b,
            Err(e) => {
                reply.error(e);
                return;
            }
        };
        match fs.rmdir(parent, name_bytes) {
            Ok(()) => {
                if fs.journal_commit().is_err() {
                    reply.error(Errno::EIO);
                    return;
                }
                reply.ok();
            }
            Err(e) => reply.error(errno_for_fs_error(e)),
        }
    }

    fn rename(
        &self,
        _req: &Request,
        parent: INodeNo,
        name: &OsStr,
        newparent: INodeNo,
        newname: &OsStr,
        _flags: RenameFlags,
        reply: ReplyEmpty,
    ) {
        let parent = parent.0;
        let newparent = newparent.0;
        let mut fs = self.fs.lock().unwrap();
        let name_bytes = match validate_name(name) {
            Ok(b) => b,
            Err(e) => {
                reply.error(e);
                return;
            }
        };
        let newname_bytes = match validate_name(newname) {
            Ok(b) => b,
            Err(e) => {
                reply.error(e);
                return;
            }
        };
        match fs.rename(parent, name_bytes, newparent, newname_bytes) {
            Ok(()) => {
                if fs.journal_commit().is_err() {
                    reply.error(Errno::EIO);
                    return;
                }
                reply.ok();
            }
            Err(e) => reply.error(errno_for_fs_error(e)),
        }
    }

    fn setattr(
        &self,
        _req: &Request,
        ino: INodeNo,
        mode: Option<u32>,
        uid: Option<u32>,
        gid: Option<u32>,
        size: Option<u64>,
        atime: Option<TimeOrNow>,
        mtime: Option<TimeOrNow>,
        _ctime: Option<SystemTime>,
        _fh: Option<FileHandle>,
        _crtime: Option<SystemTime>,
        _chgtime: Option<SystemTime>,
        _bkuptime: Option<SystemTime>,
        _flags: Option<fuser::BsdFileFlags>,
        reply: ReplyAttr,
    ) {
        let ino = ino.0;
        let mut fs = self.fs.lock().unwrap();
        let mut inode = match fs.get_inode(ino) {
            Ok(Some(i)) => i,
            Ok(None) => {
                reply.error(Errno::ENOENT);
                return;
            }
            Err(_) => {
                reply.error(Errno::EIO);
                return;
            }
        };
        if let Some(m) = mode {
            inode.mode = (inode.mode & S_IFMT as u32) | (m & 0o7777);
        }
        if let Some(u) = uid {
            inode.uid = u;
        }
        if let Some(g) = gid {
            inode.gid = g;
        }
        if let Some(new_size) = size {
            if new_size < inode.size && truncate_extents(&mut fs, ino, new_size).is_err() {
                reply.error(Errno::EIO);
                return;
            }
            inode.size = new_size;
        }
        let now = now_secs();
        match atime {
            Some(TimeOrNow::SpecificTime(t)) => {
                inode.atime = t
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
            }
            Some(TimeOrNow::Now) => inode.atime = now,
            None => {}
        }
        match mtime {
            Some(TimeOrNow::SpecificTime(t)) => {
                inode.mtime = t
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
            }
            Some(TimeOrNow::Now) => inode.mtime = now,
            None => {}
        }
        inode.ctime = now;
        if fs.put_inode(ino, &inode).is_err() {
            reply.error(Errno::EIO);
            return;
        }
        if fs.journal_commit().is_err() {
            reply.error(Errno::EIO);
            return;
        }
        reply.attr(&TTL, &to_attr(ino, &inode));
    }

    fn symlink(
        &self,
        req: &Request,
        parent: INodeNo,
        link_name: &OsStr,
        target: &Path,
        reply: ReplyEntry,
    ) {
        let parent = parent.0;
        let mut fs = self.fs.lock().unwrap();
        let name_bytes = match validate_name(link_name) {
            Ok(b) => b,
            Err(e) => {
                reply.error(e);
                return;
            }
        };
        if let Ok(Some(_)) = fs.lookup_dirent(parent, name_bytes) {
            reply.error(Errno::EEXIST);
            return;
        }
        let target_bytes = target.as_os_str().as_encoded_bytes();
        if target_bytes.len() > BLOCK_SIZE_USIZE {
            reply.error(Errno::ENAMETOOLONG);
            return;
        }
        let new_ino = fs.alloc_ino();
        let now = now_secs();
        let inode = InodeV1 {
            mode: S_IFLNK as u32 | 0o777,
            uid: req.uid(),
            gid: req.gid(),
            nlink: 1,
            size: target_bytes.len() as u64,
            atime: now,
            mtime: now,
            ctime: now,
            parent_ino: parent,
        };
        if fs.put_inode(new_ino, &inode).is_err() {
            reply.error(Errno::EIO);
            return;
        }
        if fs.put_extent(new_ino, 0, target_bytes).is_err() {
            reply.error(Errno::EIO);
            return;
        }
        let d = DirentV1::new(new_ino, FILE_KIND_SYMLINK);
        if fs.put_dirent(parent, name_bytes, &d).is_err() {
            reply.error(Errno::EIO);
            return;
        }
        if fs.journal_commit().is_err() {
            reply.error(Errno::EIO);
            return;
        }
        reply.entry(&TTL, &to_attr(new_ino, &inode), GENERATION);
    }

    fn readlink(&self, _req: &Request, ino: INodeNo, reply: ReplyData) {
        let fs = self.fs.lock().unwrap();
        let inode = match fs.get_inode(ino.0) {
            Ok(Some(i)) => i,
            Ok(None) => {
                reply.error(Errno::ENOENT);
                return;
            }
            Err(_) => {
                reply.error(Errno::EIO);
                return;
            }
        };
        if inode.mode & S_IFMT as u32 != S_IFLNK as u32 {
            reply.error(Errno::EINVAL);
            return;
        }
        match do_read(&fs, ino.0, 0, inode.size as u32) {
            Ok(buf) => reply.data(&buf),
            Err(_) => reply.error(Errno::EIO),
        }
    }

    fn link(
        &self,
        _req: &Request,
        ino: INodeNo,
        newparent: INodeNo,
        newname: &OsStr,
        reply: ReplyEntry,
    ) {
        let ino = ino.0;
        let newparent = newparent.0;
        let mut fs = self.fs.lock().unwrap();
        let name_bytes = match validate_name(newname) {
            Ok(b) => b,
            Err(e) => {
                reply.error(e);
                return;
            }
        };
        let mut inode = match fs.get_inode(ino) {
            Ok(Some(i)) => i,
            Ok(None) => {
                reply.error(Errno::ENOENT);
                return;
            }
            Err(_) => {
                reply.error(Errno::EIO);
                return;
            }
        };
        if inode.mode & S_IFMT as u32 == S_IFDIR as u32 {
            reply.error(Errno::EPERM);
            return;
        }
        if let Ok(Some(_)) = fs.lookup_dirent(newparent, name_bytes) {
            reply.error(Errno::EEXIST);
            return;
        }
        inode.nlink += 1;
        inode.ctime = now_secs();
        if fs.put_inode(ino, &inode).is_err() {
            reply.error(Errno::EIO);
            return;
        }
        let kind = if inode.mode & S_IFMT as u32 == S_IFLNK as u32 {
            FILE_KIND_SYMLINK
        } else {
            FILE_KIND_REGULAR
        };
        let d = DirentV1::new(ino, kind);
        if fs.put_dirent(newparent, name_bytes, &d).is_err() {
            reply.error(Errno::EIO);
            return;
        }
        if fs.journal_commit().is_err() {
            reply.error(Errno::EIO);
            return;
        }
        reply.entry(&TTL, &to_attr(ino, &inode), GENERATION);
    }

    fn statfs(&self, _req: &Request, _ino: INodeNo, reply: ReplyStatfs) {
        let fs = self.fs.lock().unwrap();
        let total_blocks = fs.store_next_block_nr();
        let free_blocks = fs.free_block_count();
        reply.statfs(
            total_blocks,
            free_blocks,
            free_blocks,
            0,
            0,
            BLOCK_SIZE as u32,
            MAX_NAME_LEN as u32,
            BLOCK_SIZE as u32,
        );
    }

    fn fsync(
        &self,
        _req: &Request,
        _ino: INodeNo,
        _fh: FileHandle,
        _datasync: bool,
        reply: ReplyEmpty,
    ) {
        let mut fs = self.fs.lock().unwrap();
        match fs.journal_commit() {
            Ok(()) => reply.ok(),
            Err(_) => reply.error(Errno::EIO),
        }
    }

    fn open(&self, _req: &Request, _ino: INodeNo, _flags: OpenFlags, reply: ReplyOpen) {
        reply.opened(FileHandle(0), FopenFlags::empty());
    }

    fn opendir(&self, _req: &Request, _ino: INodeNo, _flags: OpenFlags, reply: ReplyOpen) {
        reply.opened(FileHandle(0), FopenFlags::empty());
    }

    fn release(
        &self,
        _req: &Request,
        _ino: INodeNo,
        _fh: FileHandle,
        _flags: OpenFlags,
        _lock_owner: Option<LockOwner>,
        _flush: bool,
        reply: ReplyEmpty,
    ) {
        reply.ok();
    }

    fn releasedir(
        &self,
        _req: &Request,
        _ino: INodeNo,
        _fh: FileHandle,
        _flags: OpenFlags,
        reply: ReplyEmpty,
    ) {
        reply.ok();
    }

    fn flush(
        &self,
        _req: &Request,
        _ino: INodeNo,
        _fh: FileHandle,
        _lock_owner: LockOwner,
        reply: ReplyEmpty,
    ) {
        reply.ok();
    }

    /// Called by FUSE when the filesystem is being unmounted. Best-effort
    /// sync: write a final superblock + fsync. On a memory-only mount this
    /// is a no-op (BlockStore::fsync is None-typed).
    fn destroy(&mut self) {
        if let Ok(mut fs) = self.fs.lock()
            && let Err(e) = fs.sync()
        {
            eprintln!("rfs: sync on unmount failed: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_inode(kind: u8) -> InodeV1 {
        let mode = match kind {
            FILE_KIND_DIR => S_IFDIR as u32 | 0o755,
            _ => S_IFREG as u32 | 0o644,
        };
        InodeV1 {
            mode,
            uid: 0,
            gid: 0,
            nlink: 1,
            size: 0,
            atime: 0,
            mtime: 0,
            ctime: 0,
            parent_ino: ROOT_INO,
        }
    }

    #[test]
    fn to_attr_maps_kind() {
        let dir = fresh_inode(FILE_KIND_DIR);
        assert_eq!(to_attr(1, &dir).kind, FileType::Directory);
        let reg = fresh_inode(FILE_KIND_REGULAR);
        assert_eq!(to_attr(2, &reg).kind, FileType::RegularFile);
    }

    #[test]
    fn to_attr_preserves_perm_and_size() {
        let mut i = fresh_inode(FILE_KIND_REGULAR);
        i.size = 10_000;
        let a = to_attr(42, &i);
        assert_eq!(a.ino, INodeNo(42));
        assert_eq!(a.perm, 0o644);
        assert_eq!(a.size, 10_000);
        // 10000 bytes → ceil(10000/4096) = 3 blocks
        assert_eq!(a.blocks, 3);
    }

    fn setup_fs_with_file(size_bytes: usize) -> (Fs, u64) {
        let mut fs = Fs::new();
        let ino = fs.alloc_ino(); // 1
        let inode = InodeV1 {
            mode: S_IFREG as u32 | 0o644,
            uid: 0,
            gid: 0,
            nlink: 1,
            size: size_bytes as u64,
            atime: 0,
            mtime: 0,
            ctime: 0,
            parent_ino: ROOT_INO,
        };
        fs.put_inode(ino, &inode).unwrap();
        (fs, ino)
    }

    #[test]
    fn write_then_read_small() {
        let (mut fs, ino) = setup_fs_with_file(0);
        do_write(&mut fs, ino, 0, b"hello").unwrap();
        let mut i = fs.get_inode(ino).unwrap().unwrap();
        i.size = 5;
        fs.put_inode(ino, &i).unwrap();
        let got = do_read(&fs, ino, 0, 100).unwrap();
        assert_eq!(got, b"hello");
    }

    #[test]
    fn read_zero_fills_past_inode_size() {
        let (fs, ino) = setup_fs_with_file(5);
        let got = do_read(&fs, ino, 0, 100).unwrap();
        assert_eq!(got, vec![0u8; 5]);
    }

    #[test]
    fn write_crosses_block_boundary() {
        let (mut fs, ino) = setup_fs_with_file(0);
        let data = b"ABCDEFGH";
        do_write(&mut fs, ino, 4094, data).unwrap();
        let mut i = fs.get_inode(ino).unwrap().unwrap();
        i.size = 4094 + 8;
        fs.put_inode(ino, &i).unwrap();

        let extents = fs.list_extents(ino).unwrap();
        assert_eq!(extents.len(), 2);
        assert_eq!(extents[0].0, 0);
        assert_eq!(extents[1].0, BLOCK_SIZE);

        let got = do_read(&fs, ino, 4094, 8).unwrap();
        assert_eq!(got, data);
    }

    #[test]
    fn write_rmw_preserves_surrounding_bytes() {
        let (mut fs, ino) = setup_fs_with_file(0);
        do_write(&mut fs, ino, 0, &[b'A'; 10]).unwrap();
        do_write(&mut fs, ino, 4, b"XX").unwrap();
        let mut i = fs.get_inode(ino).unwrap().unwrap();
        i.size = 10;
        fs.put_inode(ino, &i).unwrap();

        let got = do_read(&fs, ino, 0, 10).unwrap();
        assert_eq!(got, b"AAAAXXAAAA");
    }

    #[test]
    fn read_skips_extents_outside_range() {
        let (mut fs, ino) = setup_fs_with_file(0);
        do_write(&mut fs, ino, 0, b"first").unwrap();
        do_write(&mut fs, ino, BLOCK_SIZE, b"second").unwrap();
        let mut i = fs.get_inode(ino).unwrap().unwrap();
        i.size = BLOCK_SIZE + 6;
        fs.put_inode(ino, &i).unwrap();

        let got = do_read(&fs, ino, BLOCK_SIZE, 6).unwrap();
        assert_eq!(got, b"second");
    }

    // Regression guard for the scan window's lower bound. The extent covering a
    // read is keyed on its block boundary, so a read starting *inside* a
    // non-first block (offset > key) must align the scan start down to that
    // boundary. If do_read passed the raw offset as the lower bound instead of
    // `offset & !(BLOCK_SIZE-1)`, extents_in_range would exclude the covering
    // extent (its key < offset) and the read would wrongly return zeros.
    #[test]
    fn read_from_mid_high_block_includes_covering_extent() {
        let (mut fs, ino) = setup_fs_with_file(0);
        do_write(&mut fs, ino, BLOCK_SIZE, b"ABCDEFGH").unwrap();
        let mut i = fs.get_inode(ino).unwrap().unwrap();
        i.size = BLOCK_SIZE + 8;
        fs.put_inode(ino, &i).unwrap();

        // Start two bytes into block 1: offset=BLOCK_SIZE+2, whose covering
        // extent is keyed at BLOCK_SIZE (< offset).
        let got = do_read(&fs, ino, BLOCK_SIZE + 2, 6).unwrap();
        assert_eq!(got, b"CDEFGH");
    }

    #[test]
    fn validate_name_rejects_bad_names() {
        assert!(validate_name(OsStr::new("")).is_err());
        assert!(validate_name(OsStr::new(".")).is_err());
        assert!(validate_name(OsStr::new("..")).is_err());
        assert!(validate_name(OsStr::new("a/b")).is_err());
        let long = "x".repeat(MAX_NAME_LEN + 1);
        assert!(validate_name(OsStr::new(&long)).is_err());
    }

    #[test]
    fn validate_name_accepts_normal_name() {
        assert!(validate_name(OsStr::new("hello.txt")).is_ok());
    }

    #[test]
    fn to_attr_maps_symlink_kind() {
        let mut i = fresh_inode(FILE_KIND_REGULAR);
        i.mode = S_IFLNK as u32 | 0o777;
        assert_eq!(to_attr(3, &i).kind, FileType::Symlink);
    }

    #[test]
    fn truncate_extents_removes_blocks_past_boundary() {
        let mut fs = Fs::new();
        let ino = fs.alloc_ino();
        let inode = InodeV1 {
            mode: S_IFREG as u32 | 0o644,
            uid: 0,
            gid: 0,
            nlink: 1,
            size: BLOCK_SIZE * 3,
            atime: 0,
            mtime: 0,
            ctime: 0,
            parent_ino: ROOT_INO,
        };
        fs.put_inode(ino, &inode).unwrap();
        for i in 0..3u64 {
            fs.put_extent(ino, i * BLOCK_SIZE, b"x").unwrap();
        }
        truncate_extents(&mut fs, ino, BLOCK_SIZE).unwrap();
        let remaining = fs.list_extents(ino).unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].0, 0);
    }
}
