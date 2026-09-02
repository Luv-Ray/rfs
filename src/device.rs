//! Block-device abstraction: the single IO layer under `BlockStore` and
//! `Journal`.

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::sync::RwLock;

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
///
/// Backing assumption — **dense, contiguous block numbers**. The `Vec` indexes
/// bytes at `block_nr * BLOCK_SIZE`, so memory grows to the *highest* offset
/// ever written, and any gap below it is materialized as real zero bytes. A
/// real image file gets sparse holes for free from the OS (an unwritten range
/// costs no physical blocks); this `Vec` does not model that. It is fine today
/// because the allocator hands out block numbers monotonically from
/// `FIRST_DATA_BLOCK_NR` with free-list reuse, so the space stays compact with
/// no large holes.
///
/// If a future allocator introduces sparse/structured block numbers — e.g.
/// bcachefs-style per-device partitioning or zoning, where addresses become
/// `(device, bucket, offset)` rather than one compact `u64` — this `Vec`
/// backing would balloon to fill the holes. At that point swap it for a
/// per-block sparse map (`HashMap<u64, Box<[u8; BLOCK_SIZE]>>`: a missing key =
/// a hole = reads-as-zero), which matches an extent tree's "no entry = hole"
/// semantics. (Identity-addressed layouts like `block = f(ino)` are *not* on
/// this path: in the bcachefs style the btree already provides the
/// identity→location indirection.)
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
