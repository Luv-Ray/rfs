use std::os::unix::fs::FileExt;

use zerocopy::{FromBytes, IntoBytes};

use crate::block_btree::BLOCK_SIZE;
use crate::btree::{Error, Result};
use crate::storage::*;

pub struct Journal {
    file: std::fs::File,
}

impl Journal {
    pub fn new(file: std::fs::File) -> Self {
        Journal { file }
    }

    pub fn entry_offset(seq: u64) -> u64 {
        let ring_pos = seq % JOURNAL_CAPACITY;
        let block_index = ring_pos / ENTRIES_PER_BLOCK as u64 + FIRST_JOURNAL_BLOCK;
        let slot_index = ring_pos % ENTRIES_PER_BLOCK as u64;
        block_index * BLOCK_SIZE as u64 + slot_index * std::mem::size_of::<JournalEntry>() as u64
    }

    pub fn append(&self, entry: &JournalEntry) -> Result<()> {
        let offset = Self::entry_offset(entry.seq);
        self.file.write_all_at(entry.as_bytes(), offset)?;
        Ok(())
    }

    pub fn read_entry(&self, seq: u64) -> Result<Option<JournalEntry>> {
        let offset = Self::entry_offset(seq);
        let mut buf = [0u8; std::mem::size_of::<JournalEntry>()];
        self.file.read_exact_at(&mut buf, offset)?;
        let entry = JournalEntry::ref_from_bytes(&buf)
            .map_err(|_| Error::Io(std::io::Error::other("journal entry size mismatch")))?;
        if entry.is_valid(seq) {
            Ok(Some(*entry))
        } else {
            Ok(None)
        }
    }

    pub fn scan_from(&self, start_seq: u64) -> Result<Option<JournalEntry>> {
        let mut last_valid: Option<JournalEntry> = None;
        let mut seq = start_seq;
        loop {
            if seq - start_seq >= JOURNAL_CAPACITY {
                break;
            }
            match self.read_entry(seq)? {
                Some(e) => {
                    last_valid = Some(e);
                    seq += 1;
                }
                None => break,
            }
        }
        Ok(last_valid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn make_entry(seq: u64, root_block: u64) -> JournalEntry {
        let mut e = JournalEntry {
            magic: JOURNAL_MAGIC,
            checksum: 0,
            seq,
            root_block,
            next_block_nr: 100,
            next_bset_seq: 1,
            next_ino: 2,
            next_snap_id: u32::MAX - 1,
            next_subvol_id: 1,
            current_subvol: 1,
            _reserved: [0; 68],
        };
        e.checksum = e.compute_checksum();
        e
    }

    #[test]
    fn append_and_read_back() {
        let tmp = NamedTempFile::new().unwrap();
        // Extend file to cover journal region
        tmp.as_file()
            .set_len((FIRST_JOURNAL_BLOCK + JOURNAL_BLOCKS) * BLOCK_SIZE as u64)
            .unwrap();
        let journal = Journal::new(tmp.as_file().try_clone().unwrap());

        let e1 = make_entry(1, 65);
        journal.append(&e1).unwrap();

        let read = journal.read_entry(1).unwrap();
        assert_eq!(read.unwrap().root_block, 65);
    }

    #[test]
    fn scan_finds_last_valid() {
        let tmp = NamedTempFile::new().unwrap();
        tmp.as_file()
            .set_len((FIRST_JOURNAL_BLOCK + JOURNAL_BLOCKS) * BLOCK_SIZE as u64)
            .unwrap();
        let journal = Journal::new(tmp.as_file().try_clone().unwrap());

        for seq in 1..=5 {
            journal.append(&make_entry(seq, 60 + seq)).unwrap();
        }
        let last = journal.scan_from(1).unwrap().unwrap();
        assert_eq!(last.seq, 5);
        assert_eq!(last.root_block, 65);
    }

    #[test]
    fn scan_returns_none_on_empty() {
        let tmp = NamedTempFile::new().unwrap();
        tmp.as_file()
            .set_len((FIRST_JOURNAL_BLOCK + JOURNAL_BLOCKS) * BLOCK_SIZE as u64)
            .unwrap();
        let journal = Journal::new(tmp.as_file().try_clone().unwrap());

        assert!(journal.scan_from(1).unwrap().is_none());
    }

    #[test]
    fn ring_wrap() {
        let tmp = NamedTempFile::new().unwrap();
        tmp.as_file()
            .set_len((FIRST_JOURNAL_BLOCK + JOURNAL_BLOCKS) * BLOCK_SIZE as u64)
            .unwrap();
        let journal = Journal::new(tmp.as_file().try_clone().unwrap());

        // Write entries that wrap the ring
        let start = JOURNAL_CAPACITY - 2;
        for i in 0..5u64 {
            journal.append(&make_entry(start + i, 100 + i)).unwrap();
        }
        let last = journal.scan_from(start).unwrap().unwrap();
        assert_eq!(last.seq, start + 4);
    }
}
