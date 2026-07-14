use std::os::unix::fs::FileExt;

use zerocopy::{FromBytes, IntoBytes};

use crate::block_btree::BLOCK_SIZE;
use crate::btree::{Error, Result};
use crate::storage::*;

/// One recovered commit group: the logged-op frames of the group followed by
/// its closing `CommitEnd` frame (which carries the resulting fs state).
pub struct CommitGroup {
    /// `(op_kind, op_payload)` for each LoggedOp frame, in order.
    pub ops: Vec<(u8, Vec<u8>)>,
    /// The group's CommitEnd frame (state scalars live here).
    pub end: JournalFrame,
}

pub struct Journal {
    file: std::fs::File,
}

impl Journal {
    pub fn new(file: std::fs::File) -> Self {
        Journal { file }
    }

    /// Byte offset of the frame slot for a given seq (fixed-size ring).
    pub fn frame_offset(seq: u64) -> u64 {
        let ring_pos = seq % JOURNAL_CAPACITY;
        let block_index = ring_pos / ENTRIES_PER_BLOCK as u64 + FIRST_JOURNAL_BLOCK;
        let slot_index = ring_pos % ENTRIES_PER_BLOCK as u64;
        block_index * BLOCK_SIZE as u64 + slot_index * JOURNAL_FRAME_SIZE as u64
    }

    /// Append a single frame at its seq's ring slot. Caller fsyncs.
    pub fn append(&self, frame: &JournalFrame) -> Result<()> {
        let offset = Self::frame_offset(frame.seq);
        self.file.write_all_at(frame.as_bytes(), offset)?;
        Ok(())
    }

    /// Read the frame at `seq`, returning it only if it is valid for that seq.
    pub fn read_frame(&self, seq: u64) -> Result<Option<JournalFrame>> {
        let offset = Self::frame_offset(seq);
        let mut buf = [0u8; JOURNAL_FRAME_SIZE];
        self.file.read_exact_at(&mut buf, offset)?;
        // Copy out with `read_from_bytes` — a `[u8; N]` stack buffer is only
        // 1-byte aligned, but JournalFrame has u64 fields needing 8-byte
        // alignment, so a borrowing `ref_from_bytes` would be UB on misaligned
        // reads (Miri flags it). The owned copy sidesteps the requirement.
        let frame = JournalFrame::read_from_bytes(&buf)
            .map_err(|_| Error::Io(std::io::Error::other("journal frame size mismatch")))?;
        if frame.is_valid(seq) {
            Ok(Some(frame))
        } else {
            Ok(None)
        }
    }

    /// Scan forward from `start_seq`, returning every *complete* commit group
    /// (ending in a `CommitEnd`) in order. A trailing run of `LoggedOp` frames
    /// with no closing `CommitEnd` — a crash mid-commit — is discarded. Scan
    /// stops at the first invalid/missing frame or after a full ring's worth
    /// of frames (wraparound bound).
    pub fn scan_groups(&self, start_seq: u64) -> Result<Vec<CommitGroup>> {
        let mut groups = Vec::new();
        let mut pending: Vec<(u8, Vec<u8>)> = Vec::new();
        let mut seq = start_seq;
        loop {
            if seq - start_seq >= JOURNAL_CAPACITY {
                break;
            }
            let Some(frame) = self.read_frame(seq)? else {
                break;
            };
            match frame.kind() {
                Some(FrameKind::LoggedOp) => {
                    pending.push((frame.op_kind, frame.op_payload().to_vec()));
                }
                Some(FrameKind::CommitEnd) => {
                    groups.push(CommitGroup {
                        ops: std::mem::take(&mut pending),
                        end: frame,
                    });
                }
                None => break,
            }
            seq += 1;
        }
        // `pending` left non-empty = a torn commit at the tail; discard it.
        Ok(groups)
    }

    /// The seq one past the last frame of the last complete commit group,
    /// i.e. where the next append should go. Returns `start_seq` if no
    /// complete group was found.
    pub fn next_seq_after_scan(&self, start_seq: u64) -> Result<u64> {
        let mut seq = start_seq;
        let mut last_group_end = start_seq;
        loop {
            if seq - start_seq >= JOURNAL_CAPACITY {
                break;
            }
            let Some(frame) = self.read_frame(seq)? else {
                break;
            };
            if frame.kind().is_none() {
                break;
            }
            seq += 1;
            if frame.kind() == Some(FrameKind::CommitEnd) {
                last_group_end = seq;
            }
        }
        Ok(last_group_end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn make_journal() -> (NamedTempFile, Journal) {
        let tmp = NamedTempFile::new().unwrap();
        tmp.as_file()
            .set_len((FIRST_JOURNAL_BLOCK + JOURNAL_BLOCKS) * BLOCK_SIZE as u64)
            .unwrap();
        let journal = Journal::new(tmp.as_file().try_clone().unwrap());
        (tmp, journal)
    }

    fn end_frame(seq: u64, root_block: u64) -> JournalFrame {
        let mut f = JournalFrame::commit_end(seq, root_block, 100, 1, 2, u32::MAX - 1, 1, 1);
        f.checksum = f.compute_checksum();
        f
    }

    fn op_frame(seq: u64, op_kind: u8, data: &[u8]) -> JournalFrame {
        let mut f = JournalFrame::logged_op(seq, op_kind, data);
        f.checksum = f.compute_checksum();
        f
    }

    #[test]
    fn append_and_read_back() {
        let (_tmp, journal) = make_journal();
        journal.append(&end_frame(1, 65)).unwrap();
        let read = journal.read_frame(1).unwrap().unwrap();
        assert_eq!(read.root_block, 65);
        assert_eq!(read.kind(), Some(FrameKind::CommitEnd));
    }

    #[test]
    fn scan_groups_collects_ops_then_end() {
        let (_tmp, journal) = make_journal();
        // Group 1: two ops + end.
        journal.append(&op_frame(1, 3, b"op-a")).unwrap();
        journal.append(&op_frame(2, 3, b"op-b")).unwrap();
        journal.append(&end_frame(3, 65)).unwrap();
        // Group 2: one op + end.
        journal.append(&op_frame(4, 5, b"op-c")).unwrap();
        journal.append(&end_frame(5, 70)).unwrap();

        let groups = journal.scan_groups(1).unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].ops.len(), 2);
        assert_eq!(groups[0].ops[0], (3u8, b"op-a".to_vec()));
        assert_eq!(groups[0].end.root_block, 65);
        assert_eq!(groups[1].ops.len(), 1);
        assert_eq!(groups[1].end.root_block, 70);
        assert_eq!(journal.next_seq_after_scan(1).unwrap(), 6);
    }

    #[test]
    fn scan_groups_discards_torn_tail() {
        let (_tmp, journal) = make_journal();
        // One complete group, then a dangling op with no CommitEnd (crash).
        journal.append(&op_frame(1, 3, b"op-a")).unwrap();
        journal.append(&end_frame(2, 65)).unwrap();
        journal.append(&op_frame(3, 3, b"orphan")).unwrap();

        let groups = journal.scan_groups(1).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].end.root_block, 65);
        // next append resumes after the last *complete* group.
        assert_eq!(journal.next_seq_after_scan(1).unwrap(), 3);
    }

    #[test]
    fn scan_returns_empty_on_empty() {
        let (_tmp, journal) = make_journal();
        assert!(journal.scan_groups(1).unwrap().is_empty());
        assert_eq!(journal.next_seq_after_scan(1).unwrap(), 1);
    }

    #[test]
    fn ring_wrap() {
        let (_tmp, journal) = make_journal();
        // Groups that wrap the ring boundary.
        let start = JOURNAL_CAPACITY - 2;
        journal.append(&op_frame(start, 3, b"x")).unwrap();
        journal.append(&end_frame(start + 1, 101)).unwrap();
        journal.append(&op_frame(start + 2, 3, b"y")).unwrap();
        journal.append(&end_frame(start + 3, 102)).unwrap();

        let groups = journal.scan_groups(start).unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[1].end.root_block, 102);
    }
}
