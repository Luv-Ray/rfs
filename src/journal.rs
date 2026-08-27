use std::sync::Arc;

use zerocopy::{FromZeros, IntoBytes};

use crate::block_btree::BLOCK_SIZE;
use crate::btree::Result;
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
    device: Arc<dyn BlockDevice>,
}

impl Journal {
    pub fn new(device: Arc<dyn BlockDevice>) -> Self {
        Journal { device }
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
        self.device.write_at(frame.as_bytes(), offset)?;
        Ok(())
    }

    /// Read the frame at `seq`, returning it only if it is valid for that seq.
    pub fn read_frame(&self, seq: u64) -> Result<Option<JournalFrame>> {
        let offset = Self::frame_offset(seq);
        // Read straight into a `JournalFrame` value. As a `#[repr(C)]` type with
        // u64 fields it is 8-byte aligned by construction, so its own byte slice
        // is a valid, correctly-aligned read target — no misaligned-cast UB and
        // no intermediate `[u8; N]` buffer + `read_from_bytes` copy. `read_exact_at`
        // fills exactly `size_of::<JournalFrame>()` bytes or errors, so there is
        // no size-mismatch case to handle here.
        let mut frame = JournalFrame::new_zeroed();
        self.device.read_at(frame.as_mut_bytes(), offset)?;
        Ok(frame.is_valid(seq).then_some(frame))
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
    fn make_journal() -> Journal {
        // Back the journal with an in-RAM device — the ring is just a byte
        // range, so no on-disk file is needed to exercise append/scan.
        let device = Arc::new(MemDevice::new());
        device
            .set_len((FIRST_JOURNAL_BLOCK + JOURNAL_BLOCKS) * BLOCK_SIZE as u64)
            .unwrap();
        Journal::new(device)
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
        let journal = make_journal();
        journal.append(&end_frame(1, 65)).unwrap();
        let read = journal.read_frame(1).unwrap().unwrap();
        assert_eq!(read.root_block, 65);
        assert_eq!(read.kind(), Some(FrameKind::CommitEnd));
    }

    #[test]
    fn scan_groups_collects_ops_then_end() {
        let journal = make_journal();
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
        let journal = make_journal();
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
        let journal = make_journal();
        assert!(journal.scan_groups(1).unwrap().is_empty());
        assert_eq!(journal.next_seq_after_scan(1).unwrap(), 1);
    }

    #[test]
    fn ring_wrap() {
        let journal = make_journal();
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
