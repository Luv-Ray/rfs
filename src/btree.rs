use std::sync::Arc;

use crate::storage::{BlockStore, FIRST_DATA_BLOCK_NR};

use crate::block_btree::{
    BSET_SOFT_LIMIT, BSET_TREE_NR_MAX, BtreeNodeRaw, DiskEntry, EntryKind, MAGIC_NUMBER,
    MAX_ENTRIES, MAX_INTERNAL_KEYS, MAX_KEY_SIZE, MAX_LOGICAL_KEY_SIZE, MAX_VALUE_SIZE, MergedIter,
    ROOT_SNAP, SNAP_ID_BYTES, SnapId, merged_find,
};

/// Build the sortable byte form of a (logical, snap_id) pair on the stack.
/// Returns a 32-byte buffer with the logical key in `[..n]` and snap_id_be
/// in `[n..n+4]`; the live slice is `[..n+4]` where `n = logical.len()`.
fn sortable_key(logical: &[u8], snap: SnapId) -> ([u8; MAX_KEY_SIZE], usize) {
    assert!(
        logical.len() <= MAX_LOGICAL_KEY_SIZE,
        "logical key too long: {} > {MAX_LOGICAL_KEY_SIZE}",
        logical.len()
    );
    let mut buf = [0u8; MAX_KEY_SIZE];
    let n = logical.len();
    buf[..n].copy_from_slice(logical);
    buf[n..n + SNAP_ID_BYTES].copy_from_slice(&snap.to_be_bytes());
    (buf, n + SNAP_ID_BYTES)
}

/// One row of `Btree::range_scan_all`: full entry view with snap_id and kind.
/// Aliased for readability (clippy nags at the inline form).
pub type AllSnapRow = (Vec<u8>, SnapId, EntryKind, Vec<u8>);

#[derive(Debug)]
pub enum Error {
    /// A block referenced by the tree is missing from the block map.
    /// In memory-only mode this means the tree pointed at an unallocated
    /// block; with an image backend, it means the block isn't cached
    /// AND there's no backing file (i.e. the image was opened in memory-
    /// -only mode by mistake).
    BlockNotFound(u64),
    /// I/O error against the backing image file (read, write, fsync,
    /// open, ftruncate).
    Io(std::io::Error),
    /// A persisted block header had the wrong magic number. Most likely
    /// causes: image truncated, image overwritten by another tool, or
    /// the allocator handed out the same block_nr twice.
    BadMagic { block: u64, got: u32, expected: u32 },
    /// Magic was correct but the per-block CRC didn't match. The block is
    /// torn / corrupted.
    ChecksumMismatch { block: u64 },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::BlockNotFound(nr) => write!(f, "block {nr} not found"),
            Error::Io(e) => write!(f, "i/o error: {e}"),
            Error::BadMagic {
                block,
                got,
                expected,
            } => write!(
                f,
                "block {block}: bad magic {got:#010x} (expected {expected:#010x})"
            ),
            Error::ChecksumMismatch { block } => {
                write!(f, "block {block}: checksum mismatch")
            }
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// COW B-tree: all mutations produce new nodes; old nodes remain reachable
/// via their block numbers (key for crash recovery and snapshots).
/// A resolved key/value write, captured for the write-ahead log. This is the
/// *outcome* of a btree op (after Deleted-vs-Whiteout resolution), so replay
/// is a single mode: re-insert `(sortable_key, kind, value)` — no snapshot
/// chain, idempotent. `sortable_key` already includes the 4-byte snap suffix.
#[derive(Clone, Debug)]
pub struct LogRecord {
    pub sortable_key: Vec<u8>,
    pub kind: EntryKind,
    pub value: Vec<u8>,
}

pub struct Btree {
    pub root_block: u64,
    /// Shared block storage. May be in-memory (cache only) or image-backed.
    /// Wrapped in `Rc` because `Fs` shares the same store for data blocks.
    pub store: Arc<BlockStore>,
    /// Monotonically increasing seq assigned to each newly opened bset.
    /// Used by the cross-bset merged iterator to break ties when the same
    /// sortable key appears in multiple bsets (latest write wins).
    next_bset_seq: u64,
    /// Resolved writes since the last drain, for the WAL. Drained by
    /// `Fs::journal_commit`. Recording is suppressed during replay via
    /// `logging`.
    log: Vec<LogRecord>,
    /// Whether to record resolved writes into `log`. False during replay so
    /// re-applied ops are not re-logged.
    logging: bool,
}

impl Btree {
    /// Create an in-memory tree. The root block is allocated first from the
    /// store (which starts at `FIRST_DATA_BLOCK_NR`) and seeded with an empty
    /// bset.
    pub fn new() -> Self {
        let store = Arc::new(BlockStore::in_memory());
        Self::initialize_with(store, FIRST_DATA_BLOCK_NR)
    }

    /// Create a tree backed by the given store, allocating a fresh root.
    /// Use this when seeding a brand-new image: caller has already created
    /// the `BlockStore`, and we'll allocate the very first non-superblock/
    /// non-journal block to hold the empty root.
    pub fn create_in(store: Arc<BlockStore>) -> Self {
        Self::initialize_with(store, FIRST_DATA_BLOCK_NR)
    }

    /// Reattach to an existing tree whose root and seq counter are stored
    /// in a superblock. Called by `Fs::open` after parsing the superblock.
    pub fn reopen(store: Arc<BlockStore>, root_block: u64, next_bset_seq: u64) -> Self {
        Btree {
            root_block,
            store,
            next_bset_seq,
            log: Vec::new(),
            logging: true,
        }
    }

    fn initialize_with(store: Arc<BlockStore>, expected_root: u64) -> Self {
        let root_block = store.alloc();
        debug_assert_eq!(
            root_block, expected_root,
            "fresh tree should get block {expected_root} as its root"
        );
        let mut root = BtreeNodeRaw::new(0);
        root.start_new_bset(0);
        store
            .write_node(root_block, &root)
            .expect("write root into fresh store");
        Btree {
            root_block,
            store,
            // Seq 0 was consumed by the root's initial bset. Subsequent bsets
            // (from inserts that open a new bset, or from compaction) start
            // at 1.
            next_bset_seq: 1,
            log: Vec::new(),
            logging: true,
        }
    }

    /// Take the accumulated log records, clearing the buffer.
    pub fn drain_log(&mut self) -> Vec<LogRecord> {
        std::mem::take(&mut self.log)
    }

    /// Flush all in-place (dirty) edits made since the last checkpoint onto
    /// fresh blocks and update `root_block` to the relocated root. Every write
    /// targets a freshly-allocated block, so the previous on-disk checkpoint is
    /// left intact until the caller publishes the new root (superblock swap).
    ///
    /// Called by `Fs::sync`. Idempotent when nothing is dirty: with no dirty
    /// nodes the walk keeps every block_nr and `root_block` is unchanged.
    pub fn checkpoint(&mut self) -> Result<()> {
        let new_root = checkpoint_flush(&self.store, self.root_block)?;
        self.root_block = new_root;
        Ok(())
    }

    /// Current number of buffered log records (for transaction rollback).
    fn log_len(&self) -> usize {
        self.log.len()
    }

    /// Truncate the log back to `len` (undo records from an aborted tx).
    fn log_truncate(&mut self, len: usize) {
        self.log.truncate(len);
    }

    /// Run `f` with logging suppressed (used during replay so re-applied ops
    /// aren't re-recorded), restoring the prior setting afterward.
    fn without_logging<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        let prev = self.logging;
        self.logging = false;
        let r = f(self);
        self.logging = prev;
        r
    }

    /// Read-only access to the next bset seq counter (for sync into
    /// the superblock).
    pub fn next_bset_seq(&self) -> u64 {
        self.next_bset_seq
    }

    /// Find a key at the root snapshot. For snap-aware lookups use [`find_at`].
    pub fn find(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.find_at(key, ROOT_SNAP)
    }

    /// Find the value of `(key, snap)`. For Phase 2 this is exact-match only;
    /// the ancestor walk that makes parent-snapshot keys visible at a child
    /// snapshot lands in Phase 4.
    pub fn find_at(&self, key: &[u8], snap: SnapId) -> Result<Option<Vec<u8>>> {
        let (buf, len) = sortable_key(key, snap);
        find(&self.store, self.root_block, &buf[..len])
    }

    pub fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        self.insert_at(key, ROOT_SNAP, value)
    }

    pub fn insert_at(&mut self, key: &[u8], snap: SnapId, value: &[u8]) -> Result<()> {
        self.insert_with_kind(key, snap, EntryKind::Live, value)
    }

    /// Insert (or overwrite) an entry at `(key, snap)` with the given kind.
    /// Tombstones (Deleted/Whiteout) typically pass an empty `value`.
    pub fn insert_with_kind(
        &mut self,
        key: &[u8],
        snap: SnapId,
        kind: EntryKind,
        value: &[u8],
    ) -> Result<()> {
        let (buf, len) = sortable_key(key, snap);
        let new_root = insert(
            &self.store,
            &mut self.next_bset_seq,
            self.root_block,
            &buf[..len],
            value,
            kind,
            WriteMode::InPlace,
        )?;
        self.root_block = new_root;
        if self.logging {
            self.log.push(LogRecord {
                sortable_key: buf[..len].to_vec(),
                kind,
                value: value.to_vec(),
            });
        }
        Ok(())
    }

    /// Delete `key` at the root snapshot (no ancestor walk). Convenience for
    /// tests; production paths should use [`delete_at`].
    pub fn delete(&mut self, key: &[u8]) -> Result<bool> {
        self.delete_at(key, ROOT_SNAP, &[ROOT_SNAP])
    }

    /// Delete `key` at `snap`. The snapshot ancestor chain is required to
    /// determine whether the visible entry comes from `snap` itself
    /// (→ KEY_TYPE_deleted, trivial tombstone) or from an ancestor
    /// (→ KEY_TYPE_whiteout, snapshot tombstone shadowing the ancestor).
    ///
    /// Returns `true` if a tombstone was written (a visible Live entry was
    /// found at `snap`), `false` if `key` was already invisible (no-op).
    pub fn delete_at(&mut self, key: &[u8], snap: SnapId, chain: &[SnapId]) -> Result<bool> {
        let Some((_value, visible_snap)) = self.find_visible_with_snap(key, chain)? else {
            return Ok(false);
        };
        if visible_snap == snap {
            // Same-snap delete: the live entry sits at sortable_key(key, snap)
            // in our own tree. Flip its kind to Deleted in place — nkeys is
            // unchanged, no new bset is opened, no split can be triggered.
            let (buf, len) = sortable_key(key, snap);
            let new_root = flip_kind_in_place(
                &self.store,
                self.root_block,
                &buf[..len],
                EntryKind::Deleted,
                WriteMode::InPlace,
            )?;
            self.root_block = new_root;
            if self.logging {
                self.log.push(LogRecord {
                    sortable_key: buf[..len].to_vec(),
                    kind: EntryKind::Deleted,
                    value: Vec::new(),
                });
            }
        } else {
            // Inherited from an ancestor. Must shadow the ancestor's
            // still-live entry with a Whiteout written at our snap.
            // insert_with_kind records the log entry itself.
            self.insert_with_kind(key, snap, EntryKind::Whiteout, &[])?;
        }
        Ok(true)
    }

    /// Like `find_visible` but also returns the snap_id at which the visible
    /// entry was stored. Used by `delete_at` to pick Deleted vs Whiteout.
    pub fn find_visible_with_snap(
        &self,
        logical: &[u8],
        chain: &[SnapId],
    ) -> Result<Option<(Vec<u8>, SnapId)>> {
        find_visible_with_snap(&self.store, self.root_block, logical, chain)
    }

    pub fn range_scan(&self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.range_scan_at(start, end, ROOT_SNAP)
    }

    /// Range-scan over `[start, end)` at the given snap_id.
    ///
    /// `snap` is currently unused at the storage level — Phase 4 will use it
    /// to filter visible-vs-shadowed entries via the ancestor chain. For now
    /// we expand the logical range to include any snap_id (`snap_id` byte
    /// suffix `[0;4]` on both ends), which is the same as the old behavior
    /// when every entry sits at `ROOT_SNAP`.
    ///
    /// Returned keys are *logical only* (snap_id stripped).
    pub fn range_scan_at(
        &self,
        start: &[u8],
        end: &[u8],
        _snap: SnapId,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        // Build sortable bounds with snap_id bytes set to 0. For any logical
        // key K and any snap S: `K_logical_bytes ++ S_be >= K_logical_bytes ++ [0;4]`,
        // so this captures all snap_ids of every logical key in [start, end).
        let mut start_buf = [0u8; MAX_KEY_SIZE];
        let n_start = start.len();
        assert!(n_start <= MAX_LOGICAL_KEY_SIZE);
        start_buf[..n_start].copy_from_slice(start);
        let mut end_buf = [0u8; MAX_KEY_SIZE];
        let n_end = end.len();
        assert!(n_end <= MAX_LOGICAL_KEY_SIZE);
        end_buf[..n_end].copy_from_slice(end);

        let mut results = Vec::new();
        range_scan(
            &self.store,
            self.root_block,
            &start_buf[..n_start + SNAP_ID_BYTES],
            &end_buf[..n_end + SNAP_ID_BYTES],
            &mut results,
        )?;
        Ok(results)
    }

    /// Find the value visible at `target_snap` by walking its ancestor chain.
    ///
    /// `chain` must be the snapshot ancestor chain in ASC-by-specificity order:
    /// `[target_snap, parent(target_snap), grandparent, ..., tree_root]`. The
    /// caller is responsible for building it from the snapshot tree (lives in
    /// `Fs`, not `Btree`) — this method just iterates.
    ///
    /// Returns the value of the first ancestor with a `Live` entry. If the
    /// closest ancestor with an entry has a tombstone (Deleted/Whiteout), the
    /// chain walk stops and `None` is returned (the tombstone shadows the
    /// rest of the chain).
    pub fn find_visible(&self, logical: &[u8], chain: &[SnapId]) -> Result<Option<Vec<u8>>> {
        find_visible(&self.store, self.root_block, logical, chain)
    }

    /// Raw range scan: returns every entry in `[start, end)` regardless of
    /// snap_id or kind, including tombstones. Used by `range_scan_visible`
    /// and by GC / debug dumps. The bcachefs equivalent is the
    /// `BTREE_ITER_ALL_SNAPSHOTS` iterator mode.
    pub fn range_scan_all(&self, start: &[u8], end: &[u8]) -> Result<Vec<AllSnapRow>> {
        let mut start_buf = [0u8; MAX_KEY_SIZE];
        let n_start = start.len();
        assert!(n_start <= MAX_LOGICAL_KEY_SIZE);
        start_buf[..n_start].copy_from_slice(start);
        let mut end_buf = [0u8; MAX_KEY_SIZE];
        let n_end = end.len();
        assert!(n_end <= MAX_LOGICAL_KEY_SIZE);
        end_buf[..n_end].copy_from_slice(end);

        let mut results = Vec::new();
        range_scan_all(
            &self.store,
            self.root_block,
            &start_buf[..n_start + SNAP_ID_BYTES],
            &end_buf[..n_end + SNAP_ID_BYTES],
            &mut results,
        )?;
        Ok(results)
    }

    /// Range scan with snapshot ancestor filtering applied per logical key.
    ///
    /// For each distinct logical key in `[start, end)`, returns at most one
    /// `(logical_key, value)` pair: the value visible at `target_snap`
    /// according to the ancestor chain. Tombstones shadow rather than emit.
    pub fn range_scan_visible(
        &self,
        start: &[u8],
        end: &[u8],
        chain: &[SnapId],
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        // chain_set lets us answer "is S on the ancestor chain of target?"
        // in O(1) during the linear scan. Chain length is at most snapshot
        // depth (small).
        let chain_set: std::collections::HashSet<SnapId> = chain.iter().copied().collect();
        let raw = self.range_scan_all(start, end)?;
        // raw is in (logical ASC, snap_id ASC) order. For each logical key
        // group, the FIRST entry whose snap_id is on the chain decides
        // visibility (Live → emit, tombstone → skip the whole group).
        let mut out: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        let mut current_logical: Option<Vec<u8>> = None;
        let mut current_decided = false;
        for (logical, snap, kind, value) in raw {
            if current_logical.as_deref() != Some(logical.as_slice()) {
                current_logical = Some(logical.clone());
                current_decided = false;
            }
            if current_decided {
                continue;
            }
            if !chain_set.contains(&snap) {
                continue;
            }
            match kind {
                EntryKind::Live => {
                    out.push((logical, value));
                    current_decided = true;
                }
                EntryKind::Deleted | EntryKind::Whiteout => {
                    current_decided = true;
                }
            }
        }
        Ok(out)
    }

    pub fn dump(&self) {
        println!(
            "=== B-tree (next_block_nr={}) ===",
            self.store.next_block_nr()
        );
        dump(&self.store, self.root_block, 0);
        println!("=== end ===");
    }

    /// Walk the entire tree and panic if any invariant is violated.
    #[cfg(test)]
    pub fn verify(&self) {
        verify_node(&self.store, self.root_block, None, None);
    }
}

// ---------- Transaction (Phase 6) ----------
//
// A transaction batches multiple inserts/deletes into a single root-swap
// commit. While the transaction is open, mutations produce new blocks but
// the btree's `root_block` is unchanged — readers outside the closure see
// the old state. On commit, the new root takes effect atomically.
//
// On abort (closure returning Err), the btree's `root_block` is unchanged.
// New blocks allocated during the transaction remain in `store` as
// orphans (unreachable, awaiting future GC).
//
// Limitations vs bcachefs `btree_trans`:
// - No conflict detection (we have no concurrent writers).
// - No journal — we only track the root pointer.
// - Errors during commit are not survivable (the closure must succeed
//   end-to-end or be retried).

pub struct Tx<'a> {
    btree: &'a mut Btree,
    /// Root block as seen by operations within this transaction. Starts as
    /// the btree's current root and is updated by each insert/delete.
    pending_root: u64,
}

impl<'a> Tx<'a> {
    pub fn insert(&mut self, key: &[u8], snap: SnapId, value: &[u8]) -> Result<()> {
        self.insert_with_kind(key, snap, EntryKind::Live, value)
    }

    pub fn insert_with_kind(
        &mut self,
        key: &[u8],
        snap: SnapId,
        kind: EntryKind,
        value: &[u8],
    ) -> Result<()> {
        let (buf, len) = sortable_key(key, snap);
        let new_root = insert(
            &self.btree.store,
            &mut self.btree.next_bset_seq,
            self.pending_root,
            &buf[..len],
            value,
            kind,
            WriteMode::Cow,
        )?;
        self.pending_root = new_root;
        if self.btree.logging {
            self.btree.log.push(LogRecord {
                sortable_key: buf[..len].to_vec(),
                kind,
                value: value.to_vec(),
            });
        }
        Ok(())
    }

    pub fn delete_at(&mut self, key: &[u8], snap: SnapId, chain: &[SnapId]) -> Result<bool> {
        let Some((_value, visible_snap)) =
            find_visible_with_snap(&self.btree.store, self.pending_root, key, chain)?
        else {
            return Ok(false);
        };
        if visible_snap == snap {
            let (buf, len) = sortable_key(key, snap);
            let new_root = flip_kind_in_place(
                &self.btree.store,
                self.pending_root,
                &buf[..len],
                EntryKind::Deleted,
                WriteMode::Cow,
            )?;
            self.pending_root = new_root;
            if self.btree.logging {
                self.btree.log.push(LogRecord {
                    sortable_key: buf[..len].to_vec(),
                    kind: EntryKind::Deleted,
                    value: Vec::new(),
                });
            }
        } else {
            self.insert_with_kind(key, snap, EntryKind::Whiteout, &[])?;
        }
        Ok(true)
    }

    pub fn find_at(&self, key: &[u8], snap: SnapId) -> Result<Option<Vec<u8>>> {
        let (buf, len) = sortable_key(key, snap);
        find(&self.btree.store, self.pending_root, &buf[..len])
    }

    pub fn find_visible(&self, key: &[u8], chain: &[SnapId]) -> Result<Option<Vec<u8>>> {
        find_visible(&self.btree.store, self.pending_root, key, chain)
    }

    pub fn find_visible_with_snap(
        &self,
        key: &[u8],
        chain: &[SnapId],
    ) -> Result<Option<(Vec<u8>, SnapId)>> {
        find_visible_with_snap(&self.btree.store, self.pending_root, key, chain)
    }
}

impl Btree {
    /// Run `f` against a fresh transaction. On `Ok(_)` the pending root is
    /// installed as the btree's new root in a single atomic step. On `Err`
    /// the btree is left untouched (orphan blocks are leaked; v1 has no GC).
    pub fn transaction<F, R>(&mut self, f: F) -> Result<R>
    where
        F: FnOnce(&mut Tx<'_>) -> Result<R>,
    {
        let log_mark = self.log_len();
        let mut tx = Tx {
            pending_root: self.root_block,
            btree: self,
        };
        let result = f(&mut tx);
        match result {
            Ok(value) => {
                let new_root = tx.pending_root;
                tx.btree.root_block = new_root;
                Ok(value)
            }
            Err(e) => {
                // Abort: root untouched, and drop any log records the closure
                // pushed before failing (those writes never became visible).
                self.log_truncate(log_mark);
                Err(e)
            }
        }
    }

    /// Re-apply one logged record during recovery. Writes `(sortable_key,
    /// kind, value)` directly with logging suppressed (so replay doesn't
    /// re-log). `sortable_key` already carries the snap suffix, so this splits
    /// it back into logical key + snap for the underlying insert.
    pub fn replay_record(&mut self, rec: &LogRecord) -> Result<()> {
        let n = rec.sortable_key.len();
        debug_assert!(n >= SNAP_ID_BYTES, "sortable key too short");
        let logical = &rec.sortable_key[..n - SNAP_ID_BYTES];
        let snap = SnapId::from_be_bytes(
            rec.sortable_key[n - SNAP_ID_BYTES..]
                .try_into()
                .expect("snap suffix is 4 bytes"),
        );
        self.without_logging(|bt| bt.insert_with_kind(logical, snap, rec.kind, &rec.value))
    }
}

impl Default for Btree {
    fn default() -> Self {
        Btree::new()
    }
}

/// Count total keys in the subtree rooted at `block_nr`.
#[cfg(test)]
fn count_keys(store: &BlockStore, block_nr: u64) -> usize {
    let node = store.read_node_copy(block_nr).unwrap();
    if node.level() == 0 {
        // Distinct keys after merge dedup. Multi-bset leaves can store the
        // same key multiple times (older bsets shadowed by newer ones), so
        // raw nkeys() is not what we want — MergedIter yields each unique
        // key once (the highest-seq entry).
        MergedIter::new(&node).count()
    } else {
        // Internal keys are separators — they duplicate keys that also exist
        // in the subtrees (separator-in-right convention). Only count leaves.
        let mut total = 0;
        for i in 0..=node.nkeys() {
            total += count_keys(store, node.child_block(i));
        }
        total
    }
}

/// Collect the depth of every leaf node. All leaves must be at the same depth
/// for the tree to be balanced.
#[cfg(test)]
fn collect_leaf_depths(store: &BlockStore, block_nr: u64, depth: usize, depths: &mut Vec<usize>) {
    let node = store.read_node_copy(block_nr).unwrap();
    if node.level() == 0 {
        depths.push(depth);
    } else {
        for i in 0..=node.nkeys() {
            collect_leaf_depths(store, node.child_block(i), depth + 1, depths);
        }
    }
}

/// Recursive invariant check. `lo`/`hi` are exclusive bounds inherited from
/// the parent separator (separator-in-right convention: the separator belongs
/// to the right subtree, so child[i] holds keys in [lo, hi)).
#[cfg(test)]
fn verify_node(store: &BlockStore, block_nr: u64, lo: Option<&[u8]>, hi: Option<&[u8]>) {
    let node = store.read_node_copy(block_nr).unwrap();
    let n = node.nkeys();

    if node.level() == 0 {
        debug_assert!(n <= MAX_ENTRIES, "leaf blk={block_nr} nkeys={n}");
    } else {
        debug_assert!(n <= MAX_INTERNAL_KEYS, "internal blk={block_nr} nkeys={n}");
    }

    if node.level() == 0 {
        // Leaves may have multiple bsets. Each bset is sorted internally;
        // across bsets the merged view must also be sorted (this falls out
        // of MergedIter, but we double-check by walking it).
        let mut prev: Option<Vec<u8>> = None;
        for (b, i) in MergedIter::new(&node) {
            let k = node.entry_at(b, i).key_bytes().to_vec();
            if let Some(p) = &prev {
                debug_assert!(
                    p.as_slice() < k.as_slice(),
                    "blk={block_nr} merged-view not strictly sorted: {p:?} >= {k:?}"
                );
            }
            if let Some(lo) = lo {
                debug_assert!(k.as_slice() >= lo, "blk={block_nr} key {k:?} < lo {lo:?}");
            }
            if let Some(hi) = hi {
                debug_assert!(k.as_slice() < hi, "blk={block_nr} key {k:?} >= hi {hi:?}");
            }
            prev = Some(k);
        }
        return;
    }

    // Internal nodes: single-bset, classic separator layout.
    for i in 1..n {
        debug_assert!(
            node.entry(i - 1).key_bytes() < node.entry(i).key_bytes(),
            "blk={block_nr} keys not sorted at idx {}: {:?} >= {:?}",
            i - 1,
            node.entry(i - 1).key_bytes(),
            node.entry(i).key_bytes(),
        );
    }

    for i in 0..n {
        let k = node.entry(i).key_bytes();
        if let Some(lo) = lo
            && k < lo
        {
            panic!(
                "blk={block_nr} level={} nkeys={n} key[{i}]={k:?} < lo={lo:?}\n\
                 first_key={:?} last_key={:?}",
                node.level(),
                node.entry(0).key_bytes(),
                node.entry(n - 1).key_bytes(),
            );
        }
        if let Some(hi) = hi {
            debug_assert!(k < hi, "blk={block_nr} key[{i}]={k:?} >= hi={hi:?}");
        }
    }

    for i in 0..=n {
        let child_nr = node.child_block(i);
        debug_assert!(
            store.read_node_copy(child_nr).is_ok(),
            "blk={block_nr} child[{i}]={child_nr} not in store"
        );
        let child = store.read_node_copy(child_nr).unwrap();
        debug_assert_eq!(
            child.level(),
            node.level() - 1,
            "blk={block_nr} child[{i}] level mismatch"
        );

        // Separator-in-right: child[i] holds keys in [separator(i-1), separator(i)).
        let child_lo = if i == 0 {
            lo
        } else {
            Some(node.entry(i - 1).key_bytes())
        };
        let child_hi = if i == n {
            hi
        } else {
            Some(node.entry(i).key_bytes())
        };
        verify_node(store, child_nr, child_lo, child_hi);
    }
}

/// Heap-allocate a zeroed node with a fresh empty bset 0 (avoids 4KB stack
/// temporaries). The single-bset start makes the node usable by single-bset
/// code paths (split, promote, compact) immediately.
fn new_node_on_heap(level: u8) -> Box<BtreeNodeRaw> {
    // SAFETY: repr(C) + FromBytes — zeroed memory is valid; we set header fields below.
    let mut b: Box<BtreeNodeRaw> = unsafe { Box::<BtreeNodeRaw>::new_zeroed().assume_init() };
    b.header.magic = MAGIC_NUMBER;
    b.header.level = level;
    b.start_new_bset(0);
    b
}

/// COW clone: heap-copy without touching the stack.
fn clone_to_heap(node: &BtreeNodeRaw) -> Box<BtreeNodeRaw> {
    // SAFETY: repr(C) struct — memcpy on the heap is equivalent to Clone
    // but avoids the 4KB stack temporary.
    let mut b: Box<BtreeNodeRaw> = unsafe { Box::<BtreeNodeRaw>::new_zeroed().assume_init() };
    unsafe {
        std::ptr::copy_nonoverlapping(node as *const BtreeNodeRaw, &mut *b as *mut BtreeNodeRaw, 1);
    }
    b
}

// ---------- Multi-bset support helpers ----------

/// Counts how many times `ensure_writable_last_bset` chose the
/// 4-bsets-full → compact branch. Used by `compaction_path_is_reachable`
/// to guard against accidentally making that path unreachable via constant
/// changes (a regression that would silently disable the multi-bset
/// optimization on full nodes).
#[cfg(test)]
static COMPACT_ON_FULL_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Allocate the next monotonic bset seq.
fn alloc_seq(next_bset_seq: &mut u64) -> u64 {
    let s = *next_bset_seq;
    *next_bset_seq += 1;
    s
}

/// Build a merged-sorted snapshot of every entry in a leaf node, walking
/// across all of its bsets and collapsing duplicate keys (highest-seq wins).
/// Used by compaction. Returns owned `DiskEntry` values so the source node
/// can be modified afterwards without aliasing.
fn collect_leaf_entries_sorted(node: &BtreeNodeRaw) -> Vec<DiskEntry> {
    debug_assert_eq!(node.level(), 0);
    MergedIter::new(node)
        .map(|(b, i)| *node.entry_at(b, i))
        .collect()
}

/// Compact every bset of a leaf node into a single fresh bset. The merged
/// view (deduplicated by `MergedIter`) is the new content; older shadowed
/// entries are dropped. The new bset is tagged with `new_seq`.
///
/// Always operates on leaves — internals stay single-bset by construction
/// and never need compaction.
fn compact_leaf_in_place(node: &mut BtreeNodeRaw, new_seq: u64) {
    debug_assert_eq!(node.level(), 0);
    let entries = collect_leaf_entries_sorted(node);
    let gen_no = node.generation();
    // Reset the body to a single empty bset; replay entries in sorted order.
    *node = BtreeNodeRaw::new(0);
    node.set_generation(gen_no);
    node.start_new_bset(new_seq);
    for entry in &entries {
        node.append_to_last_bset(entry);
    }
}

/// Make the latest bset of `leaf` ready to receive a new sort-insert.
///
/// Decision tree:
/// 1. If there's no bset yet → open one.
/// 2. If the latest bset still has room (< BSET_SOFT_LIMIT entries) →
///    sort-insert directly.
/// 3. Otherwise, if `bset_count < BSET_TREE_NR_MAX` → open a new bset.
/// 4. Otherwise (all four bsets present and the last one is full) → compact
///    everything into a fresh single bset.
///
/// After this, the caller's sort_insert_into_last_bset is guaranteed to
/// have at least one slot of room (assuming total nkeys < MAX_ENTRIES).
fn ensure_writable_last_bset(leaf: &mut BtreeNodeRaw, next_bset_seq: &mut u64) {
    debug_assert_eq!(leaf.level(), 0);
    if leaf.bset_count() == 0 {
        leaf.start_new_bset(alloc_seq(next_bset_seq));
        return;
    }
    let last = leaf.bset_count() - 1;
    let last_h = leaf.bset_header(last);
    let last_full = (last_h.nkeys as usize) >= BSET_SOFT_LIMIT;
    if !last_full {
        return;
    }
    if leaf.bset_count() < BSET_TREE_NR_MAX {
        leaf.start_new_bset(alloc_seq(next_bset_seq));
        return;
    }
    #[cfg(test)]
    {
        COMPACT_ON_FULL_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    compact_leaf_in_place(leaf, alloc_seq(next_bset_seq));
}

/// Build a `DiskEntry` for a leaf write from a (sortable_key, kind, value)
/// triple. Caller has already appended snap_id to the logical key.
fn build_leaf_entry(sortable_key: &[u8], kind: EntryKind, value: &[u8]) -> DiskEntry {
    let mut entry = DiskEntry::empty();
    entry.set_key(sortable_key);
    entry.set_kind(kind);
    let len = value.len().min(MAX_VALUE_SIZE);
    entry.payload.value_mut()[..len].copy_from_slice(&value[..len]);
    entry.payload.value_mut()[len..].fill(0);
    entry.value_len = len as u8;
    entry
}

// ---------- Recursive operations ----------

/// How a leaf write lands on its target node.
///
/// - `InPlace`: mutate the cached leaf at a stable `block_nr` (the hot,
///   non-transactional path). Ancestors keep their child pointers, the root
///   is unchanged, and the dirty node is relocated later by `checkpoint_flush`.
/// - `Cow`: clone the leaf onto a fresh block (classic copy-on-write). Used
///   inside transactions so an aborted transaction leaves the original tree
///   untouched, and in-transaction reads at `pending_root` see prior writes.
///
/// A split always allocates fresh blocks regardless of mode.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WriteMode {
    InPlace,
    Cow,
}

fn find(store: &BlockStore, block_nr: u64, key: &[u8]) -> Result<Option<Vec<u8>>> {
    Ok(
        find_raw(store, block_nr, key)?.and_then(|(value, kind)| match kind {
            EntryKind::Live => Some(value),
            EntryKind::Deleted | EntryKind::Whiteout => None,
        }),
    )
}

/// Like `find` but returns the entry's value AND its kind tag, so callers can
/// distinguish "no entry" from "tombstone". Used by `find_visible` to decide
/// whether to keep walking the ancestor chain or stop.
fn find_raw(store: &BlockStore, block_nr: u64, key: &[u8]) -> Result<Option<(Vec<u8>, EntryKind)>> {
    // One node visit under the cache lock: either resolve the result (leaf)
    // or compute the child block number to descend into. We recurse *after*
    // the closure returns, so the lock is not held across the recursion.
    enum Step {
        Found(Option<(Vec<u8>, EntryKind)>),
        Descend(u64),
    }
    let step = store.with_node(block_nr, |node| {
        if node.level() == 0 {
            // Leaf: cross-bset merged lookup. The highest-seq match wins so a
            // newer write in a later bset shadows the older entry.
            match merged_find(node, key) {
                Some(hit) => {
                    let entry = node.entry_at(hit.bset_idx, hit.entry_idx);
                    Step::Found(Some((
                        node.value_bytes_at(hit.bset_idx, hit.entry_idx).to_vec(),
                        entry.kind_enum(),
                    )))
                }
                None => Step::Found(None),
            }
        } else {
            // Internal nodes are always single-bset (they don't accumulate
            // sorted runs the way leaves do).
            match node.search(key) {
                Ok(idx) => Step::Descend(node.child_block(idx + 1)),
                Err(idx) => Step::Descend(node.child_block(idx)),
            }
        }
    })?;
    match step {
        Step::Found(res) => Ok(res),
        Step::Descend(child) => find_raw(store, child, key),
    }
}

/// Walk the snapshot ancestor chain and return the value visible at the
/// chain's head, or None on tombstone / no entry. Used by both
/// `Btree::find_visible` and `Tx::find_visible`.
fn find_visible(
    store: &BlockStore,
    block_nr: u64,
    logical: &[u8],
    chain: &[SnapId],
) -> Result<Option<Vec<u8>>> {
    Ok(find_visible_with_snap(store, block_nr, logical, chain)?.map(|(v, _)| v))
}

fn find_visible_with_snap(
    store: &BlockStore,
    block_nr: u64,
    logical: &[u8],
    chain: &[SnapId],
) -> Result<Option<(Vec<u8>, SnapId)>> {
    for &snap in chain {
        let (buf, len) = sortable_key(logical, snap);
        match find_raw(store, block_nr, &buf[..len])? {
            Some((value, EntryKind::Live)) => return Ok(Some((value, snap))),
            Some((_, EntryKind::Deleted | EntryKind::Whiteout)) => return Ok(None),
            None => continue,
        }
    }
    Ok(None)
}

/// COW walk that locates `sortable_key` in a leaf and rewrites its `kind`
/// byte without changing `nkeys` or the bset layout. Used by `delete_at`
/// when the visible entry already lives at our own snap_id, so we can flip
/// its kind to `Deleted` instead of writing a fresh tombstone entry (which
/// would grow the leaf and may force opening a new bset / a split).
///
/// Preconditions:
/// - `sortable_key` is present in the tree (caller verified via
///   `find_visible_with_snap`).
/// - The visible entry's snap_id equals the snap encoded in `sortable_key`.
///
/// Picks the highest-seq bset that contains the key — that's the entry the
/// merged read path would return. Flipping any older copy would be shadowed.
fn flip_kind_in_place(
    store: &BlockStore,
    block_nr: u64,
    sortable_key: &[u8],
    new_kind: EntryKind,
    mode: WriteMode,
) -> Result<u64> {
    // Locate the highest-seq copy of the key in a leaf (that's the one
    // merged_find surfaces) and set its kind. Shared by both modes.
    let flip = |leaf: &mut BtreeNodeRaw| {
        let mut best: Option<(usize, usize, u64)> = None;
        for b in 0..leaf.bset_count() {
            if let Ok(idx) = leaf.bset_search(b, sortable_key) {
                let seq = leaf.bset_header(b).seq;
                if best.is_none_or(|(_, _, s)| seq > s) {
                    best = Some((b, idx, seq));
                }
            }
        }
        let (b, i, _) = best.expect("flip_kind_in_place: key not found in leaf");
        leaf.entry_at_mut(b, i).set_kind(new_kind);
    };

    let level = store.with_node(block_nr, |n| n.level())?;
    if level == 0 {
        return match mode {
            // nkeys and block_nr unchanged, so no ancestor pointer changes.
            WriteMode::InPlace => {
                store.with_node_mut(block_nr, flip)?;
                Ok(block_nr)
            }
            // Clone onto a fresh block; caller COWs the path upward.
            WriteMode::Cow => {
                let new_block = store.alloc();
                let mut new_node = store.with_node(block_nr, clone_to_heap)?;
                flip(&mut new_node);
                store.write_node(new_block, &new_node)?;
                Ok(new_block)
            }
        };
    }
    // Internal: descend to the child that holds the key.
    let (child_idx, child_nr) = store.with_node(block_nr, |node| {
        let idx = match node.search(sortable_key) {
            Ok(i) => i + 1,
            Err(i) => i,
        };
        (idx, node.child_block(idx))
    })?;
    let new_child = flip_kind_in_place(store, child_nr, sortable_key, new_kind, mode)?;
    match mode {
        WriteMode::InPlace => {
            debug_assert_eq!(
                new_child, child_nr,
                "InPlace flip must not move the child block"
            );
            Ok(block_nr)
        }
        WriteMode::Cow => {
            // Child was COWed to a fresh block; patch our pointer onto a fresh
            // block too, propagating the copy up to the root.
            let new_block = store.alloc();
            let mut new_node = store.with_node(block_nr, clone_to_heap)?;
            new_node.set_child_block(child_idx, new_child);
            store.write_node(new_block, &new_node)?;
            Ok(new_block)
        }
    }
}

/// Insert an entry, returning the block number of the (possibly new) root of
/// the subtree.
///
/// Between checkpoints this is **not** COW for the common case: a write that
/// the target leaf can absorb (overwrite, append into a bset, open a new bset,
/// or compact-when-full) is applied **in place** via `with_node_mut`, leaving
/// the leaf's `block_nr` — and therefore every ancestor's child pointer, and
/// the root — unchanged. Only a **split** (leaf/internal at capacity) still
/// allocates fresh blocks and COWs the path from the split point up to the
/// root. The still-dirty in-place nodes are relocated to fresh blocks later,
/// in one batch, by [`checkpoint_flush`]. See docs/in-place-append-plan.md.
///
/// `kind` is the entry kind (Live for normal inserts, Deleted/Whiteout for
/// tombstones). On a found-key path the existing entry's kind is replaced.
fn insert(
    store: &BlockStore,
    next_bset_seq: &mut u64,
    block_nr: u64,
    key: &[u8],
    value: &[u8],
    kind: EntryKind,
    mode: WriteMode,
) -> Result<u64> {
    let level = store.with_node(block_nr, |n| n.level())?;
    if level == 0 {
        insert_leaf(store, next_bset_seq, block_nr, key, value, kind, mode)
    } else {
        insert_internal(store, next_bset_seq, block_nr, key, value, kind, mode)
    }
}

fn insert_leaf(
    store: &BlockStore,
    next_bset_seq: &mut u64,
    block_nr: u64,
    key: &[u8],
    value: &[u8],
    kind: EntryKind,
    mode: WriteMode,
) -> Result<u64> {
    // Peek (read-lock only) to decide the shape of the write:
    // - `absorbs_in_place`: the key already lives in the latest bset, so the
    //   write is a pure overwrite (no growth, no new bset).
    // - otherwise, if the leaf is already at capacity, we must split.
    // Splitting needs the whole node cloned out; the in-place cases don't.
    enum Plan {
        InPlace { absorbs: bool },
        Split(Box<BtreeNodeRaw>),
    }
    let plan = store.with_node(block_nr, |node| {
        let absorbs = if node.bset_count() == 0 {
            false
        } else {
            let last = node.bset_count() - 1;
            node.bset_search(last, key).is_ok()
        };
        if !absorbs && node.nkeys() >= MAX_ENTRIES {
            Plan::Split(clone_to_heap(node))
        } else {
            Plan::InPlace { absorbs }
        }
    })?;

    // A closure that applies the write (overwrite-in-latest-bset, or
    // ensure-writable + sort-insert) to a leaf node in memory. Shared by both
    // the in-place and COW application paths.
    let apply = |leaf: &mut BtreeNodeRaw, absorbs: bool, next_bset_seq: &mut u64| {
        if absorbs {
            let last_idx = leaf.bset_count() - 1;
            let entry_idx = leaf
                .bset_search(last_idx, key)
                .expect("absorbs implies present in last bset");
            *leaf.entry_at_mut(last_idx, entry_idx) = build_leaf_entry(key, kind, value);
        } else {
            ensure_writable_last_bset(leaf, next_bset_seq);
            let entry = build_leaf_entry(key, kind, value);
            // Return value (overwrite vs new slot) is irrelevant: the entry is
            // written either way.
            let _ = leaf.sort_insert_into_last_bset(&entry);
        }
    };

    let old_node = match plan {
        Plan::InPlace { absorbs } => match mode {
            // Hot path: mutate the cached leaf and keep its block_nr. The node
            // is dirty; checkpoint_flush relocates it to a fresh block later.
            WriteMode::InPlace => {
                store.with_node_mut(block_nr, |leaf| apply(leaf, absorbs, next_bset_seq))?;
                return Ok(block_nr);
            }
            // Transaction path: clone onto a fresh block so the original stays
            // intact for abort / in-transaction reads at the old root.
            WriteMode::Cow => {
                let new_block = store.alloc();
                let mut new_node = store.with_node(block_nr, clone_to_heap)?;
                apply(&mut new_node, absorbs, next_bset_seq);
                store.write_node(new_block, &new_node)?;
                return Ok(new_block);
            }
        },
        Plan::Split(node) => node,
    };

    // Split path (rare): allocate fresh blocks and COW upward. The split
    // internally compacts a multi-bset source so the resulting halves are
    // clean single-bset nodes. New blocks always have block_nr above every
    // block referenced by the last on-disk checkpoint, so writing them
    // through never overwrites committed state.
    let (mut root_box, root_nr, left_block, right_block) =
        split_leaf_node(store, next_bset_seq, &old_node)?;
    root_box.set_child_block(0, left_block);
    root_box.set_child_block(1, right_block);

    // Decide which half the key should be inserted into based on the
    // separator (which root_box already knows about).
    let child_idx = match root_box.search(key) {
        Ok(i) => i + 1,
        Err(i) => i,
    };
    let child_nr = root_box.child_block(child_idx);
    let child_level = store.with_node(child_nr, |n| n.level())?;
    let new_child_nr = insert(store, next_bset_seq, child_nr, key, value, kind, mode)?;

    let new_child_level = store.with_node(new_child_nr, |n| n.level())?;
    if new_child_level > child_level {
        // Child itself split — promote its median to the (still unwritten)
        // parent. promote_to_parent will write a fresh parent (possibly
        // cascade-splitting it) and return its nr.
        let (median_key, left, right) = store.with_node(new_child_nr, |new_child| {
            (
                new_child.entry(0).key_bytes().to_vec(),
                new_child.child_block(0),
                new_child.child_block(1),
            )
        })?;
        let _ = root_nr; // intentionally leaked: orphaned alloc, no GC v1
        return promote_to_parent(store, &root_box, child_idx, &median_key, left, right);
    }

    // No cascade split: finalize root_box by pointing the child slot at the
    // (possibly COW-replaced) child and commit to store.
    root_box.set_child_block(child_idx, new_child_nr);
    store.write_node(root_nr, &root_box)?;
    Ok(root_nr)
}

fn insert_internal(
    store: &BlockStore,
    next_bset_seq: &mut u64,
    block_nr: u64,
    key: &[u8],
    value: &[u8],
    kind: EntryKind,
    mode: WriteMode,
) -> Result<u64> {
    let (child_idx, child_nr) = store.with_node(block_nr, |node| {
        let idx = match node.search(key) {
            Ok(i) => i + 1,
            Err(i) => i,
        };
        (idx, node.child_block(idx))
    })?;
    let child_level = store.with_node(child_nr, |n| n.level())?;

    let new_child_nr = insert(store, next_bset_seq, child_nr, key, value, kind, mode)?;

    // Child split and grew a level — promote its median key to this level.
    let new_child_level = store.with_node(new_child_nr, |n| n.level())?;
    if new_child_level > child_level {
        let (median_key, left, right) = store.with_node(new_child_nr, |new_child| {
            (
                new_child.entry(0).key_bytes().to_vec(),
                new_child.child_block(0),
                new_child.child_block(1),
            )
        })?;
        // promote_to_parent builds a fresh parent; child_idx is only valid
        // against the old parent, so we must not touch the result afterward.
        // Clone the node out from under the lock first — promote_to_parent
        // re-enters the store (alloc / write_node), which would deadlock if
        // called inside a with_node closure.
        let old_node = store.with_node(block_nr, clone_to_heap)?;
        return promote_to_parent(store, &old_node, child_idx, &median_key, left, right);
    }

    // Common case: the child was mutated in place, so its block_nr did not
    // change — the parent's pointer is still correct and the parent needs no
    // rewrite. Return our own block_nr unchanged (no COW, no alloc).
    if new_child_nr == child_nr {
        return Ok(block_nr);
    }

    // The child moved (it was COWed — e.g. a grandchild split forced it to a
    // fresh block). Patch our pointer, COWing this node onto a fresh block.
    let new_block = store.alloc();
    let mut new_node = store.with_node(block_nr, clone_to_heap)?;
    new_node.set_child_block(child_idx, new_child_nr);
    store.write_node(new_block, &new_node)?;
    Ok(new_block)
}

// ---------- Split & promote ----------

/// Split a full leaf node into two halves and create a parent node.
///
/// Returns `(unwritten_root_box, root_nr, left_nr, right_nr)`. The two leaf
/// halves are already persisted in `store`; the parent node is **not yet
/// written** so the caller can finish setting `child_block(0/1)` (and any
/// further child updates after recursing) before the single committing
/// `store.write_node(root_nr, ...)` call.
fn split_leaf_node(
    store: &BlockStore,
    next_bset_seq: &mut u64,
    node: &BtreeNodeRaw,
) -> Result<(Box<BtreeNodeRaw>, u64, u64, u64)> {
    debug_assert!(node.level() == 0);
    debug_assert!(node.nkeys() >= MAX_ENTRIES);

    // Splits assume a single contiguous sorted run. If `node` is multi-bset,
    // compact a temp copy down to a single bset first; the entries thereafter
    // sit in bset 0 in sorted order and the existing copy_within / set_nkeys
    // logic works as before.
    let canonical: Box<BtreeNodeRaw> = if node.bset_count() > 1 {
        let mut tmp = clone_to_heap(node);
        compact_leaf_in_place(&mut tmp, alloc_seq(next_bset_seq));
        tmp
    } else {
        clone_to_heap(node)
    };
    let n = canonical.nkeys();
    let mid = n / 2;
    debug_assert!(mid > 0 && mid < n);
    let median_key_buf = canonical.entry(mid).key_bytes().to_vec();

    let left_block = store.alloc();
    let mut left = clone_to_heap(&canonical);
    left.set_nkeys(mid);
    store.write_node(left_block, &left)?;

    let right_block = store.alloc();
    let mut right = clone_to_heap(&canonical);
    right.bset_entries_mut(0).copy_within(mid..n, 0);
    right.set_nkeys(n - mid);
    store.write_node(right_block, &right)?;

    let root_block = store.alloc();
    let mut root = new_node_on_heap(1);
    root.set_generation(canonical.generation());
    // Internal node with 1 separator + 2 children: claim the size first so
    // entry_mut(0) lands in a valid slot. Children are filled by the caller.
    root.set_nkeys(1);
    root.entry_mut(0).set_key(&median_key_buf);

    Ok((root, root_block, left_block, right_block))
}

/// Split a full internal node into two and create a new parent at level+1.
/// Returns `(unwritten_root_box, root_nr, left_nr, right_nr)`. As with
/// `split_leaf_node`, left/right are already written; the new root is left
/// to the caller to finish + commit.
fn split_internal_node(
    store: &BlockStore,
    node: &BtreeNodeRaw,
) -> Result<(Box<BtreeNodeRaw>, u64, u64, u64)> {
    let n = node.nkeys();
    debug_assert!(n >= MAX_INTERNAL_KEYS);
    debug_assert!(node.level() > 0);
    let mid = n / 2;
    debug_assert!(mid > 0 && mid < n);
    let median_key_buf = node.entry(mid).key_bytes().to_vec();

    let left_block = store.alloc();
    let mut left = clone_to_heap(node);
    left.set_nkeys(mid);
    store.write_node(left_block, &left)?;

    // Separator-in-right: key[mid] goes to the parent only.
    // Right child gets key[mid+1..n] with children c_{mid+1}..c_n.
    let right_block = store.alloc();
    let mut right = new_node_on_heap(node.level());
    right.set_generation(node.generation());
    // Reserve slots before any entry writes (internal nodes store nkeys+1 entries).
    right.set_nkeys(n - mid - 1);
    for i in (mid + 1)..n {
        right
            .entry_mut(i - mid - 1)
            .set_key(node.entry(i).key_bytes());
    }
    for i in 0..=(n - mid - 1) {
        right.set_child_block(i, node.child_block(mid + 1 + i));
    }
    store.write_node(right_block, &right)?;

    let root_block = store.alloc();
    let mut root = new_node_on_heap(node.level() + 1);
    root.set_generation(node.generation());
    root.set_nkeys(1);
    root.entry_mut(0).set_key(&median_key_buf);

    Ok((root, root_block, left_block, right_block))
}

/// Insert a split result (median_key + left/right children) into the parent
/// at child_idx.  Returns the new parent block number (may cascade-split).
fn promote_to_parent(
    store: &BlockStore,
    old_parent: &BtreeNodeRaw,
    child_idx: usize,
    median_key: &[u8],
    left_child: u64,
    right_child: u64,
) -> Result<u64> {
    debug_assert!(old_parent.level() > 0);
    debug_assert!(child_idx <= old_parent.nkeys());

    let old_nkeys = old_parent.nkeys();
    let new_block = store.alloc();
    let mut new_node = new_node_on_heap(old_parent.level());
    new_node.set_generation(old_parent.generation());
    // Reserve slots first (internal nkeys+1 storage). Capacity overflow is
    // checked after the writes via the MAX_INTERNAL_KEYS comparison below.
    new_node.set_nkeys(old_nkeys + 1);

    for i in 0..child_idx {
        new_node
            .entry_mut(i)
            .set_key(old_parent.entry(i).key_bytes());
        new_node.set_child_block(i, old_parent.child_block(i));
    }
    new_node.entry_mut(child_idx).set_key(median_key);
    new_node.set_child_block(child_idx, left_child);
    new_node.set_child_block(child_idx + 1, right_child);
    // Shift remaining keys and children right by 1 to make room.
    for i in child_idx..old_nkeys {
        new_node
            .entry_mut(i + 1)
            .set_key(old_parent.entry(i).key_bytes());
        new_node.set_child_block(i + 2, old_parent.child_block(i + 1));
    }

    if new_node.nkeys() <= MAX_INTERNAL_KEYS {
        store.write_node(new_block, &new_node)?;
        Ok(new_block)
    } else {
        // Internal node also overflows: split it. The freshly-built
        // `new_node` is the source; `new_block` was speculatively allocated
        // and is now orphaned (no GC v1).
        let _ = new_block;
        let (mut root_box, root_nr, new_left, new_right) = split_internal_node(store, &new_node)?;
        root_box.set_child_block(0, new_left);
        root_box.set_child_block(1, new_right);
        store.write_node(root_nr, &root_box)?;
        Ok(root_nr)
    }
}

// ---------- Checkpoint flush ----------

/// Post-order COW-relocate of every dirty node reachable from `block_nr` onto
/// fresh blocks, persisting them via write-through. Returns the (possibly new)
/// block number of this subtree's root.
///
/// This is how in-place edits made between checkpoints become durable. During
/// the interval, hot leaves are mutated in place at a stable block_nr and left
/// dirty-in-cache-only; at checkpoint we walk the touched subtree and rewrite
/// each dirty node to a **fresh** block (never overwriting a block referenced
/// by the last on-disk checkpoint), fixing up parent pointers as children
/// move. A clean node — or a node not resident in cache at all (never touched
/// this interval) — is kept at its current block_nr and not rewritten.
///
/// Safety: because every write targets a freshly-allocated block, a crash
/// anywhere during the flush leaves the previous checkpoint's on-disk tree
/// byte-for-byte intact; the superblock still points at the old root until the
/// caller swaps it (double-fsync in `Fs::sync`). The newly written blocks
/// become unreferenced orphans until a future GC.
fn checkpoint_flush(store: &BlockStore, block_nr: u64) -> Result<u64> {
    // A node absent from the cache was neither read nor written since the last
    // checkpoint, so its whole on-disk subtree is already clean — keep it.
    let Some((level, dirty)) = store.with_cached_node(block_nr, |n, dirty| (n.level(), dirty))
    else {
        return Ok(block_nr);
    };

    if level == 0 {
        if !dirty {
            return Ok(block_nr);
        }
        // Relocate this leaf to a fresh block.
        let node = store.with_node(block_nr, clone_to_heap)?;
        let new_block = store.alloc();
        store.write_node(new_block, &node)?;
        return Ok(new_block);
    }

    // Internal node: flush every child first, collecting any that moved.
    let child_blocks: Vec<u64> = store
        .with_cached_node(block_nr, |node, _| {
            (0..=node.nkeys()).map(|i| node.child_block(i)).collect()
        })
        .expect("node was resident at level read above");

    let mut new_children = Vec::with_capacity(child_blocks.len());
    let mut any_moved = false;
    for &child in &child_blocks {
        let new_child = checkpoint_flush(store, child)?;
        any_moved |= new_child != child;
        new_children.push(new_child);
    }

    // Rewrite this node iff it is itself dirty or a child moved.
    if !dirty && !any_moved {
        return Ok(block_nr);
    }
    let mut node = store.with_node(block_nr, clone_to_heap)?;
    for (i, &new_child) in new_children.iter().enumerate() {
        node.set_child_block(i, new_child);
    }
    let new_block = store.alloc();
    store.write_node(new_block, &node)?;
    Ok(new_block)
}

// ---------- Scan & debug ----------

fn range_scan(
    store: &BlockStore,
    block_nr: u64,
    start: &[u8],
    end: &[u8],
    results: &mut Vec<(Vec<u8>, Vec<u8>)>,
) -> Result<()> {
    // Leaf: collect matching rows under the lock. Internal: collect the
    // (child block, following separator) plan under the lock, then recurse
    // outside it. `InternalPlan` carries each child to descend and the
    // separator key immediately to its right (None for the last child).
    enum Node {
        LeafRows,
        Internal(Vec<(u64, Option<Vec<u8>>)>),
    }
    let plan = store.with_node(block_nr, |node| {
        if node.level() == 0 {
            // Walk the merged iterator (cross-bset, latest-seq wins). The iter
            // yields entries in sortable-key ascending order so we can
            // early-out once we pass `end`.
            for (b, i) in MergedIter::new(node) {
                let entry = node.entry_at(b, i);
                let sk = entry.key_bytes();
                if sk < start {
                    continue;
                }
                if sk >= end {
                    break;
                }
                if entry.kind_enum() != EntryKind::Live {
                    continue;
                }
                results.push((
                    entry.logical_key_bytes().to_vec(),
                    node.value_bytes_at(b, i).to_vec(),
                ));
            }
            Node::LeafRows
        } else {
            let mut i = match node.search(start) {
                Ok(idx) => idx + 1,
                Err(idx) => idx,
            };
            let nchildren = node.nkeys() + 1;
            if i >= nchildren {
                i = nchildren - 1;
            }
            let mut plan = Vec::new();
            while i < nchildren {
                let sep = if i < node.nkeys() {
                    Some(node.entry(i).key_bytes().to_vec())
                } else {
                    None
                };
                plan.push((node.child_block(i), sep));
                i += 1;
            }
            Node::Internal(plan)
        }
    })?;
    if let Node::Internal(plan) = plan {
        for (child, sep) in plan {
            range_scan(store, child, start, end, results)?;
            // Separator-in-right: once a separator >= end, all further
            // subtrees are out of range.
            if let Some(sep) = sep
                && sep.as_slice() >= end
            {
                return Ok(());
            }
        }
    }
    Ok(())
}

/// Like `range_scan` but emits *every* entry (including tombstones) with
/// full metadata: `(logical_key, snap_id, kind, value)`. Backs the public
/// `Btree::range_scan_all` (BTREE_ITER_ALL_SNAPSHOTS in bcachefs).
fn range_scan_all(
    store: &BlockStore,
    block_nr: u64,
    start: &[u8],
    end: &[u8],
    results: &mut Vec<AllSnapRow>,
) -> Result<()> {
    enum Node {
        LeafRows,
        Internal(Vec<(u64, Option<Vec<u8>>)>),
    }
    let plan = store.with_node(block_nr, |node| {
        if node.level() == 0 {
            // Cross-bset merged iter, but unlike `range_scan` we keep
            // tombstones. Lower-seq duplicates are dropped by the iterator's
            // tie-break logic.
            for (b, i) in MergedIter::new(node) {
                let entry = node.entry_at(b, i);
                let sk = entry.key_bytes();
                if sk < start {
                    continue;
                }
                if sk >= end {
                    break;
                }
                results.push((
                    entry.logical_key_bytes().to_vec(),
                    entry.snap_id(),
                    entry.kind_enum(),
                    node.value_bytes_at(b, i).to_vec(),
                ));
            }
            Node::LeafRows
        } else {
            let mut i = match node.search(start) {
                Ok(idx) => idx + 1,
                Err(idx) => idx,
            };
            let nchildren = node.nkeys() + 1;
            if i >= nchildren {
                i = nchildren - 1;
            }
            let mut plan = Vec::new();
            while i < nchildren {
                let sep = if i < node.nkeys() {
                    Some(node.entry(i).key_bytes().to_vec())
                } else {
                    None
                };
                plan.push((node.child_block(i), sep));
                i += 1;
            }
            Node::Internal(plan)
        }
    })?;
    if let Node::Internal(plan) = plan {
        for (child, sep) in plan {
            range_scan_all(store, child, start, end, results)?;
            if let Some(sep) = sep
                && sep.as_slice() >= end
            {
                return Ok(());
            }
        }
    }
    Ok(())
}

fn dump(store: &BlockStore, block_nr: u64, indent: usize) {
    let prefix = "  ".repeat(indent);
    // Collect the child (block, separator) plan under the lock; recurse after.
    let children: Vec<(u64, Option<Vec<u8>>)> = store
        .with_node(block_nr, |node| {
            if node.level() == 0 {
                print!(
                    "{prefix}[leaf blk={block_nr} bsets={} keys=",
                    node.bset_count()
                );
                // Walk merged-sorted view so debug output is in canonical order.
                for (b, i) in MergedIter::new(node) {
                    print!(" {:02x?}", node.entry_at(b, i).key_bytes());
                }
                println!("]");
                Vec::new()
            } else {
                println!(
                    "{prefix}[internal blk={block_nr} level={} keys={}]",
                    node.level(),
                    node.nkeys()
                );
                (0..=node.nkeys())
                    .map(|i| {
                        let sep = if i < node.nkeys() {
                            Some(node.entry(i).key_bytes().to_vec())
                        } else {
                            None
                        };
                        (node.child_block(i), sep)
                    })
                    .collect()
            }
        })
        .unwrap();
    for (child, sep) in children {
        dump(store, child, indent + 1);
        if let Some(sep) = sep {
            println!("{prefix}  -- {sep:02x?} --");
        }
    }
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn key(i: u32) -> [u8; 4] {
        i.to_be_bytes()
    }

    fn val(i: u32) -> [u8; 4] {
        (i + 10000).to_be_bytes()
    }

    /// Direct call to the internal `find` against a specific root block.
    /// Tests use this to look into historical (pre-overwrite) roots that the
    /// public Btree API doesn't expose. Wraps the snap_id append step.
    fn find_at_root(store: &BlockStore, root: u64, logical: &[u8]) -> Result<Option<Vec<u8>>> {
        let (buf, len) = sortable_key(logical, ROOT_SNAP);
        find(store, root, &buf[..len])
    }

    #[test]
    fn test_single_insert_and_find() {
        let mut tree = Btree::new();
        tree.insert(&key(1), &val(1)).unwrap();
        assert_eq!(
            tree.find(&key(1)).unwrap().as_deref(),
            Some(val(1).as_slice())
        );
        assert_eq!(tree.find(&key(99)).unwrap().as_deref(), None);
        tree.verify();
    }

    #[test]
    #[cfg_attr(miri, ignore = "throughput stress test; no new unsafe path")]
    fn test_multiple_inserts() {
        let mut tree = Btree::new();
        let n = 1000u32;
        for i in 0..n {
            tree.insert(&key(i), &val(i)).unwrap();
        }
        for i in 0..n {
            assert_eq!(
                tree.find(&key(i)).unwrap().as_deref(),
                Some(val(i).as_slice()),
                "key {i} not found"
            );
        }
        assert_eq!(tree.find(&key(u32::MAX)).unwrap().as_deref(), None);
        tree.verify();
    }

    #[test]
    fn test_cow_preserves_old_root() {
        // Fill the root leaf to capacity (MAX_ENTRIES = 29), record the
        // root_block of that just-full single-leaf state, then push one
        // more key — this forces the leaf to split and the root to grow
        // into an internal node. The old root_block is still in the
        // store (COW = old block stays); reading at it must give back
        // exactly the pre-split snapshot.
        let mut tree = Btree::new();
        let n = MAX_ENTRIES as u32;
        for i in 0..n {
            tree.insert(&key(i), &val(i)).unwrap();
        }

        let old_root_block = tree.root_block;
        // Pre-split, the root is still a single leaf.
        assert_eq!(
            tree.store.read_node_copy(old_root_block).unwrap().level(),
            0,
            "root should still be a leaf at MAX_ENTRIES = {n}"
        );

        // This insert overflows the leaf and forces a split. After it,
        // the *current* root is an internal node at a fresh block; the
        // old block is frozen.
        tree.insert(&key(n), &val(n)).unwrap();
        assert_ne!(tree.root_block, old_root_block);
        assert!(
            tree.store.read_node_copy(tree.root_block).unwrap().level() >= 1,
            "new root should be internal after split"
        );

        // Reading via the frozen old root must see all n keys it had
        // when we recorded it, and must NOT see the post-split insert.
        for i in 0..n {
            assert_eq!(
                find_at_root(&tree.store, old_root_block, &key(i))
                    .unwrap()
                    .as_deref(),
                Some(val(i).as_slice()),
                "frozen old root lost key {i}"
            );
        }
        assert_eq!(
            find_at_root(&tree.store, old_root_block, &key(n))
                .unwrap()
                .as_deref(),
            None,
            "frozen old root must not see post-snap key"
        );
        // Modern view sees everything including key n.
        for i in 0..=n {
            assert_eq!(
                tree.find(&key(i)).unwrap().as_deref(),
                Some(val(i).as_slice())
            );
        }
        tree.verify();
    }

    #[test]
    #[cfg_attr(miri, ignore = "throughput stress test; no new unsafe path")]
    fn test_split_many_keys() {
        let mut tree = Btree::new();
        let n = 2000u32;
        for i in 0..n {
            tree.insert(&key(i), &val(i)).unwrap();
        }
        for i in 0..n {
            assert_eq!(
                tree.find(&key(i)).unwrap().as_deref(),
                Some(val(i).as_slice()),
                "key {i} not found"
            );
        }
        tree.verify();
    }

    #[test]
    #[cfg_attr(miri, ignore = "throughput stress test; no new unsafe path")]
    fn test_range_scan() {
        let mut tree = Btree::new();
        for i in 0u32..5000 {
            tree.insert(&key(i), &val(i)).unwrap();
        }
        tree.verify();

        let results = tree.range_scan(&key(10), &key(20)).unwrap();
        assert_eq!(results.len(), 10);
        for (j, (k, v)) in results.iter().enumerate() {
            assert_eq!(k.as_slice(), &key(10 + j as u32));
            assert_eq!(v.as_slice(), &val(10 + j as u32));
        }
    }

    #[test]
    fn test_overwrite() {
        let mut tree = Btree::new();
        tree.insert(&key(1), b"old").unwrap();
        assert_eq!(
            tree.find(&key(1)).unwrap().as_deref(),
            Some(b"old".as_slice())
        );

        tree.insert(&key(1), b"new").unwrap();
        assert_eq!(
            tree.find(&key(1)).unwrap().as_deref(),
            Some(b"new".as_slice())
        );
        tree.verify();
    }

    #[test]
    fn test_dump() {
        let mut tree = Btree::new();
        for i in 0u32..10 {
            tree.insert(&key(i), &val(i)).unwrap();
        }
        tree.dump();
        tree.verify();
    }

    #[test]
    #[cfg_attr(miri, ignore = "throughput stress test; no new unsafe path")]
    fn test_reverse_insert() {
        let mut tree = Btree::new();
        let n = 2000u32;
        for i in (0..n).rev() {
            tree.insert(&key(i), &val(i)).unwrap();
        }
        for i in 0..n {
            assert_eq!(
                tree.find(&key(i)).unwrap().as_deref(),
                Some(val(i).as_slice())
            );
        }
        tree.verify();
    }

    #[test]
    fn test_zerocopy_round_trip() {
        use crate::block_btree::BtreeNodeRaw;
        use zerocopy::{FromBytes, IntoBytes};

        let mut node = BtreeNodeRaw::new(0);
        node.start_new_bset(0);
        node.set_nkeys(3);
        node.set_generation(42);
        node.entry_mut(0).set_key_with_snap(b"hello", ROOT_SNAP);
        node.entry_mut(1).set_key_with_snap(b"world", ROOT_SNAP);
        node.entry_mut(2).set_key_with_snap(b"foo", ROOT_SNAP);

        let bytes: &[u8] = node.as_bytes();
        assert_eq!(bytes.len(), 4096);

        let restored = BtreeNodeRaw::ref_from_bytes(bytes).unwrap();
        assert_eq!(restored.nkeys(), 3);
        assert_eq!(restored.generation(), 42);
        assert_eq!(restored.level(), 0);
        assert_eq!(restored.entry(0).logical_key_bytes(), b"hello");
        assert_eq!(restored.entry(1).logical_key_bytes(), b"world");
        assert_eq!(restored.entry(2).logical_key_bytes(), b"foo");
        assert_eq!(restored.entry(0).snap_id(), ROOT_SNAP);
    }

    // ---------- Random-key tests ----------

    /// Simple xorshift64 PRNG for deterministic random tests.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn next_u32(&mut self) -> u32 {
            (self.next() & 0xFFFF_FFFF) as u32
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "throughput stress test; no new unsafe path")]
    fn test_random_keys() {
        let mut rng = Rng(0xDEAD_BEEF_CAFE);
        let mut tree = Btree::new();
        let mut reference = HashMap::new();

        let n = 5000u32;
        for _ in 0..n {
            let k = rng.next_u32();
            let v = rng.next_u32();
            tree.insert(&key(k), &val(v)).unwrap();
            reference.insert(k, v);
        }

        // Verify every inserted key can be found with the correct value.
        for (&k, &v) in &reference {
            assert_eq!(
                tree.find(&key(k)).unwrap().as_deref(),
                Some(val(v).as_slice()),
                "random key {k} not found or value mismatch"
            );
        }

        // Verify keys not in the reference return None.
        for _ in 0..100 {
            let probe = rng.next_u32();
            if !reference.contains_key(&probe) {
                assert_eq!(tree.find(&key(probe)).unwrap().as_deref(), None);
            }
        }

        assert_eq!(count_keys(&tree.store, tree.root_block), reference.len());
        tree.verify();
    }

    #[test]
    #[cfg_attr(miri, ignore = "throughput stress test; no new unsafe path")]
    fn test_random_overwrite_consistency() {
        let mut rng = Rng(0x1234_5678);
        let mut tree = Btree::new();
        let mut reference = HashMap::new();

        // Insert, then overwrite a subset.
        for i in 0..2000u32 {
            tree.insert(&key(i), &val(i)).unwrap();
            reference.insert(i, i);
        }
        for _ in 0..1000 {
            let k = rng.next_u32() % 2000;
            let v = rng.next_u32();
            tree.insert(&key(k), &val(v)).unwrap();
            reference.insert(k, v);
        }

        for (&k, &v) in &reference {
            assert_eq!(
                tree.find(&key(k)).unwrap().as_deref(),
                Some(val(v).as_slice())
            );
        }
        assert_eq!(count_keys(&tree.store, tree.root_block), reference.len());
        tree.verify();
    }

    // ---------- Split propagation ----------

    #[test]
    #[cfg_attr(miri, ignore = "throughput stress test; no new unsafe path")]
    fn test_split_propagation_forced_root_split() {
        // Insert keys in ascending order until the root splits multiple times.
        // With MAX_ENTRIES=29, filling 29 leaves forces a 3-level tree.
        let mut tree = Btree::new();
        let n = 2000u32;
        for i in 0..n {
            tree.insert(&key(i), &val(i)).unwrap();
        }

        // Tree should have grown beyond level 1.
        let root = tree.store.read_node_copy(tree.root_block).unwrap();
        assert!(
            root.level() >= 2,
            "expected multi-level tree, got level {}",
            root.level()
        );

        // Every key must still be findable.
        for i in 0..n {
            assert_eq!(
                tree.find(&key(i)).unwrap().as_deref(),
                Some(val(i).as_slice())
            );
        }
        assert_eq!(count_keys(&tree.store, tree.root_block), n as usize);
        tree.verify();
    }

    #[test]
    fn test_split_at_exact_max_entries() {
        // Insert exactly MAX_ENTRIES keys (no split needed), then one more (forces split).
        let mut tree = Btree::new();
        let max = crate::block_btree::MAX_ENTRIES as u32;
        for i in 0..max {
            tree.insert(&key(i), &val(i)).unwrap();
        }
        // Root should still be a leaf.
        assert_eq!(
            tree.store.read_node_copy(tree.root_block).unwrap().level(),
            0
        );

        // One more triggers a split.
        tree.insert(&key(max), &val(max)).unwrap();
        assert_eq!(
            tree.store.read_node_copy(tree.root_block).unwrap().level(),
            1
        );

        for i in 0..=max {
            assert_eq!(
                tree.find(&key(i)).unwrap().as_deref(),
                Some(val(i).as_slice())
            );
        }
        assert_eq!(count_keys(&tree.store, tree.root_block), (max + 1) as usize);
        tree.verify();
    }

    // ---------- Key size boundary ----------

    #[test]
    fn test_max_key_size() {
        // Phase 2 introduced the snap_id suffix on every entry; the
        // caller-visible "logical" key is bounded by MAX_LOGICAL_KEY_SIZE
        // and the 4-byte snap_id is appended internally.
        let mut tree = Btree::new();

        // Key at exactly MAX_LOGICAL_KEY_SIZE.
        let full_key = vec![0xABu8; MAX_LOGICAL_KEY_SIZE];
        tree.insert(&full_key, b"full").unwrap();
        assert_eq!(
            tree.find(&full_key).unwrap().as_deref(),
            Some(b"full".as_slice())
        );

        // A key one byte shorter must not match the longer one.
        let short_key = vec![0xABu8; MAX_LOGICAL_KEY_SIZE - 1];
        assert_eq!(tree.find(&short_key).unwrap().as_deref(), None);

        tree.verify();
    }

    #[test]
    #[should_panic(expected = "logical key too long")]
    fn test_logical_key_overflow_panics() {
        let mut tree = Btree::new();
        let too_long = vec![0xCDu8; MAX_LOGICAL_KEY_SIZE + 1];
        let _ = tree.insert(&too_long, b"x");
    }

    #[test]
    fn test_empty_key() {
        let mut tree = Btree::new();
        tree.insert(b"", b"empty-key").unwrap();
        assert_eq!(
            tree.find(b"").unwrap().as_deref(),
            Some(b"empty-key".as_slice())
        );
        assert_eq!(tree.find(b"x").unwrap().as_deref(), None);
        tree.verify();
    }

    // ---------- Value edge cases ----------

    #[test]
    fn test_empty_value() {
        let mut tree = Btree::new();
        tree.insert(&key(1), b"").unwrap();
        assert_eq!(tree.find(&key(1)).unwrap().as_deref(), Some(b"".as_slice()));
        tree.verify();
    }

    #[test]
    fn test_max_value_size() {
        use crate::block_btree::MAX_VALUE_SIZE;

        let mut tree = Btree::new();
        let full_val = vec![0x42u8; MAX_VALUE_SIZE];
        tree.insert(&key(1), &full_val).unwrap();
        assert_eq!(
            tree.find(&key(1)).unwrap().as_deref(),
            Some(full_val.as_slice())
        );

        // Value exceeding MAX_VALUE_SIZE — truncated.
        let long_val = vec![0x99u8; MAX_VALUE_SIZE + 50];
        tree.insert(&key(2), &long_val).unwrap();
        let truncated_val = vec![0x99u8; MAX_VALUE_SIZE];
        assert_eq!(
            tree.find(&key(2)).unwrap().as_deref(),
            Some(truncated_val.as_slice())
        );

        tree.verify();
    }

    // ---------- COW deeper verification ----------

    #[test]
    fn test_checkpoint_preserves_content() {
        // In-place append dropped the old "a saved root_block is a frozen
        // snapshot" property at the btree level (see
        // docs/in-place-append-plan.md §1.1). What replaces it is the
        // checkpoint: `checkpoint()` relocates every dirty node onto a fresh
        // block and returns a new root, without losing or corrupting content.
        // This test drives inserts across splits, checkpoints, then more
        // in-place edits, and verifies the content survives every checkpoint.
        let mut tree = Btree::new();
        for i in 0u32..500 {
            tree.insert(&key(i), &val(i)).unwrap();
        }

        // Checkpoint: dirty nodes relocate, root changes (something was dirty).
        let pre = tree.root_block;
        tree.checkpoint().unwrap();
        assert_ne!(
            tree.root_block, pre,
            "checkpoint should relocate dirty root"
        );
        for i in 0..500 {
            assert_eq!(
                tree.find(&key(i)).unwrap().as_deref(),
                Some(val(i).as_slice()),
                "key {i} lost across checkpoint"
            );
        }
        assert_eq!(count_keys(&tree.store, tree.root_block), 500);

        // A second checkpoint with nothing dirty is a no-op on the root.
        let clean = tree.root_block;
        tree.checkpoint().unwrap();
        assert_eq!(
            tree.root_block, clean,
            "clean checkpoint must not move root"
        );

        // More in-place overwrites, then checkpoint again — content follows.
        for i in 0..250u32 {
            tree.insert(&key(i), &val(i + 1_000_000)).unwrap();
        }
        tree.checkpoint().unwrap();
        for i in 0..250 {
            assert_eq!(
                tree.find(&key(i)).unwrap().as_deref(),
                Some(val(i + 1_000_000).as_slice())
            );
        }
        for i in 250..500 {
            assert_eq!(
                tree.find(&key(i)).unwrap().as_deref(),
                Some(val(i).as_slice())
            );
        }
        assert_eq!(count_keys(&tree.store, tree.root_block), 500);
        tree.verify();
    }

    #[test]
    fn test_inplace_append_kills_write_amplification() {
        // The headline win: a run of writes the target leaf can absorb should
        // persist ZERO nodes during the run (they mutate the cached leaf in
        // place), and only a small bounded number at checkpoint. Under the old
        // COW-per-op scheme each of the N writes rewrote the whole root→leaf
        // path, so this counter would have been ≥ N.
        let mut tree = Btree::new();
        // Seed a single key so the tree exists; then take the baseline.
        tree.insert(&key(0), &val(0)).unwrap();
        tree.checkpoint().unwrap();
        let base = tree.store.node_writes();

        // Overwrite the same key many times — pure in-place, single leaf.
        const N: u32 = 200;
        for i in 0..N {
            tree.insert(&key(0), &val(i)).unwrap();
        }
        assert_eq!(
            tree.store.node_writes(),
            base,
            "in-place overwrites must not persist any node before checkpoint"
        );

        // One checkpoint persists just the touched path (a single leaf here).
        tree.checkpoint().unwrap();
        let after = tree.store.node_writes();
        assert!(
            after - base <= 2,
            "checkpoint of one hot leaf should persist ≤2 nodes, got {}",
            after - base
        );
        assert_eq!(
            tree.find(&key(0)).unwrap().as_deref(),
            Some(val(N - 1).as_slice())
        );
        tree.verify();
    }

    #[test]
    fn test_inplace_overwrite_keeps_root_stable() {
        // A pure overwrite of a key already in the latest bset is absorbed in
        // place: the leaf's block_nr is unchanged, so no ancestor pointer
        // changes and root_block stays put — the whole point of in-place
        // append (kill per-op COW write amplification). Contrast with a split,
        // which still allocates fresh blocks (covered by
        // test_cow_preserves_old_root).
        let mut tree = Btree::new();
        let n = MAX_ENTRIES as u32 + 10; // ≥ 39, forces an internal root
        for i in 0..n {
            tree.insert(&key(i), &val(i)).unwrap();
        }
        assert!(
            tree.store.read_node_copy(tree.root_block).unwrap().level() >= 1,
            "expected internal root after {n} inserts"
        );

        // Overwrite the most-recently-inserted key (guaranteed to sit in its
        // leaf's latest bset → true in-place overwrite, no growth, no split).
        let root_before = tree.root_block;
        tree.insert(&key(n - 1), &val(n - 1 + 1_000_000)).unwrap();
        assert_eq!(
            tree.root_block, root_before,
            "in-place overwrite must not change root_block (no COW)"
        );
        assert_eq!(
            tree.find(&key(n - 1)).unwrap().as_deref(),
            Some(val(n - 1 + 1_000_000).as_slice()),
            "overwrite must be visible in the modern tree"
        );
        tree.verify();
    }

    // ---------- Range scan edge cases ----------

    #[test]
    fn test_range_scan_empty() {
        let mut tree = Btree::new();
        for i in 0..100u32 {
            tree.insert(&key(i), &val(i)).unwrap();
        }
        // Range that contains nothing.
        let results = tree.range_scan(&key(200), &key(300)).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_range_scan_single_element() {
        let mut tree = Btree::new();
        for i in 0..100u32 {
            tree.insert(&key(i), &val(i)).unwrap();
        }
        let results = tree.range_scan(&key(50), &key(51)).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.as_slice(), &key(50));
    }

    #[test]
    #[cfg_attr(miri, ignore = "throughput stress test; no new unsafe path")]
    fn test_range_scan_entire_tree() {
        let mut tree = Btree::new();
        let n = 500u32;
        for i in 0..n {
            tree.insert(&key(i), &val(i)).unwrap();
        }
        let results = tree.range_scan(&key(0), &key(u32::MAX)).unwrap();
        assert_eq!(results.len(), n as usize);
        for (i, (k, v)) in results.iter().enumerate() {
            assert_eq!(k.as_slice(), &key(i as u32));
            assert_eq!(v.as_slice(), &val(i as u32));
        }
    }

    // ---------- Large-scale stress ----------

    #[test]
    #[cfg_attr(miri, ignore = "throughput stress test; no new unsafe path")]
    fn test_large_scale_10k() {
        let mut tree = Btree::new();
        let n = 10_000u32;
        for i in 0..n {
            tree.insert(&key(i), &val(i)).unwrap();
        }
        for i in 0..n {
            assert_eq!(
                tree.find(&key(i)).unwrap().as_deref(),
                Some(val(i).as_slice())
            );
        }
        assert_eq!(count_keys(&tree.store, tree.root_block), n as usize);

        // Root should be at least level 2 with this many keys.
        let root = tree.store.read_node_copy(tree.root_block).unwrap();
        assert!(root.level() >= 2);

        tree.verify();
    }

    #[test]
    #[cfg_attr(miri, ignore = "throughput stress test; no new unsafe path")]
    fn test_large_scale_random_20k() {
        let mut rng = Rng(0xCAFE_BABE);
        let mut tree = Btree::new();
        let mut reference = HashMap::new();

        for _ in 0..20_000u32 {
            let k = rng.next_u32();
            let v = rng.next_u32();
            tree.insert(&key(k), &val(v)).unwrap();
            reference.insert(k, v);
        }

        for (&k, &v) in &reference {
            assert_eq!(
                tree.find(&key(k)).unwrap().as_deref(),
                Some(val(v).as_slice())
            );
        }
        assert_eq!(count_keys(&tree.store, tree.root_block), reference.len());
        tree.verify();
    }

    // ---------- Invariant: total key count ----------

    #[test]
    #[cfg_attr(miri, ignore = "throughput stress test; no new unsafe path")]
    fn test_key_count_after_mixed_operations() {
        let mut tree = Btree::new();
        let mut reference = HashMap::new();

        // Insert 0..1000.
        for i in 0..1000u32 {
            tree.insert(&key(i), &val(i)).unwrap();
            reference.insert(i, i);
        }
        assert_eq!(count_keys(&tree.store, tree.root_block), 1000);

        // Overwrite 500..600 — count should not change.
        for i in 500..600u32 {
            tree.insert(&key(i), &val(i + 9999)).unwrap();
            reference.insert(i, i + 9999);
        }
        assert_eq!(count_keys(&tree.store, tree.root_block), 1000);

        // Insert new keys 1000..2000.
        for i in 1000..2000u32 {
            tree.insert(&key(i), &val(i)).unwrap();
            reference.insert(i, i);
        }
        assert_eq!(count_keys(&tree.store, tree.root_block), 2000);

        // Verify all values.
        for (&k, &v) in &reference {
            assert_eq!(
                tree.find(&key(k)).unwrap().as_deref(),
                Some(val(v).as_slice())
            );
        }
        tree.verify();
    }

    // ---------- Balance invariance ----------

    /// Verify all leaves are at the same depth and return that depth.
    fn assert_balanced(tree: &Btree) -> usize {
        let mut depths = Vec::new();
        collect_leaf_depths(&tree.store, tree.root_block, 0, &mut depths);
        assert!(!depths.is_empty(), "tree has no leaves");
        let first = depths[0];
        for (i, &d) in depths.iter().enumerate() {
            assert_eq!(d, first, "leaf {i} is at depth {d}, expected {first}");
        }
        first
    }

    #[test]
    #[cfg_attr(miri, ignore = "throughput stress test; no new unsafe path")]
    fn test_balance_sequential_inserts() {
        let mut tree = Btree::new();
        // Insert enough keys to create multiple levels.
        for i in 0..5000u32 {
            tree.insert(&key(i), &val(i)).unwrap();
        }
        let depth = assert_balanced(&tree);
        assert!(depth >= 2, "expected multi-level tree, got depth {depth}");
        tree.verify();
    }

    #[test]
    #[cfg_attr(miri, ignore = "throughput stress test; no new unsafe path")]
    fn test_balance_reverse_inserts() {
        let mut tree = Btree::new();
        for i in (0..5000u32).rev() {
            tree.insert(&key(i), &val(i)).unwrap();
        }
        assert_balanced(&tree);
        tree.verify();
    }

    #[test]
    #[cfg_attr(miri, ignore = "throughput stress test; no new unsafe path")]
    fn test_balance_random_inserts() {
        let mut rng = Rng(0xBEEF_FACE);
        let mut tree = Btree::new();
        for _ in 0..5000u32 {
            let k = rng.next_u32();
            tree.insert(&key(k), &val(k)).unwrap();
        }
        assert_balanced(&tree);
        tree.verify();
    }

    #[test]
    #[cfg_attr(miri, ignore = "throughput stress test; no new unsafe path")]
    fn test_balance_after_overwrites() {
        let mut tree = Btree::new();
        for i in 0..2000u32 {
            tree.insert(&key(i), &val(i)).unwrap();
        }
        // Overwrite doesn't change tree structure, balance must hold.
        for i in 0..2000u32 {
            tree.insert(&key(i), &val(i + 99999)).unwrap();
        }
        assert_balanced(&tree);
        tree.verify();
    }

    #[test]
    #[cfg_attr(miri, ignore = "throughput stress test; no new unsafe path")]
    fn test_balance_at_each_split_level() {
        // Insert one key at a time and verify balance after every insert.
        let mut tree = Btree::new();
        for i in 0..2000u32 {
            tree.insert(&key(i), &val(i)).unwrap();
            assert_balanced(&tree);
        }
        tree.verify();
    }

    #[test]
    #[cfg_attr(miri, ignore = "throughput stress test; no new unsafe path")]
    fn test_balance_height_grows_logarithmically() {
        // With MAX_ENTRIES=29, a B-tree of height h can hold at least
        // 29^h keys (each leaf full). So height should be O(log_29(n)).
        let mut tree = Btree::new();
        let mut prev_depth = 0usize;
        let mut prev_n = 0u32;

        for n in [50, 500, 5000, 20000u32] {
            for i in prev_n..n {
                tree.insert(&key(i), &val(i)).unwrap();
            }
            let depth = assert_balanced(&tree);

            // Height should never jump by more than 1 between these checkpoints.
            if prev_depth > 0 {
                assert!(
                    depth <= prev_depth + 1,
                    "height jumped from {prev_depth} to {depth} \
                     when inserting {prev_n}..{n}"
                );
            }

            prev_depth = depth;
            prev_n = n;
        }

        // With 20000 keys and branching factor 29, height should be small.
        assert!(
            prev_depth <= 5,
            "unexpectedly deep tree: height={prev_depth}"
        );
        tree.verify();
    }

    #[test]
    #[cfg_attr(miri, ignore = "throughput stress test; no new unsafe path")]
    fn test_balance_leaf_fill_ratio() {
        // Every leaf (except possibly the last split remainder) should have
        // at least MAX_ENTRIES/2 keys. This is the standard B-tree invariant.
        let mut tree = Btree::new();
        for i in 0..5000u32 {
            tree.insert(&key(i), &val(i)).unwrap();
        }

        let min_keys = crate::block_btree::MAX_ENTRIES / 2;
        check_leaf_fill(&tree.store, tree.root_block, min_keys);
        tree.verify();
    }

    fn check_leaf_fill(store: &BlockStore, block_nr: u64, min_keys: usize) {
        let node = store.read_node_copy(block_nr).unwrap();
        if node.level() == 0 {
            assert!(
                node.nkeys() >= min_keys,
                "leaf blk={block_nr} has only {} keys, expected >= {min_keys}",
                node.nkeys()
            );
        } else {
            for i in 0..=node.nkeys() {
                check_leaf_fill(store, node.child_block(i), min_keys);
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "throughput stress test; no new unsafe path")]
    fn test_balance_cow_snapshots_all_balanced() {
        let mut tree = Btree::new();
        let mut snapshots = Vec::new();

        for i in 0..3000u32 {
            tree.insert(&key(i), &val(i)).unwrap();
            if i % 500 == 499 {
                snapshots.push(tree.root_block);
            }
        }

        // Every historical snapshot must also be balanced.
        for &snap in &snapshots {
            let mut depths = Vec::new();
            collect_leaf_depths(&tree.store, snap, 0, &mut depths);
            let first = depths[0];
            for (i, &d) in depths.iter().enumerate() {
                assert_eq!(d, first, "snapshot leaf {i} depth {d} != {first}");
            }
        }
        tree.verify();
    }

    // ---------- Phase 2: snap_id encoding ----------

    #[test]
    fn snap_id_distinguishes_entries_with_same_logical_key() {
        // Two entries sharing a logical key but living at different snap_ids
        // must coexist; find_at must dispatch by snap_id.
        let mut tree = Btree::new();
        tree.insert_at(b"k", ROOT_SNAP, b"v_root").unwrap();
        tree.insert_at(b"k", 100, b"v_100").unwrap();

        assert_eq!(
            tree.find_at(b"k", ROOT_SNAP).unwrap().as_deref(),
            Some(b"v_root".as_slice())
        );
        assert_eq!(
            tree.find_at(b"k", 100).unwrap().as_deref(),
            Some(b"v_100".as_slice())
        );
        // No ancestor walk yet (Phase 4): a snap_id with no entry returns None.
        assert_eq!(tree.find_at(b"k", 200).unwrap().as_deref(), None);
        tree.verify();
    }

    #[test]
    fn snap_id_sort_order_smaller_first_per_logical_key() {
        // Within the same logical key, entries are sorted by snap_id ascending
        // (smaller snap_id = more specific = closer to leaf in the snap tree).
        let mut tree = Btree::new();
        tree.insert_at(b"k", ROOT_SNAP, b"v_root").unwrap();
        tree.insert_at(b"k", 50, b"v_50").unwrap();
        tree.insert_at(b"k", 200, b"v_200").unwrap();

        // range_scan_at returns logical keys (no snap_id), so all three appear
        // as the same `b"k"`. With three entries sharing one logical key,
        // results should have 3 rows. Phase 4 will collapse these via the
        // ancestor filter.
        let scan = tree.range_scan_at(b"j", b"l", ROOT_SNAP).unwrap();
        assert_eq!(scan.len(), 3, "{scan:?}");
        // Values appear in snap_id ascending order: 50 < 200 < ROOT_SNAP.
        assert_eq!(scan[0].1, b"v_50");
        assert_eq!(scan[1].1, b"v_200");
        assert_eq!(scan[2].1, b"v_root");
        tree.verify();
    }

    #[test]
    fn range_scan_at_returns_logical_keys_only() {
        // Verify the snap_id suffix is stripped from returned keys.
        let mut tree = Btree::new();
        tree.insert_at(b"alpha", 100, b"a").unwrap();
        tree.insert_at(b"beta", ROOT_SNAP, b"b").unwrap();

        let scan = tree.range_scan(b"a", b"z").unwrap();
        let keys: Vec<&[u8]> = scan.iter().map(|(k, _)| k.as_slice()).collect();
        assert_eq!(keys, vec![&b"alpha"[..], b"beta"]);
    }

    // ---------- Phase 4: ancestor-aware visibility filter ----------

    #[test]
    fn find_visible_inherits_from_ancestor() {
        // ROOT_SNAP has K=v_root. Child snap=100 has no entry of its own.
        // Reading at snap=100 (chain = [100, ROOT_SNAP]) should fall through
        // to the ancestor's entry.
        let mut tree = Btree::new();
        tree.insert_at(b"K", ROOT_SNAP, b"v_root").unwrap();

        let chain_root = vec![ROOT_SNAP];
        let chain_100 = vec![100, ROOT_SNAP];
        assert_eq!(
            tree.find_visible(b"K", &chain_root).unwrap().as_deref(),
            Some(b"v_root".as_slice())
        );
        assert_eq!(
            tree.find_visible(b"K", &chain_100).unwrap().as_deref(),
            Some(b"v_root".as_slice())
        );
    }

    #[test]
    fn find_visible_child_overrides_parent() {
        // Child writes its own value; the more-specific (smaller snap_id)
        // entry wins for the child but NOT for the parent.
        let mut tree = Btree::new();
        tree.insert_at(b"K", ROOT_SNAP, b"v_root").unwrap();
        tree.insert_at(b"K", 100, b"v_child").unwrap();

        let chain_child = vec![100, ROOT_SNAP];
        let chain_root = vec![ROOT_SNAP];
        assert_eq!(
            tree.find_visible(b"K", &chain_child).unwrap().as_deref(),
            Some(b"v_child".as_slice())
        );
        assert_eq!(
            tree.find_visible(b"K", &chain_root).unwrap().as_deref(),
            Some(b"v_root".as_slice()),
            "parent must NOT see child's override"
        );
    }

    #[test]
    fn find_visible_siblings_isolated() {
        // Two children of the same parent see only their own writes (and
        // anything they jointly inherit from the parent).
        let mut tree = Btree::new();
        tree.insert_at(b"shared", ROOT_SNAP, b"v_inherited")
            .unwrap();
        tree.insert_at(b"a_only", 100, b"v_a").unwrap();
        tree.insert_at(b"b_only", 80, b"v_b").unwrap();

        // Sibling chains: A descends from ROOT_SNAP via 100; B via 80.
        let chain_a = vec![100, ROOT_SNAP];
        let chain_b = vec![80, ROOT_SNAP];

        assert_eq!(
            tree.find_visible(b"shared", &chain_a).unwrap().as_deref(),
            Some(b"v_inherited".as_slice())
        );
        assert_eq!(
            tree.find_visible(b"shared", &chain_b).unwrap().as_deref(),
            Some(b"v_inherited".as_slice())
        );
        assert_eq!(
            tree.find_visible(b"a_only", &chain_a).unwrap().as_deref(),
            Some(b"v_a".as_slice())
        );
        // B's chain does not include 100, so a_only is invisible.
        assert_eq!(tree.find_visible(b"a_only", &chain_b).unwrap(), None);
        assert_eq!(tree.find_visible(b"b_only", &chain_a).unwrap(), None);
        assert_eq!(
            tree.find_visible(b"b_only", &chain_b).unwrap().as_deref(),
            Some(b"v_b".as_slice())
        );
    }

    #[test]
    fn range_scan_visible_collapses_per_logical_key() {
        // Three entries on the same logical key, only the most specific
        // ancestor is emitted.
        let mut tree = Btree::new();
        tree.insert_at(b"K", ROOT_SNAP, b"v_root").unwrap();
        tree.insert_at(b"K", 100, b"v_100").unwrap();
        tree.insert_at(b"L", 100, b"v_L").unwrap();

        let chain_50 = vec![50, 100, ROOT_SNAP];
        let scan = tree.range_scan_visible(b"A", b"Z", &chain_50).unwrap();
        // Both K (via snap=100) and L (via snap=100) visible exactly once.
        assert_eq!(scan.len(), 2);
        assert_eq!(scan[0], (b"K".to_vec(), b"v_100".to_vec()));
        assert_eq!(scan[1], (b"L".to_vec(), b"v_L".to_vec()));
    }

    #[test]
    fn range_scan_all_includes_every_entry() {
        // raw scan returns every (logical, snap_id) without any filtering.
        let mut tree = Btree::new();
        tree.insert_at(b"K", ROOT_SNAP, b"v_root").unwrap();
        tree.insert_at(b"K", 100, b"v_100").unwrap();
        tree.insert_at(b"L", 50, b"v_L").unwrap();

        let raw = tree.range_scan_all(b"A", b"Z").unwrap();
        assert_eq!(raw.len(), 3);
        // (logical ASC, snap_id ASC): K@100, K@ROOT, L@50.
        assert_eq!(raw[0].0, b"K");
        assert_eq!(raw[0].1, 100);
        assert_eq!(raw[0].2, EntryKind::Live);
        assert_eq!(raw[0].3, b"v_100");
        assert_eq!(raw[1].0, b"K");
        assert_eq!(raw[1].1, ROOT_SNAP);
        assert_eq!(raw[2].0, b"L");
        assert_eq!(raw[2].1, 50);
    }

    // ---------- Phase 5: Delete with Deleted/Whiteout dispatch ----------

    #[test]
    fn delete_same_snap_writes_deleted_tombstone() {
        // X writes K=v_X, then X deletes K. Visible entry is X's own, so the
        // tombstone kind is Deleted (trivial — compactable).
        let mut tree = Btree::new();
        let chain = vec![ROOT_SNAP];
        tree.insert_at(b"K", ROOT_SNAP, b"v_X").unwrap();
        assert!(tree.delete_at(b"K", ROOT_SNAP, &chain).unwrap());

        // Key invisible at X.
        assert_eq!(tree.find_visible(b"K", &chain).unwrap(), None);
        // Raw scan still has the tombstone with kind=Deleted.
        let raw = tree.range_scan_all(b"J", b"L").unwrap();
        let kinds: Vec<EntryKind> = raw.iter().map(|(_, _, k, _)| *k).collect();
        assert_eq!(kinds, vec![EntryKind::Deleted]);
        tree.verify();
    }

    #[test]
    fn delete_cross_snap_writes_whiteout_tombstone() {
        // Parent has K=v_root. Child snap=100 deletes K — the visible entry
        // came from an ancestor, so we must shadow it with a Whiteout
        // (KEY_TYPE_whiteout in bcachefs).
        let mut tree = Btree::new();
        tree.insert_at(b"K", ROOT_SNAP, b"v_root").unwrap();

        let chain_100 = vec![100, ROOT_SNAP];
        assert!(tree.delete_at(b"K", 100, &chain_100).unwrap());

        // Child no longer sees K.
        assert_eq!(tree.find_visible(b"K", &chain_100).unwrap(), None);
        // Parent still sees its own entry.
        let chain_root = vec![ROOT_SNAP];
        assert_eq!(
            tree.find_visible(b"K", &chain_root).unwrap().as_deref(),
            Some(b"v_root".as_slice())
        );
        // Tombstone is Whiteout, not Deleted.
        let raw = tree.range_scan_all(b"J", b"L").unwrap();
        let by_snap: Vec<(SnapId, EntryKind)> = raw
            .iter()
            .map(|(_, snap, kind, _)| (*snap, *kind))
            .collect();
        assert_eq!(
            by_snap,
            vec![(100, EntryKind::Whiteout), (ROOT_SNAP, EntryKind::Live)]
        );
    }

    #[test]
    fn delete_noop_on_invisible_key() {
        let mut tree = Btree::new();
        let chain = vec![ROOT_SNAP];
        // No insert; key is not visible.
        assert!(!tree.delete_at(b"missing", ROOT_SNAP, &chain).unwrap());
        // Nothing was written to the tree.
        assert_eq!(tree.range_scan_all(b"a", b"z").unwrap().len(), 0);
    }

    #[test]
    fn delete_noop_when_already_tombstoned() {
        // Once K is tombstoned, a second delete is a no-op (find_visible
        // already returns None, no Live entry to shadow).
        let mut tree = Btree::new();
        let chain = vec![ROOT_SNAP];
        tree.insert_at(b"K", ROOT_SNAP, b"v").unwrap();
        assert!(tree.delete_at(b"K", ROOT_SNAP, &chain).unwrap());
        assert!(!tree.delete_at(b"K", ROOT_SNAP, &chain).unwrap());
    }

    #[test]
    fn reinsert_after_delete_overwrites_tombstone() {
        // Re-inserting at the same (key, snap) flips kind back to Live.
        let mut tree = Btree::new();
        let chain = vec![ROOT_SNAP];
        tree.insert_at(b"K", ROOT_SNAP, b"v1").unwrap();
        tree.delete_at(b"K", ROOT_SNAP, &chain).unwrap();
        assert_eq!(tree.find_visible(b"K", &chain).unwrap(), None);

        tree.insert_at(b"K", ROOT_SNAP, b"v2").unwrap();
        assert_eq!(
            tree.find_visible(b"K", &chain).unwrap().as_deref(),
            Some(b"v2".as_slice())
        );
        // Only one entry exists at (K, ROOT_SNAP); the tombstone was patched.
        let raw = tree.range_scan_all(b"J", b"L").unwrap();
        assert_eq!(raw.len(), 1);
        assert_eq!(raw[0].2, EntryKind::Live);
    }

    #[test]
    fn delete_then_range_scan_visible_excludes_tombstone() {
        let mut tree = Btree::new();
        let chain = vec![ROOT_SNAP];
        tree.insert_at(b"a", ROOT_SNAP, b"va").unwrap();
        tree.insert_at(b"b", ROOT_SNAP, b"vb").unwrap();
        tree.insert_at(b"c", ROOT_SNAP, b"vc").unwrap();
        tree.delete_at(b"b", ROOT_SNAP, &chain).unwrap();

        let visible = tree.range_scan_visible(b"a", b"d", &chain).unwrap();
        let keys: Vec<&[u8]> = visible.iter().map(|(k, _)| k.as_slice()).collect();
        assert_eq!(keys, vec![&b"a"[..], b"c"]);
    }

    #[test]
    fn whiteout_does_not_affect_sibling_snapshot() {
        // Two child snapshots A=100, B=80 of ROOT_SNAP. Parent has K=v_root.
        // Child A deletes K — A's whiteout must shadow K only on A's chain,
        // never on B's chain. (Per bcachefs Snapshots doc.)
        let mut tree = Btree::new();
        tree.insert_at(b"K", ROOT_SNAP, b"v_root").unwrap();
        let chain_a = vec![100, ROOT_SNAP];
        let chain_b = vec![80, ROOT_SNAP];

        tree.delete_at(b"K", 100, &chain_a).unwrap();
        assert_eq!(tree.find_visible(b"K", &chain_a).unwrap(), None);
        assert_eq!(
            tree.find_visible(b"K", &chain_b).unwrap().as_deref(),
            Some(b"v_root".as_slice()),
            "sibling B's view of inherited K must be unaffected by A's whiteout"
        );
    }

    // ---------- Phase 6: Btree::transaction ----------

    #[test]
    fn transaction_commits_multiple_ops_atomically() {
        // Drive a single transaction with enough inserts to force splits
        // inside the tx. The tx must commit atomically — readers outside
        // see exactly the pre-tx state until the closure returns Ok, then
        // see all 200+ keys at once. root_block changes exactly once.
        let mut tree = Btree::new();
        tree.insert(b"a", b"1").unwrap();
        let pre_root = tree.root_block;

        const TX_KEYS: u32 = 200; // ≫ MAX_ENTRIES = 29

        tree.transaction(|tx| {
            for i in 0..TX_KEYS {
                let k = format!("k{i:05}");
                let v = format!("v{i:05}");
                tx.insert(k.as_bytes(), ROOT_SNAP, v.as_bytes())?;
            }
            // Reads inside tx see all pending writes (spot check across
            // the range so we know the in-tx walk finds keys that landed
            // in different leaves after split).
            for &i in &[0u32, 1, 27, 28, 29, 100, TX_KEYS - 1] {
                let k = format!("k{i:05}");
                let v = format!("v{i:05}");
                assert_eq!(
                    tx.find_at(k.as_bytes(), ROOT_SNAP)?.as_deref(),
                    Some(v.as_bytes()),
                    "in-tx find of k{i:05} missed"
                );
            }
            // Outside-the-tx state: pre_root is still the published root,
            // so a reader at pre_root sees only "a".
            assert_eq!(
                find_at_root(&tx.btree.store, pre_root, b"a")?.as_deref(),
                Some(b"1".as_slice())
            );
            assert_eq!(find_at_root(&tx.btree.store, pre_root, b"k00000")?, None);
            Ok(())
        })
        .unwrap();

        // After commit, every key is visible from the modern root.
        assert_eq!(tree.find(b"a").unwrap().as_deref(), Some(b"1".as_slice()));
        for i in 0..TX_KEYS {
            let k = format!("k{i:05}");
            let v = format!("v{i:05}");
            assert_eq!(
                tree.find(k.as_bytes()).unwrap().as_deref(),
                Some(v.as_bytes()),
                "post-commit find of k{i:05} missed"
            );
        }
        // root_block changed exactly once (the tx commits as one atomic
        // root swap, even though many splits happened inside it).
        assert_ne!(tree.root_block, pre_root);
        tree.verify();
    }

    #[test]
    fn transaction_aborts_leave_root_unchanged() {
        let mut tree = Btree::new();
        tree.insert(b"a", b"1").unwrap();
        let pre_root = tree.root_block;

        let result: Result<()> = tree.transaction(|tx| {
            tx.insert(b"b", ROOT_SNAP, b"2")?;
            tx.insert(b"c", ROOT_SNAP, b"3")?;
            // Simulate a failure: bail out with an error.
            Err(Error::BlockNotFound(u64::MAX))
        });
        assert!(result.is_err());

        // The btree's view is the pre-tx state.
        assert_eq!(tree.root_block, pre_root);
        assert_eq!(tree.find(b"a").unwrap().as_deref(), Some(b"1".as_slice()));
        assert_eq!(tree.find(b"b").unwrap(), None);
        assert_eq!(tree.find(b"c").unwrap(), None);
        tree.verify();
    }

    #[test]
    fn transaction_atomic_rename() {
        // Rename = (delete old; insert new). With transaction(), readers
        // outside the closure see either the pre state OR the post state,
        // never an intermediate where neither key exists.
        let mut tree = Btree::new();
        tree.insert(b"old", b"value").unwrap();

        tree.transaction(|tx| {
            tx.insert(b"new", ROOT_SNAP, b"value")?;
            tx.delete_at(b"old", ROOT_SNAP, &[ROOT_SNAP])?;
            Ok(())
        })
        .unwrap();

        assert_eq!(tree.find(b"old").unwrap(), None);
        assert_eq!(
            tree.find(b"new").unwrap().as_deref(),
            Some(b"value".as_slice())
        );
        tree.verify();
    }

    // ---------- Phase 9: randomized insert/delete vs BTreeMap ----------

    #[test]
    #[cfg_attr(miri, ignore = "throughput stress test; no new unsafe path")]
    fn random_insert_delete_matches_btreemap() {
        // Drive a long random sequence of inserts and deletes into both the
        // Btree and a std BTreeMap, then check that visibility matches at
        // the end. Catches any tombstone-handling regression that would
        // otherwise only surface under specific access patterns.
        use std::collections::BTreeMap;

        let mut tree = Btree::new();
        let mut model: BTreeMap<u32, u32> = BTreeMap::new();
        let chain = vec![ROOT_SNAP];
        let mut rng = Rng(0xCAFE_BABE);

        // Use a small key space so collisions force overwrites and re-inserts.
        const KEY_SPACE: u32 = 200;
        const STEPS: u32 = 2000;

        for step in 0..STEPS {
            let k = rng.next_u32() % KEY_SPACE;
            // Bias toward insert until the model is reasonably populated.
            let do_delete = step > 200 && rng.next_u32().is_multiple_of(3);
            if do_delete {
                let removed = tree.delete_at(&k.to_be_bytes(), ROOT_SNAP, &chain).unwrap();
                let model_had = model.remove(&k).is_some();
                assert_eq!(removed, model_had, "delete return at step {step}");
            } else {
                let v = rng.next_u32();
                tree.insert_at(&k.to_be_bytes(), ROOT_SNAP, &v.to_be_bytes())
                    .unwrap();
                model.insert(k, v);
            }
        }

        // End-state consistency: every key in the model must have the
        // matching value in the btree, and any key not in the model must be
        // invisible.
        for k in 0..KEY_SPACE {
            let from_tree = tree.find(&k.to_be_bytes()).unwrap();
            let from_model = model.get(&k).copied();
            match (from_tree, from_model) {
                (Some(bytes), Some(v)) => assert_eq!(bytes, v.to_be_bytes(), "key {k}"),
                (None, None) => {}
                (got, want) => panic!("key {k}: tree={got:?} model={want:?}"),
            }
        }

        // Range scan must list exactly the model's keys.
        let scan = tree
            .range_scan(&0u32.to_be_bytes(), &KEY_SPACE.to_be_bytes())
            .unwrap();
        let scan_keys: Vec<u32> = scan
            .iter()
            .map(|(k, _)| u32::from_be_bytes(k.as_slice().try_into().unwrap()))
            .collect();
        let model_keys: Vec<u32> = model.keys().copied().collect();
        assert_eq!(scan_keys, model_keys);
        tree.verify();
    }

    // ---------- Multi-bset specific tests ----------
    //
    // These directly probe the multi-bset machinery: the leaf must actually
    // contain >1 bset after enough inserts, reads must merge across bsets,
    // newer-bset writes must shadow older-bset entries, and compaction +
    // split must collapse bsets correctly.

    /// Helper: walk the tree to find the (single) leaf, returning its bset_count.
    fn root_leaf_bset_count(tree: &Btree) -> usize {
        let mut blk = tree.root_block;
        loop {
            let node = tree.store.read_node_copy(blk).unwrap();
            if node.level() == 0 {
                return node.bset_count();
            }
            blk = node.child_block(0);
        }
    }

    #[test]
    fn leaf_grows_to_multiple_bsets() {
        // Insert past BSET_SOFT_LIMIT but stay below MAX_ENTRIES so the tree
        // is still a single leaf. We expect bset_count to climb above 1.
        let mut tree = Btree::new();
        for i in 0..16u32 {
            tree.insert(&key(i), &val(i)).unwrap();
        }
        assert!(
            root_leaf_bset_count(&tree) >= 2,
            "expected leaf to span multiple bsets after 16 inserts"
        );
        // Reads still see every key.
        for i in 0..16u32 {
            assert_eq!(
                tree.find(&key(i)).unwrap().as_deref(),
                Some(val(i).as_slice()),
                "key {i} not found",
            );
        }
        tree.verify();
    }

    #[test]
    fn newer_bset_shadows_older_bset_on_overwrite() {
        // Force an overwrite to land in a newer bset (not in-place in the
        // older one). The merged read must return the newer value.
        let mut tree = Btree::new();
        // Fill bset 0 above BSET_SOFT_LIMIT so the next overwrite goes to a
        // fresh bset 1.
        for i in 0..(BSET_SOFT_LIMIT as u32 + 2) {
            tree.insert(&key(i), &val(i)).unwrap();
        }
        // Bset 0 is now full and bset 1 has at least one entry; any further
        // insert that doesn't already live in the latest bset opens / lands
        // in a new bset.
        let new_value = 0xDEADBEEFu32.to_be_bytes();
        tree.insert(&key(0), &new_value).unwrap();
        assert!(root_leaf_bset_count(&tree) >= 2);
        assert_eq!(
            tree.find(&key(0)).unwrap().as_deref(),
            Some(new_value.as_slice()),
            "newer-bset write must shadow older-bset entry",
        );
        // Other keys unaffected.
        for i in 1..(BSET_SOFT_LIMIT as u32 + 2) {
            assert_eq!(
                tree.find(&key(i)).unwrap().as_deref(),
                Some(val(i).as_slice())
            );
        }
        tree.verify();
    }

    #[test]
    fn compaction_collapses_bsets_when_full() {
        // Push the leaf past the BSET_TREE_NR_MAX threshold; compaction must
        // fold all bsets back into one.
        let mut tree = Btree::new();
        // Each insert at distinct key grows the latest bset; once it crosses
        // BSET_SOFT_LIMIT, a new bset opens. After enough inserts, all four
        // bsets fill and compact triggers.
        let n = (BSET_SOFT_LIMIT * BSET_TREE_NR_MAX) as u32 + 4;
        for i in 0..n {
            tree.insert(&key(i), &val(i)).unwrap();
        }
        // Compaction has bounded bset_count to BSET_TREE_NR_MAX. Just check
        // the cap holds and reads still work.
        let bcnt = root_leaf_bset_count(&tree);
        assert!(
            bcnt <= BSET_TREE_NR_MAX,
            "bset_count {bcnt} exceeds BSET_TREE_NR_MAX {BSET_TREE_NR_MAX}"
        );
        for i in 0..n {
            assert_eq!(
                tree.find(&key(i)).unwrap().as_deref(),
                Some(val(i).as_slice())
            );
        }
        tree.verify();
    }

    #[test]
    fn split_handles_multi_bset_source() {
        // Drive the leaf to MAX_ENTRIES via inserts that span multiple bsets,
        // then push one more to force a split. The split must compact first
        // so the resulting halves are well-formed single-bset leaves.
        let mut tree = Btree::new();
        for i in 0..MAX_ENTRIES as u32 {
            tree.insert(&key(i), &val(i)).unwrap();
        }
        // Now the leaf has MAX_ENTRIES entries; the next distinct-key insert
        // forces a split.
        let trigger = MAX_ENTRIES as u32;
        tree.insert(&key(trigger), &val(trigger)).unwrap();
        // Tree is now level >= 1; root is internal.
        let root = tree.store.read_node_copy(tree.root_block).unwrap();
        assert!(
            root.level() >= 1,
            "root should be internal after split, level={}",
            root.level()
        );
        // All keys still readable.
        for i in 0..=trigger {
            assert_eq!(
                tree.find(&key(i)).unwrap().as_deref(),
                Some(val(i).as_slice())
            );
        }
        tree.verify();
    }

    #[test]
    fn delete_after_fill_keeps_key_invisible() {
        // Write a key, fill the bset to roll over to a new bset, then delete
        // the key. With the in-place optimization the tombstone now lives
        // *inside* the original bset (kind flipped from Live to Deleted)
        // rather than as a fresh entry in the latest bset, but the
        // observable behavior is identical: find returns None.
        let mut tree = Btree::new();
        for i in 0..(BSET_SOFT_LIMIT as u32 + 2) {
            tree.insert(&key(i), &val(i)).unwrap();
        }
        // Delete key(0). delete_at writes a Deleted tombstone (same-snap).
        let removed = tree.delete(&key(0)).unwrap();
        assert!(removed, "delete should report a tombstone written");
        assert_eq!(tree.find(&key(0)).unwrap(), None, "tombstone must shadow");
        // Other keys still present.
        for i in 1..(BSET_SOFT_LIMIT as u32 + 2) {
            assert_eq!(
                tree.find(&key(i)).unwrap().as_deref(),
                Some(val(i).as_slice())
            );
        }
        tree.verify();
    }

    #[test]
    fn inplace_delete_does_not_grow_node() {
        // Delete-of-own-snap-key flips the existing entry's kind in place
        // rather than sort-inserting a fresh tombstone. nkeys and
        // bset_count must therefore be unchanged across the delete.
        let mut tree = Btree::new();
        for i in 0..(BSET_SOFT_LIMIT as u32 + 2) {
            tree.insert(&key(i), &val(i)).unwrap();
        }
        let nkeys_before = {
            let blk = tree.root_block;
            tree.store.read_node_copy(blk).unwrap().nkeys()
        };
        let bcnt_before = root_leaf_bset_count(&tree);
        // key(0) lives in bset 0 (older); the in-place flip should target
        // that bset directly without touching the latest bset.
        assert!(tree.delete(&key(0)).unwrap());
        let nkeys_after = {
            let blk = tree.root_block;
            tree.store.read_node_copy(blk).unwrap().nkeys()
        };
        let bcnt_after = root_leaf_bset_count(&tree);
        assert_eq!(
            nkeys_before, nkeys_after,
            "in-place delete must not grow nkeys ({nkeys_before} -> {nkeys_after})"
        );
        assert_eq!(
            bcnt_before, bcnt_after,
            "in-place delete must not open a new bset ({bcnt_before} -> {bcnt_after})"
        );
        assert_eq!(tree.find(&key(0)).unwrap(), None);
        tree.verify();
    }

    #[test]
    fn inplace_delete_works_across_multiple_bsets() {
        // Spread keys across all four bsets, then delete a key that lives
        // in bset 0. The flip must reach the right entry in the right bset
        // and survive the merged read path.
        let mut tree = Btree::new();
        // Pack each bset to soft-limit so we span 4 bsets without splitting.
        let n = (BSET_SOFT_LIMIT * BSET_TREE_NR_MAX) as u32;
        for i in 0..n {
            tree.insert(&key(i), &val(i)).unwrap();
        }
        let bcnt_before = root_leaf_bset_count(&tree);
        assert!(
            bcnt_before >= 2,
            "test setup needs multiple bsets, got {bcnt_before}"
        );
        // Delete one key from each quartile to exercise different bsets.
        for i in [0u32, n / 4, n / 2, 3 * n / 4] {
            assert!(tree.delete(&key(i)).unwrap(), "delete key {i} failed");
        }
        // Deleted keys are gone; surviving keys still readable.
        for i in 0..n {
            let want_present = ![0u32, n / 4, n / 2, 3 * n / 4].contains(&i);
            let got = tree.find(&key(i)).unwrap();
            if want_present {
                assert_eq!(got.as_deref(), Some(val(i).as_slice()), "key {i} lost");
            } else {
                assert_eq!(got, None, "key {i} should be deleted");
            }
        }
        tree.verify();
    }

    #[test]
    fn merged_iter_respects_seq_for_dup_keys() {
        // Build a leaf where the same key lives in two bsets (older has
        // value A, newer has value B). MergedIter must yield exactly one
        // entry — the one from the newer bset.
        use crate::block_btree::{BtreeNodeRaw, MergedIter};

        let mut node = BtreeNodeRaw::new(0);
        // bset 0: seq=10, contains (key1, valA).
        node.start_new_bset(10);
        let mut e_a = DiskEntry::empty();
        e_a.set_key_with_snap(b"k1", ROOT_SNAP);
        e_a.set_kind(EntryKind::Live);
        e_a.payload.value_mut()[..1].copy_from_slice(b"A");
        e_a.value_len = 1;
        node.append_to_last_bset(&e_a);
        // bset 1: seq=20, contains (key1, valB).
        node.start_new_bset(20);
        let mut e_b = e_a;
        e_b.payload.value_mut()[..1].copy_from_slice(b"B");
        node.append_to_last_bset(&e_b);

        let mut iter = MergedIter::new(&node);
        let (b, i) = iter.next().expect("at least one entry");
        assert_eq!(b, 1, "newer bset must win");
        let val_bytes = &node.entry_at(b, i).payload.value()[..1];
        assert_eq!(val_bytes, b"B");
        assert!(
            iter.next().is_none(),
            "older shadowed entry must not be emitted"
        );
    }

    /// Reachability check: drive a workload that should pack four bsets to
    /// the soft limit and verify the compaction path in
    /// `ensure_writable_last_bset` actually fires. If a future constant
    /// change pushes `BSET_SOFT_LIMIT * BSET_TREE_NR_MAX` past `MAX_ENTRIES`
    /// the path silently becomes unreachable (split fires first) — this
    /// test catches that regression.
    #[test]
    fn compaction_path_is_reachable() {
        use std::sync::atomic::Ordering;

        let baseline = COMPACT_ON_FULL_HITS.load(Ordering::Relaxed);
        let mut tree = Btree::new();
        let mut rng = Rng(0xC0FF_EEFE_EDC0_FFEE);
        // Mix: small key space + frequent overwrites packs lots of writes
        // into a single leaf without forcing a structural split, exercising
        // the multi-bset accumulator until 4 bsets fill.
        for _ in 0..2000u32 {
            let k = rng.next_u32() % 50;
            if rng.next_u32().is_multiple_of(4) {
                let _ = tree.delete(&key(k));
            } else {
                tree.insert(&key(k), &val(rng.next_u32())).unwrap();
            }
        }
        let hits = COMPACT_ON_FULL_HITS.load(Ordering::Relaxed) - baseline;
        assert!(
            hits > 0,
            "ensure_writable_last_bset never compacted; the 4-bsets-full path \
             is unreachable (likely BSET_SOFT_LIMIT * BSET_TREE_NR_MAX >= MAX_ENTRIES)",
        );
    }
}
