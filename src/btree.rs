use std::collections::HashMap;

use crate::block_btree::{
    BtreeNodeRaw, EntryKind, MAGIC_NUMBER, MAX_ENTRIES, MAX_INTERNAL_KEYS, MAX_KEY_SIZE,
    MAX_LOGICAL_KEY_SIZE, ROOT_SNAP, SNAP_ID_BYTES, SnapId,
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
    /// On a real block device this would cover both I/O failure and
    /// an inconsistent tree pointing at an unallocated block.
    BlockNotFound(u64),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::BlockNotFound(nr) => write!(f, "block {nr} not found"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// COW B-tree: all mutations produce new nodes; old nodes remain reachable
/// via their block numbers (key for crash recovery and snapshots).
pub struct Btree {
    pub root_block: u64,
    block_map: HashMap<u64, BtreeNodeRaw>,
    next_block_nr: u64,
}

impl Btree {
    pub fn new() -> Self {
        let root_block = 0;
        let mut block_map = HashMap::new();
        block_map.insert(root_block, BtreeNodeRaw::new(0));
        Btree {
            root_block,
            block_map,
            next_block_nr: 1,
        }
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
        find(&self.block_map, self.root_block, &buf[..len])
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
            &mut self.block_map,
            &mut self.next_block_nr,
            self.root_block,
            &buf[..len],
            value,
            kind,
        )?;
        self.root_block = new_root;
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
        let kind = if visible_snap == snap {
            // X deletes a key X itself wrote. The matching live entry will
            // be overwritten in place (same sortable key) — a Deleted
            // tombstone here only needs to last until the next compaction.
            EntryKind::Deleted
        } else {
            // Inherited from an ancestor. Must shadow the ancestor's
            // still-live entry until the relevant snapshots are deleted.
            EntryKind::Whiteout
        };
        self.insert_with_kind(key, snap, kind, &[])?;
        Ok(true)
    }

    /// Like `find_visible` but also returns the snap_id at which the visible
    /// entry was stored. Used by `delete_at` to pick Deleted vs Whiteout.
    pub fn find_visible_with_snap(
        &self,
        logical: &[u8],
        chain: &[SnapId],
    ) -> Result<Option<(Vec<u8>, SnapId)>> {
        find_visible_with_snap(&self.block_map, self.root_block, logical, chain)
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
            &self.block_map,
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
        find_visible(&self.block_map, self.root_block, logical, chain)
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
            &self.block_map,
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
        println!("=== B-tree (next_block_nr={}) ===", self.next_block_nr);
        dump(&self.block_map, self.root_block, 0);
        println!("=== end ===");
    }

    /// Walk the entire tree and panic if any invariant is violated.
    #[cfg(test)]
    pub fn verify(&self) {
        verify_node(&self.block_map, self.root_block, None, None);
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
// New blocks allocated during the transaction remain in `block_map` as
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
            &mut self.btree.block_map,
            &mut self.btree.next_block_nr,
            self.pending_root,
            &buf[..len],
            value,
            kind,
        )?;
        self.pending_root = new_root;
        Ok(())
    }

    pub fn delete_at(&mut self, key: &[u8], snap: SnapId, chain: &[SnapId]) -> Result<bool> {
        let Some((_value, visible_snap)) =
            find_visible_with_snap(&self.btree.block_map, self.pending_root, key, chain)?
        else {
            return Ok(false);
        };
        let kind = if visible_snap == snap {
            EntryKind::Deleted
        } else {
            EntryKind::Whiteout
        };
        self.insert_with_kind(key, snap, kind, &[])?;
        Ok(true)
    }

    pub fn find_at(&self, key: &[u8], snap: SnapId) -> Result<Option<Vec<u8>>> {
        let (buf, len) = sortable_key(key, snap);
        find(&self.btree.block_map, self.pending_root, &buf[..len])
    }

    pub fn find_visible(&self, key: &[u8], chain: &[SnapId]) -> Result<Option<Vec<u8>>> {
        find_visible(&self.btree.block_map, self.pending_root, key, chain)
    }

    pub fn find_visible_with_snap(
        &self,
        key: &[u8],
        chain: &[SnapId],
    ) -> Result<Option<(Vec<u8>, SnapId)>> {
        find_visible_with_snap(&self.btree.block_map, self.pending_root, key, chain)
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
            Err(e) => Err(e),
        }
    }
}

impl Default for Btree {
    fn default() -> Self {
        Btree::new()
    }
}

/// Count total keys in the subtree rooted at `block_nr`.
#[cfg(test)]
fn count_keys(block_map: &HashMap<u64, BtreeNodeRaw>, block_nr: u64) -> usize {
    let node = &block_map[&block_nr];
    let n = node.nkeys();
    if node.level() == 0 {
        n
    } else {
        // Internal keys are separators — they duplicate keys that also exist
        // in the subtrees (separator-in-right convention). Only count leaves.
        let mut total = 0;
        for i in 0..=n {
            total += count_keys(block_map, node.child_block(i));
        }
        total
    }
}

/// Collect the depth of every leaf node. All leaves must be at the same depth
/// for the tree to be balanced.
#[cfg(test)]
fn collect_leaf_depths(
    block_map: &HashMap<u64, BtreeNodeRaw>,
    block_nr: u64,
    depth: usize,
    depths: &mut Vec<usize>,
) {
    let node = &block_map[&block_nr];
    if node.level() == 0 {
        depths.push(depth);
    } else {
        for i in 0..=node.nkeys() {
            collect_leaf_depths(block_map, node.child_block(i), depth + 1, depths);
        }
    }
}

/// Recursive invariant check. `lo`/`hi` are exclusive bounds inherited from
/// the parent separator (separator-in-right convention: the separator belongs
/// to the right subtree, so child[i] holds keys in [lo, hi)).
#[cfg(test)]
fn verify_node(
    block_map: &HashMap<u64, BtreeNodeRaw>,
    block_nr: u64,
    lo: Option<&[u8]>,
    hi: Option<&[u8]>,
) {
    let node = &block_map[&block_nr];
    let n = node.nkeys();

    if node.level() == 0 {
        debug_assert!(n <= MAX_ENTRIES, "leaf blk={block_nr} nkeys={n}");
    } else {
        debug_assert!(n <= MAX_INTERNAL_KEYS, "internal blk={block_nr} nkeys={n}");
    }

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
        if let Some(lo) = lo {
            if k < lo {
                panic!(
                    "blk={block_nr} level={} nkeys={n} key[{i}]={k:?} < lo={lo:?}\n\
                     first_key={:?} last_key={:?}",
                    node.level(),
                    node.entry(0).key_bytes(),
                    node.entry(n - 1).key_bytes(),
                );
            }
        }
        if let Some(hi) = hi {
            debug_assert!(k < hi, "blk={block_nr} key[{i}]={k:?} >= hi={hi:?}");
        }
    }

    if node.level() > 0 {
        for i in 0..=n {
            let child_nr = node.child_block(i);
            debug_assert!(
                block_map.contains_key(&child_nr),
                "blk={block_nr} child[{i}]={child_nr} not in block_map"
            );
            let child = &block_map[&child_nr];
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
            verify_node(block_map, child_nr, child_lo, child_hi);
        }
    }
}

/// Heap-allocate a zeroed node (avoids 4KB stack temporaries).
fn new_node_on_heap(level: u8) -> Box<BtreeNodeRaw> {
    // SAFETY: repr(C) + FromBytes — zeroed memory is valid; we set header fields below.
    let mut b: Box<BtreeNodeRaw> = unsafe { Box::<BtreeNodeRaw>::new_zeroed().assume_init() };
    b.header.magic = MAGIC_NUMBER;
    b.header.level = level;
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

/// Copy a single leaf entry (key + snap_id + kind + value) from `src[src_idx]`
/// to `dst[dst_idx]`. Both nodes must be leaves. The full sortable key bytes
/// (including snap_id) and the entry kind are preserved.
fn copy_leaf_entry(dst: &mut BtreeNodeRaw, dst_idx: usize, src: &BtreeNodeRaw, src_idx: usize) {
    debug_assert_eq!(dst.level(), 0);
    debug_assert_eq!(src.level(), 0);
    let src_entry = src.entry(src_idx);
    dst.entry_mut(dst_idx).set_key(src_entry.key_bytes());
    dst.entry_mut(dst_idx).set_kind(src_entry.kind_enum());
    let val = src.value_bytes(src_idx).to_vec();
    dst.set_value(dst_idx, &val);
}

// ---------- Recursive operations ----------

fn read_block(block_map: &HashMap<u64, BtreeNodeRaw>, block_nr: u64) -> Result<&BtreeNodeRaw> {
    block_map
        .get(&block_nr)
        .ok_or(Error::BlockNotFound(block_nr))
}

fn find(
    block_map: &HashMap<u64, BtreeNodeRaw>,
    block_nr: u64,
    key: &[u8],
) -> Result<Option<Vec<u8>>> {
    Ok(
        find_raw(block_map, block_nr, key)?.and_then(|(value, kind)| match kind {
            EntryKind::Live => Some(value),
            EntryKind::Deleted | EntryKind::Whiteout => None,
        }),
    )
}

/// Like `find` but returns the entry's value AND its kind tag, so callers can
/// distinguish "no entry" from "tombstone". Used by `find_visible` to decide
/// whether to keep walking the ancestor chain or stop.
fn find_raw(
    block_map: &HashMap<u64, BtreeNodeRaw>,
    block_nr: u64,
    key: &[u8],
) -> Result<Option<(Vec<u8>, EntryKind)>> {
    let node = read_block(block_map, block_nr)?;
    match node.search(key) {
        Ok(idx) => {
            if node.level() == 0 {
                let entry = node.entry(idx);
                Ok(Some((node.value_bytes(idx).to_vec(), entry.kind_enum())))
            } else {
                find_raw(block_map, node.child_block(idx + 1), key)
            }
        }
        Err(idx) => {
            if node.level() == 0 {
                Ok(None)
            } else {
                find_raw(block_map, node.child_block(idx), key)
            }
        }
    }
}

/// Walk the snapshot ancestor chain and return the value visible at the
/// chain's head, or None on tombstone / no entry. Used by both
/// `Btree::find_visible` and `Tx::find_visible`.
fn find_visible(
    block_map: &HashMap<u64, BtreeNodeRaw>,
    block_nr: u64,
    logical: &[u8],
    chain: &[SnapId],
) -> Result<Option<Vec<u8>>> {
    Ok(find_visible_with_snap(block_map, block_nr, logical, chain)?.map(|(v, _)| v))
}

fn find_visible_with_snap(
    block_map: &HashMap<u64, BtreeNodeRaw>,
    block_nr: u64,
    logical: &[u8],
    chain: &[SnapId],
) -> Result<Option<(Vec<u8>, SnapId)>> {
    for &snap in chain {
        let (buf, len) = sortable_key(logical, snap);
        match find_raw(block_map, block_nr, &buf[..len])? {
            Some((value, EntryKind::Live)) => return Ok(Some((value, snap))),
            Some((_, EntryKind::Deleted | EntryKind::Whiteout)) => return Ok(None),
            None => continue,
        }
    }
    Ok(None)
}

/// COW insert — returns the block number of the (possibly new) root.
/// `kind` is the entry kind (Live for normal inserts, Deleted/Whiteout for
/// tombstones). On a found-key path the existing entry's kind is replaced.
fn insert(
    block_map: &mut HashMap<u64, BtreeNodeRaw>,
    next_block_nr: &mut u64,
    block_nr: u64,
    key: &[u8],
    value: &[u8],
    kind: EntryKind,
) -> Result<u64> {
    // COW: clone before mutating so the original block stays intact.
    let old_node = clone_to_heap(read_block(block_map, block_nr)?);
    if old_node.level() == 0 {
        insert_leaf(block_map, next_block_nr, &old_node, key, value, kind)
    } else {
        insert_internal(block_map, next_block_nr, &old_node, key, value, kind)
    }
}

fn insert_leaf(
    block_map: &mut HashMap<u64, BtreeNodeRaw>,
    next_block_nr: &mut u64,
    old_node: &BtreeNodeRaw,
    key: &[u8],
    value: &[u8],
    kind: EntryKind,
) -> Result<u64> {
    // Leaf is full — split first, then insert into the correct child.
    // This may cascade if the child also splits.
    if old_node.nkeys() >= MAX_ENTRIES {
        let (new_root_block, left_block, right_block) =
            split_leaf_node(block_map, next_block_nr, old_node);
        block_map
            .get_mut(&new_root_block)
            .unwrap()
            .set_child_block(0, left_block);
        block_map
            .get_mut(&new_root_block)
            .unwrap()
            .set_child_block(1, right_block);

        let (child_idx, child_nr) = {
            let root = read_block(block_map, new_root_block)?;
            let child_idx = match root.search(key) {
                Ok(i) => i + 1,
                Err(i) => i,
            };
            (child_idx, root.child_block(child_idx))
        };
        let child_level = read_block(block_map, child_nr)?.level();
        let new_child_nr = insert(block_map, next_block_nr, child_nr, key, value, kind)?;

        let new_child_level = read_block(block_map, new_child_nr)?.level();
        if new_child_level > child_level {
            // Child itself split — promote its median to the new root.
            let (median_key, left, right) = {
                let new_child = read_block(block_map, new_child_nr)?;
                (
                    new_child.entry(0).key_bytes().to_vec(),
                    new_child.child_block(0),
                    new_child.child_block(1),
                )
            };
            let parent = clone_to_heap(read_block(block_map, new_root_block)?);
            return promote_to_parent(
                block_map,
                next_block_nr,
                &parent,
                child_idx,
                &median_key,
                left,
                right,
            );
        }

        block_map
            .get_mut(&new_root_block)
            .unwrap()
            .set_child_block(child_idx, new_child_nr);
        return Ok(new_root_block);
    }

    match old_node.search(key) {
        Ok(idx) => {
            // Key exists — COW clone and patch the value + kind in place.
            // Note: when overwriting a tombstone with a Live insert, kind
            // flips back to Live; when delete writes a tombstone, kind
            // flips to Deleted/Whiteout.
            let new_block = *next_block_nr;
            *next_block_nr += 1;
            let mut new_node = clone_to_heap(old_node);
            new_node.set_value(idx, value);
            new_node.entry_mut(idx).set_kind(kind);
            block_map.insert(new_block, *new_node);
            Ok(new_block)
        }
        Err(idx) => {
            // New key — build a fresh leaf with the entry inserted at idx.
            // Existing entries are copied with their kind preserved (so a
            // leaf carrying tombstones survives a neighbor's insert).
            let new_block = *next_block_nr;
            *next_block_nr += 1;
            let mut new_node = new_node_on_heap(0);
            new_node.set_generation(old_node.generation());
            for i in 0..idx {
                copy_leaf_entry(&mut new_node, i, old_node, i);
            }
            new_node.entry_mut(idx).set_key(key);
            new_node.set_value(idx, value);
            new_node.entry_mut(idx).set_kind(kind);
            for i in idx..old_node.nkeys() {
                copy_leaf_entry(&mut new_node, i + 1, old_node, i);
            }
            new_node.set_nkeys(old_node.nkeys() + 1);
            block_map.insert(new_block, *new_node);
            Ok(new_block)
        }
    }
}

fn insert_internal(
    block_map: &mut HashMap<u64, BtreeNodeRaw>,
    next_block_nr: &mut u64,
    old_node: &BtreeNodeRaw,
    key: &[u8],
    value: &[u8],
    kind: EntryKind,
) -> Result<u64> {
    let child_idx = match old_node.search(key) {
        Ok(i) => i + 1,
        Err(i) => i,
    };
    let child_nr = old_node.child_block(child_idx);
    let child_level = read_block(block_map, child_nr)?.level();

    let new_child_nr = insert(block_map, next_block_nr, child_nr, key, value, kind)?;

    // Child split and grew a level — promote its median key to this level.
    let new_child_level = read_block(block_map, new_child_nr)?.level();
    if new_child_level > child_level {
        let new_child = read_block(block_map, new_child_nr)?;
        let median_key = new_child.entry(0).key_bytes().to_vec();
        let left = new_child.child_block(0);
        let right = new_child.child_block(1);
        // promote_to_parent sets all child pointers internally; we must NOT
        // touch the returned block afterward — child_idx is only valid against
        // the old parent, not a potential new root from a cascading split.
        return promote_to_parent(
            block_map,
            next_block_nr,
            old_node,
            child_idx,
            &median_key,
            left,
            right,
        );
    }

    let new_block = *next_block_nr;
    *next_block_nr += 1;
    let mut new_node = clone_to_heap(old_node);
    new_node.set_child_block(child_idx, new_child_nr);
    block_map.insert(new_block, *new_node);
    Ok(new_block)
}

// ---------- Split & promote ----------

/// Split a full leaf into two, creating a new level-1 root.
/// Returns (new_root_block, left_block, right_block).
/// Caller must set child pointers on the new root.
fn split_leaf_node(
    block_map: &mut HashMap<u64, BtreeNodeRaw>,
    next_block_nr: &mut u64,
    node: &BtreeNodeRaw,
) -> (u64, u64, u64) {
    let n = node.nkeys();
    debug_assert!(n >= MAX_ENTRIES);
    debug_assert!(node.level() == 0);
    let mid = n / 2;
    debug_assert!(mid > 0 && mid < n);
    let median_key = node.entry(mid).key_bytes();

    let left_block = *next_block_nr;
    *next_block_nr += 1;
    let mut left = clone_to_heap(node);
    left.set_nkeys(mid);
    block_map.insert(left_block, *left);

    let right_block = *next_block_nr;
    *next_block_nr += 1;
    let mut right = clone_to_heap(node);
    right.entries.copy_within(mid..n, 0);
    right.set_nkeys(n - mid);
    block_map.insert(right_block, *right);

    let root_block = *next_block_nr;
    *next_block_nr += 1;
    let mut root = new_node_on_heap(1);
    root.set_generation(node.generation());
    root.entry_mut(0).set_key(median_key);
    root.set_nkeys(1);
    block_map.insert(root_block, *root);

    (root_block, left_block, right_block)
}

/// Split a full internal node into two, creating a new root at level+1.
/// Returns (new_root_block, left_block, right_block).
/// Caller must set child pointers on the new root.
fn split_internal_node(
    block_map: &mut HashMap<u64, BtreeNodeRaw>,
    next_block_nr: &mut u64,
    node: &BtreeNodeRaw,
) -> (u64, u64, u64) {
    let n = node.nkeys();
    debug_assert!(n >= MAX_INTERNAL_KEYS);
    debug_assert!(node.level() > 0);
    let mid = n / 2;
    debug_assert!(mid > 0 && mid < n);
    let median_key = node.entry(mid).key_bytes();

    let left_block = *next_block_nr;
    *next_block_nr += 1;
    let mut left = clone_to_heap(node);
    left.set_nkeys(mid);
    block_map.insert(left_block, *left);

    // Separator-in-right: key[mid] goes to the parent only.
    // Right child gets key[mid+1..n] with children c_{mid+1}..c_n.
    let right_block = *next_block_nr;
    *next_block_nr += 1;
    let mut right = new_node_on_heap(node.level());
    right.set_generation(node.generation());
    for i in (mid + 1)..n {
        right
            .entry_mut(i - mid - 1)
            .set_key(node.entry(i).key_bytes());
    }
    right.set_nkeys(n - mid - 1);
    for i in 0..=(n - mid - 1) {
        right.set_child_block(i, node.child_block(mid + 1 + i));
    }
    block_map.insert(right_block, *right);

    let root_block = *next_block_nr;
    *next_block_nr += 1;
    let mut root = new_node_on_heap(node.level() + 1);
    root.set_generation(node.generation());
    root.entry_mut(0).set_key(median_key);
    root.set_nkeys(1);
    block_map.insert(root_block, *root);

    (root_block, left_block, right_block)
}

/// Insert a split result (median_key + left/right children) into the parent
/// at child_idx.  Returns the new parent block number (may cascade-split).
fn promote_to_parent(
    block_map: &mut HashMap<u64, BtreeNodeRaw>,
    next_block_nr: &mut u64,
    old_parent: &BtreeNodeRaw,
    child_idx: usize,
    median_key: &[u8],
    left_child: u64,
    right_child: u64,
) -> Result<u64> {
    debug_assert!(old_parent.level() > 0);
    debug_assert!(child_idx <= old_parent.nkeys());

    let old_nkeys = old_parent.nkeys();
    let new_block = *next_block_nr;
    *next_block_nr += 1;
    let mut new_node = new_node_on_heap(old_parent.level());
    new_node.set_generation(old_parent.generation());

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
    new_node.set_nkeys(old_nkeys + 1);

    if new_node.nkeys() <= MAX_INTERNAL_KEYS {
        block_map.insert(new_block, *new_node);
        Ok(new_block)
    } else {
        let (new_root, new_left, new_right) =
            split_internal_node(block_map, next_block_nr, &new_node);
        block_map
            .get_mut(&new_root)
            .unwrap()
            .set_child_block(0, new_left);
        block_map
            .get_mut(&new_root)
            .unwrap()
            .set_child_block(1, new_right);
        Ok(new_root)
    }
}

// ---------- Scan & debug ----------

fn range_scan(
    block_map: &HashMap<u64, BtreeNodeRaw>,
    block_nr: u64,
    start: &[u8],
    end: &[u8],
    results: &mut Vec<(Vec<u8>, Vec<u8>)>,
) -> Result<()> {
    let node = read_block(block_map, block_nr)?;
    if node.level() == 0 {
        for i in 0..node.nkeys() {
            let entry = node.entry(i);
            let sk = entry.key_bytes();
            if sk >= start && sk < end {
                if entry.kind_enum() != EntryKind::Live {
                    // Future-proofing for Phase 4+: tombstones are not visible
                    // to range_scan callers.
                    continue;
                }
                // Strip the snap_id suffix; callers see logical keys only.
                results.push((
                    entry.logical_key_bytes().to_vec(),
                    node.value_bytes(i).to_vec(),
                ));
            }
        }
    } else {
        let mut i = match node.search(start) {
            Ok(idx) => idx + 1,
            Err(idx) => idx,
        };
        let nchildren = node.nkeys() + 1;
        if i >= nchildren {
            i = nchildren - 1;
        }
        while i < nchildren {
            range_scan(block_map, node.child_block(i), start, end, results)?;
            // Separator-in-right: once a separator >= end, all further
            // subtrees are out of range.
            if i < node.nkeys() && node.entry(i).key_bytes() >= end {
                return Ok(());
            }
            i += 1;
        }
    }
    Ok(())
}

/// Like `range_scan` but emits *every* entry (including tombstones) with
/// full metadata: `(logical_key, snap_id, kind, value)`. Backs the public
/// `Btree::range_scan_all` (BTREE_ITER_ALL_SNAPSHOTS in bcachefs).
fn range_scan_all(
    block_map: &HashMap<u64, BtreeNodeRaw>,
    block_nr: u64,
    start: &[u8],
    end: &[u8],
    results: &mut Vec<AllSnapRow>,
) -> Result<()> {
    let node = read_block(block_map, block_nr)?;
    if node.level() == 0 {
        for i in 0..node.nkeys() {
            let entry = node.entry(i);
            let sk = entry.key_bytes();
            if sk >= start && sk < end {
                results.push((
                    entry.logical_key_bytes().to_vec(),
                    entry.snap_id(),
                    entry.kind_enum(),
                    node.value_bytes(i).to_vec(),
                ));
            }
        }
    } else {
        let mut i = match node.search(start) {
            Ok(idx) => idx + 1,
            Err(idx) => idx,
        };
        let nchildren = node.nkeys() + 1;
        if i >= nchildren {
            i = nchildren - 1;
        }
        while i < nchildren {
            range_scan_all(block_map, node.child_block(i), start, end, results)?;
            if i < node.nkeys() && node.entry(i).key_bytes() >= end {
                return Ok(());
            }
            i += 1;
        }
    }
    Ok(())
}

fn dump(block_map: &HashMap<u64, BtreeNodeRaw>, block_nr: u64, indent: usize) {
    let node = &block_map[&block_nr];
    let prefix = "  ".repeat(indent);
    if node.level() == 0 {
        print!("{prefix}[leaf blk={block_nr} keys=");
        for i in 0..node.nkeys() {
            print!(" {:02x?}", node.entry(i).key_bytes());
        }
        println!("]");
    } else {
        println!(
            "{prefix}[internal blk={block_nr} level={} keys={}]",
            node.level(),
            node.nkeys()
        );
        for i in 0..=node.nkeys() {
            dump(block_map, node.child_block(i), indent + 1);
            if i < node.nkeys() {
                println!("{prefix}  -- {:02x?} --", node.entry(i).key_bytes());
            }
        }
    }
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    fn key(i: u32) -> [u8; 4] {
        i.to_be_bytes()
    }

    fn val(i: u32) -> [u8; 4] {
        (i + 10000).to_be_bytes()
    }

    /// Direct call to the internal `find` against a specific root block.
    /// Tests use this to look into historical (pre-overwrite) roots that the
    /// public Btree API doesn't expose. Wraps the snap_id append step.
    fn find_at_root(
        block_map: &HashMap<u64, BtreeNodeRaw>,
        root: u64,
        logical: &[u8],
    ) -> Result<Option<Vec<u8>>> {
        let (buf, len) = sortable_key(logical, ROOT_SNAP);
        find(block_map, root, &buf[..len])
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
        let mut tree = Btree::new();
        tree.insert(&key(10), &val(10)).unwrap();
        tree.insert(&key(20), &val(20)).unwrap();

        let old_root_block = tree.root_block;

        tree.insert(&key(30), &val(30)).unwrap();

        assert_eq!(
            find_at_root(&tree.block_map, old_root_block, &key(10))
                .unwrap()
                .as_deref(),
            Some(val(10).as_slice())
        );
        assert_eq!(
            find_at_root(&tree.block_map, old_root_block, &key(20))
                .unwrap()
                .as_deref(),
            Some(val(20).as_slice())
        );
        assert_eq!(
            find_at_root(&tree.block_map, old_root_block, &key(30))
                .unwrap()
                .as_deref(),
            None
        );

        assert_eq!(
            tree.find(&key(30)).unwrap().as_deref(),
            Some(val(30).as_slice())
        );
        tree.verify();
    }

    #[test]
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

        assert_eq!(
            count_keys(&tree.block_map, tree.root_block),
            reference.len()
        );
        tree.verify();
    }

    #[test]
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
        assert_eq!(
            count_keys(&tree.block_map, tree.root_block),
            reference.len()
        );
        tree.verify();
    }

    // ---------- Split propagation ----------

    #[test]
    fn test_split_propagation_forced_root_split() {
        // Insert keys in ascending order until the root splits multiple times.
        // With MAX_ENTRIES=29, filling 29 leaves forces a 3-level tree.
        let mut tree = Btree::new();
        let n = 2000u32;
        for i in 0..n {
            tree.insert(&key(i), &val(i)).unwrap();
        }

        // Tree should have grown beyond level 1.
        let root = &tree.block_map[&tree.root_block];
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
        assert_eq!(count_keys(&tree.block_map, tree.root_block), n as usize);
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
        assert_eq!(tree.block_map[&tree.root_block].level(), 0);

        // One more triggers a split.
        tree.insert(&key(max), &val(max)).unwrap();
        assert_eq!(tree.block_map[&tree.root_block].level(), 1);

        for i in 0..=max {
            assert_eq!(
                tree.find(&key(i)).unwrap().as_deref(),
                Some(val(i).as_slice())
            );
        }
        assert_eq!(
            count_keys(&tree.block_map, tree.root_block),
            (max + 1) as usize
        );
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
    fn test_cow_multiple_snapshots() {
        let mut tree = Btree::new();
        let mut snapshots = Vec::new();

        // Take a snapshot (save root_block) after every 100 inserts.
        for i in 0u32..500 {
            tree.insert(&key(i), &val(i)).unwrap();
            if i % 100 == 99 {
                snapshots.push((i, tree.root_block));
            }
        }

        // Each snapshot should reflect exactly the keys inserted up to that point.
        for &(last_key, snap_root) in &snapshots {
            for i in 0..=last_key {
                assert_eq!(
                    find_at_root(&tree.block_map, snap_root, &key(i))
                        .unwrap()
                        .as_deref(),
                    Some(val(i).as_slice()),
                    "snapshot after key {last_key}: key {i} missing"
                );
            }
            // Key just beyond the snapshot should not exist.
            assert_eq!(
                find_at_root(&tree.block_map, snap_root, &key(last_key + 1))
                    .unwrap()
                    .as_deref(),
                None,
                "snapshot after key {last_key}: key {} unexpectedly present",
                last_key + 1
            );
        }

        tree.verify();
    }

    #[test]
    fn test_cow_after_overwrite() {
        let mut tree = Btree::new();
        tree.insert(&key(1), b"v1").unwrap();
        tree.insert(&key(2), b"v2").unwrap();
        let snap = tree.root_block;

        // Overwrite key(1) — old snapshot should still see "v1".
        tree.insert(&key(1), b"v1-new").unwrap();
        assert_eq!(
            tree.find(&key(1)).unwrap().as_deref(),
            Some(b"v1-new".as_slice())
        );
        assert_eq!(
            find_at_root(&tree.block_map, snap, &key(1))
                .unwrap()
                .as_deref(),
            Some(b"v1".as_slice())
        );
        assert_eq!(
            find_at_root(&tree.block_map, snap, &key(2))
                .unwrap()
                .as_deref(),
            Some(b"v2".as_slice())
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
        assert_eq!(count_keys(&tree.block_map, tree.root_block), n as usize);

        // Root should be at least level 2 with this many keys.
        let root = &tree.block_map[&tree.root_block];
        assert!(root.level() >= 2);

        tree.verify();
    }

    #[test]
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
        assert_eq!(
            count_keys(&tree.block_map, tree.root_block),
            reference.len()
        );
        tree.verify();
    }

    // ---------- Invariant: total key count ----------

    #[test]
    fn test_key_count_after_mixed_operations() {
        let mut tree = Btree::new();
        let mut reference = HashMap::new();

        // Insert 0..1000.
        for i in 0..1000u32 {
            tree.insert(&key(i), &val(i)).unwrap();
            reference.insert(i, i);
        }
        assert_eq!(count_keys(&tree.block_map, tree.root_block), 1000);

        // Overwrite 500..600 — count should not change.
        for i in 500..600u32 {
            tree.insert(&key(i), &val(i + 9999)).unwrap();
            reference.insert(i, i + 9999);
        }
        assert_eq!(count_keys(&tree.block_map, tree.root_block), 1000);

        // Insert new keys 1000..2000.
        for i in 1000..2000u32 {
            tree.insert(&key(i), &val(i)).unwrap();
            reference.insert(i, i);
        }
        assert_eq!(count_keys(&tree.block_map, tree.root_block), 2000);

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
        collect_leaf_depths(&tree.block_map, tree.root_block, 0, &mut depths);
        assert!(!depths.is_empty(), "tree has no leaves");
        let first = depths[0];
        for (i, &d) in depths.iter().enumerate() {
            assert_eq!(d, first, "leaf {i} is at depth {d}, expected {first}");
        }
        first
    }

    #[test]
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
    fn test_balance_reverse_inserts() {
        let mut tree = Btree::new();
        for i in (0..5000u32).rev() {
            tree.insert(&key(i), &val(i)).unwrap();
        }
        assert_balanced(&tree);
        tree.verify();
    }

    #[test]
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
    fn test_balance_leaf_fill_ratio() {
        // Every leaf (except possibly the last split remainder) should have
        // at least MAX_ENTRIES/2 keys. This is the standard B-tree invariant.
        let mut tree = Btree::new();
        for i in 0..5000u32 {
            tree.insert(&key(i), &val(i)).unwrap();
        }

        let min_keys = crate::block_btree::MAX_ENTRIES / 2;
        check_leaf_fill(&tree.block_map, tree.root_block, min_keys);
        tree.verify();
    }

    fn check_leaf_fill(block_map: &HashMap<u64, BtreeNodeRaw>, block_nr: u64, min_keys: usize) {
        let node = &block_map[&block_nr];
        if node.level() == 0 {
            assert!(
                node.nkeys() >= min_keys,
                "leaf blk={block_nr} has only {} keys, expected >= {min_keys}",
                node.nkeys()
            );
        } else {
            for i in 0..=node.nkeys() {
                check_leaf_fill(block_map, node.child_block(i), min_keys);
            }
        }
    }

    #[test]
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
            collect_leaf_depths(&tree.block_map, snap, 0, &mut depths);
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
        let mut tree = Btree::new();
        tree.insert(b"a", b"1").unwrap();
        let pre_root = tree.root_block;

        tree.transaction(|tx| {
            tx.insert(b"b", ROOT_SNAP, b"2")?;
            tx.insert(b"c", ROOT_SNAP, b"3")?;
            // Reads inside tx see pending state.
            assert_eq!(
                tx.find_at(b"b", ROOT_SNAP)?.as_deref(),
                Some(b"2".as_slice())
            );
            // Outside the tx, the original root is still in place.
            Ok(())
        })
        .unwrap();

        // After commit, all three keys are visible.
        assert_eq!(tree.find(b"a").unwrap().as_deref(), Some(b"1".as_slice()));
        assert_eq!(tree.find(b"b").unwrap().as_deref(), Some(b"2".as_slice()));
        assert_eq!(tree.find(b"c").unwrap().as_deref(), Some(b"3".as_slice()));
        // root_block changed exactly once.
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
            let k = (rng.next_u32() % KEY_SPACE) as u32;
            // Bias toward insert until the model is reasonably populated.
            let do_delete = step > 200 && (rng.next_u32() % 3 == 0);
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
}
