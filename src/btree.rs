use std::collections::HashMap;

use crate::block_btree::{BtreeNodeRaw, MAGIC_NUMBER, MAX_ENTRIES, MAX_INTERNAL_KEYS};

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

    pub fn find(&self, key: &[u8]) -> Option<&[u8]> {
        find(&self.block_map, self.root_block, key)
    }

    pub fn insert(&mut self, key: &[u8], value: &[u8]) {
        let new_root = insert(
            &mut self.block_map,
            &mut self.next_block_nr,
            self.root_block,
            key,
            value,
        );
        self.root_block = new_root;
    }

    pub fn range_scan(&self, start: &[u8], end: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut results = Vec::new();
        range_scan(&self.block_map, self.root_block, start, end, &mut results);
        results
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

// ---------- Recursive operations ----------

fn find<'a>(
    block_map: &'a HashMap<u64, BtreeNodeRaw>,
    block_nr: u64,
    key: &[u8],
) -> Option<&'a [u8]> {
    let node = &block_map[&block_nr];
    match node.search(key) {
        Ok(idx) => {
            if node.level() == 0 {
                Some(node.value_bytes(idx))
            } else {
                // Separator-in-right: the matching key is the separator itself,
                // so the value lives in the right subtree (child idx+1).
                find(block_map, node.child_block(idx + 1), key)
            }
        }
        Err(idx) => {
            if node.level() == 0 {
                None
            } else {
                find(block_map, node.child_block(idx), key)
            }
        }
    }
}

/// COW insert — returns the block number of the (possibly new) root.
fn insert(
    block_map: &mut HashMap<u64, BtreeNodeRaw>,
    next_block_nr: &mut u64,
    block_nr: u64,
    key: &[u8],
    value: &[u8],
) -> u64 {
    // COW: clone before mutating so the original block stays intact.
    let old_node = clone_to_heap(&block_map[&block_nr]);
    if old_node.level() == 0 {
        insert_leaf(block_map, next_block_nr, &old_node, key, value)
    } else {
        insert_internal(block_map, next_block_nr, &old_node, key, value)
    }
}

fn insert_leaf(
    block_map: &mut HashMap<u64, BtreeNodeRaw>,
    next_block_nr: &mut u64,
    old_node: &BtreeNodeRaw,
    key: &[u8],
    value: &[u8],
) -> u64 {
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
            let root = &block_map[&new_root_block];
            let child_idx = match root.search(key) {
                Ok(i) => i + 1,
                Err(i) => i,
            };
            (child_idx, root.child_block(child_idx))
        };
        let child_level = block_map[&child_nr].level();
        let new_child_nr = insert(block_map, next_block_nr, child_nr, key, value);

        let new_child_level = block_map[&new_child_nr].level();
        if new_child_level > child_level {
            // Child itself split — promote its median to the new root.
            let (median_key, left, right) = {
                let new_child = &block_map[&new_child_nr];
                (
                    new_child.entry(0).key_bytes().to_vec(),
                    new_child.child_block(0),
                    new_child.child_block(1),
                )
            };
            let parent = clone_to_heap(&block_map[&new_root_block]);
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
        return new_root_block;
    }

    match old_node.search(key) {
        Ok(idx) => {
            // Key exists — COW clone and patch the value in place.
            let new_block = *next_block_nr;
            *next_block_nr += 1;
            let mut new_node = clone_to_heap(old_node);
            new_node.set_value(idx, value);
            block_map.insert(new_block, *new_node);
            new_block
        }
        Err(idx) => {
            // New key — build a fresh leaf with the entry inserted at idx.
            let new_block = *next_block_nr;
            *next_block_nr += 1;
            let mut new_node = new_node_on_heap(0);
            new_node.set_generation(old_node.generation());
            for i in 0..idx {
                let k = old_node.entry(i).key_bytes();
                let v = old_node.value_bytes(i);
                new_node.entry_mut(i).set_key(k);
                new_node.set_value(i, v);
            }
            new_node.entry_mut(idx).set_key(key);
            new_node.set_value(idx, value);
            for i in idx..old_node.nkeys() {
                let k = old_node.entry(i).key_bytes();
                let v = old_node.value_bytes(i);
                new_node.entry_mut(i + 1).set_key(&k);
                new_node.set_value(i + 1, &v);
            }
            new_node.set_nkeys(old_node.nkeys() + 1);
            block_map.insert(new_block, *new_node);
            new_block
        }
    }
}

fn insert_internal(
    block_map: &mut HashMap<u64, BtreeNodeRaw>,
    next_block_nr: &mut u64,
    old_node: &BtreeNodeRaw,
    key: &[u8],
    value: &[u8],
) -> u64 {
    let child_idx = match old_node.search(key) {
        Ok(i) => i + 1,
        Err(i) => i,
    };
    let child_nr = old_node.child_block(child_idx);
    let child_level = block_map[&child_nr].level();

    let new_child_nr = insert(block_map, next_block_nr, child_nr, key, value);

    // Child split and grew a level — promote its median key to this level.
    let new_child_level = block_map[&new_child_nr].level();
    if new_child_level > child_level {
        let new_child = &block_map[&new_child_nr];
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
    new_block
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
) -> u64 {
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
        new_block
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
        new_root
    }
}

// ---------- Scan & debug ----------

fn range_scan(
    block_map: &HashMap<u64, BtreeNodeRaw>,
    block_nr: u64,
    start: &[u8],
    end: &[u8],
    results: &mut Vec<(Vec<u8>, Vec<u8>)>,
) {
    let node = &block_map[&block_nr];
    if node.level() == 0 {
        for i in 0..node.nkeys() {
            let k = node.entry(i).key_bytes();
            if k >= start && k < end {
                results.push((k.to_vec(), node.value_bytes(i).to_vec()));
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
            range_scan(block_map, node.child_block(i), start, end, results);
            // Separator-in-right: once a separator >= end, all further
            // subtrees are out of range.
            if i < node.nkeys() && node.entry(i).key_bytes() >= end {
                return;
            }
            i += 1;
        }
    }
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

    #[test]
    fn test_single_insert_and_find() {
        let mut tree = Btree::new();
        tree.insert(&key(1), &val(1));
        assert_eq!(tree.find(&key(1)), Some(val(1).as_slice()));
        assert_eq!(tree.find(&key(99)), None);
        tree.verify();
    }

    #[test]
    fn test_multiple_inserts() {
        let mut tree = Btree::new();
        let n = 1000u32;
        for i in 0..n {
            tree.insert(&key(i), &val(i));
        }
        for i in 0..n {
            assert_eq!(
                tree.find(&key(i)),
                Some(val(i).as_slice()),
                "key {i} not found"
            );
        }
        assert_eq!(tree.find(&key(u32::MAX)), None);
        tree.verify();
    }

    #[test]
    fn test_cow_preserves_old_root() {
        let mut tree = Btree::new();
        tree.insert(&key(10), &val(10));
        tree.insert(&key(20), &val(20));

        let old_root_block = tree.root_block;

        tree.insert(&key(30), &val(30));

        assert_eq!(
            find(&tree.block_map, old_root_block, &key(10)),
            Some(val(10).as_slice())
        );
        assert_eq!(
            find(&tree.block_map, old_root_block, &key(20)),
            Some(val(20).as_slice())
        );
        assert_eq!(find(&tree.block_map, old_root_block, &key(30)), None);

        assert_eq!(tree.find(&key(30)), Some(val(30).as_slice()));
        tree.verify();
    }

    #[test]
    fn test_split_many_keys() {
        let mut tree = Btree::new();
        let n = 2000u32;
        for i in 0..n {
            tree.insert(&key(i), &val(i));
        }
        for i in 0..n {
            assert_eq!(
                tree.find(&key(i)),
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
            tree.insert(&key(i), &val(i));
        }
        tree.verify();

        let results = tree.range_scan(&key(10), &key(20));
        assert_eq!(results.len(), 10);
        for (j, (k, v)) in results.iter().enumerate() {
            assert_eq!(k.as_slice(), &key(10 + j as u32));
            assert_eq!(v.as_slice(), &val(10 + j as u32));
        }
    }

    #[test]
    fn test_overwrite() {
        let mut tree = Btree::new();
        tree.insert(&key(1), b"old");
        assert_eq!(tree.find(&key(1)), Some(b"old".as_slice()));

        tree.insert(&key(1), b"new");
        assert_eq!(tree.find(&key(1)), Some(b"new".as_slice()));
        tree.verify();
    }

    #[test]
    fn test_dump() {
        let mut tree = Btree::new();
        for i in 0u32..10 {
            tree.insert(&key(i), &val(i));
        }
        tree.dump();
        tree.verify();
    }

    #[test]
    fn test_reverse_insert() {
        let mut tree = Btree::new();
        let n = 2000u32;
        for i in (0..n).rev() {
            tree.insert(&key(i), &val(i));
        }
        for i in 0..n {
            assert_eq!(tree.find(&key(i)), Some(val(i).as_slice()));
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
        node.entry_mut(0).set_key(b"hello");
        node.entry_mut(1).set_key(b"world");
        node.entry_mut(2).set_key(b"foo");

        let bytes: &[u8] = node.as_bytes();
        assert_eq!(bytes.len(), 4096);

        let restored = BtreeNodeRaw::ref_from_bytes(bytes).unwrap();
        assert_eq!(restored.nkeys(), 3);
        assert_eq!(restored.generation(), 42);
        assert_eq!(restored.level(), 0);
        assert_eq!(restored.entry(0).key_bytes(), b"hello");
        assert_eq!(restored.entry(1).key_bytes(), b"world");
        assert_eq!(restored.entry(2).key_bytes(), b"foo");
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
            tree.insert(&key(k), &val(v));
            reference.insert(k, v);
        }

        // Verify every inserted key can be found with the correct value.
        for (&k, &v) in &reference {
            assert_eq!(
                tree.find(&key(k)),
                Some(val(v).as_slice()),
                "random key {k} not found or value mismatch"
            );
        }

        // Verify keys not in the reference return None.
        for _ in 0..100 {
            let probe = rng.next_u32();
            if !reference.contains_key(&probe) {
                assert_eq!(tree.find(&key(probe)), None);
            }
        }

        assert_eq!(count_keys(&tree.block_map, tree.root_block), reference.len());
        tree.verify();
    }

    #[test]
    fn test_random_overwrite_consistency() {
        let mut rng = Rng(0x1234_5678);
        let mut tree = Btree::new();
        let mut reference = HashMap::new();

        // Insert, then overwrite a subset.
        for i in 0..2000u32 {
            tree.insert(&key(i), &val(i));
            reference.insert(i, i);
        }
        for _ in 0..1000 {
            let k = rng.next_u32() % 2000;
            let v = rng.next_u32();
            tree.insert(&key(k), &val(v));
            reference.insert(k, v);
        }

        for (&k, &v) in &reference {
            assert_eq!(tree.find(&key(k)), Some(val(v).as_slice()));
        }
        assert_eq!(count_keys(&tree.block_map, tree.root_block), reference.len());
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
            tree.insert(&key(i), &val(i));
        }

        // Tree should have grown beyond level 1.
        let root = &tree.block_map[&tree.root_block];
        assert!(root.level() >= 2, "expected multi-level tree, got level {}", root.level());

        // Every key must still be findable.
        for i in 0..n {
            assert_eq!(tree.find(&key(i)), Some(val(i).as_slice()));
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
            tree.insert(&key(i), &val(i));
        }
        // Root should still be a leaf.
        assert_eq!(tree.block_map[&tree.root_block].level(), 0);

        // One more triggers a split.
        tree.insert(&key(max), &val(max));
        assert_eq!(tree.block_map[&tree.root_block].level(), 1);

        for i in 0..=max {
            assert_eq!(tree.find(&key(i)), Some(val(i).as_slice()));
        }
        assert_eq!(count_keys(&tree.block_map, tree.root_block), (max + 1) as usize);
        tree.verify();
    }

    // ---------- Key size boundary ----------

    #[test]
    fn test_max_key_size() {
        use crate::block_btree::MAX_KEY_SIZE;

        let mut tree = Btree::new();

        // Key at exactly MAX_KEY_SIZE.
        let full_key = vec![0xABu8; MAX_KEY_SIZE];
        tree.insert(&full_key, b"full");
        assert_eq!(tree.find(&full_key), Some(b"full".as_slice()));

        // Key exceeding MAX_KEY_SIZE — should be silently truncated.
        let long_key = vec![0xCDu8; MAX_KEY_SIZE + 100];
        tree.insert(&long_key, b"truncated");
        // The stored key is truncated to MAX_KEY_SIZE, so looking up the
        // truncated version should find it.
        let truncated_key = vec![0xCDu8; MAX_KEY_SIZE];
        assert_eq!(tree.find(&truncated_key), Some(b"truncated".as_slice()));

        // A key of a different length but same prefix should NOT match.
        let short_key = vec![0xCDu8; MAX_KEY_SIZE - 1];
        assert_eq!(tree.find(&short_key), None);

        tree.verify();
    }

    #[test]
    fn test_empty_key() {
        let mut tree = Btree::new();
        tree.insert(b"", b"empty-key");
        assert_eq!(tree.find(b""), Some(b"empty-key".as_slice()));
        assert_eq!(tree.find(b"x"), None);
        tree.verify();
    }

    // ---------- Value edge cases ----------

    #[test]
    fn test_empty_value() {
        let mut tree = Btree::new();
        tree.insert(&key(1), b"");
        assert_eq!(tree.find(&key(1)), Some(b"".as_slice()));
        tree.verify();
    }

    #[test]
    fn test_max_value_size() {
        use crate::block_btree::MAX_VALUE_SIZE;

        let mut tree = Btree::new();
        let full_val = vec![0x42u8; MAX_VALUE_SIZE];
        tree.insert(&key(1), &full_val);
        assert_eq!(tree.find(&key(1)), Some(full_val.as_slice()));

        // Value exceeding MAX_VALUE_SIZE — truncated.
        let long_val = vec![0x99u8; MAX_VALUE_SIZE + 50];
        tree.insert(&key(2), &long_val);
        let truncated_val = vec![0x99u8; MAX_VALUE_SIZE];
        assert_eq!(tree.find(&key(2)), Some(truncated_val.as_slice()));

        tree.verify();
    }

    // ---------- COW deeper verification ----------

    #[test]
    fn test_cow_multiple_snapshots() {
        let mut tree = Btree::new();
        let mut snapshots = Vec::new();

        // Take a snapshot (save root_block) after every 100 inserts.
        for i in 0u32..500 {
            tree.insert(&key(i), &val(i));
            if i % 100 == 99 {
                snapshots.push((i, tree.root_block));
            }
        }

        // Each snapshot should reflect exactly the keys inserted up to that point.
        for &(last_key, snap_root) in &snapshots {
            for i in 0..=last_key {
                assert_eq!(
                    find(&tree.block_map, snap_root, &key(i)),
                    Some(val(i).as_slice()),
                    "snapshot after key {last_key}: key {i} missing"
                );
            }
            // Key just beyond the snapshot should not exist.
            assert_eq!(
                find(&tree.block_map, snap_root, &key(last_key + 1)),
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
        tree.insert(&key(1), b"v1");
        tree.insert(&key(2), b"v2");
        let snap = tree.root_block;

        // Overwrite key(1) — old snapshot should still see "v1".
        tree.insert(&key(1), b"v1-new");
        assert_eq!(tree.find(&key(1)), Some(b"v1-new".as_slice()));
        assert_eq!(find(&tree.block_map, snap, &key(1)), Some(b"v1".as_slice()));
        assert_eq!(find(&tree.block_map, snap, &key(2)), Some(b"v2".as_slice()));
        tree.verify();
    }

    // ---------- Range scan edge cases ----------

    #[test]
    fn test_range_scan_empty() {
        let mut tree = Btree::new();
        for i in 0..100u32 {
            tree.insert(&key(i), &val(i));
        }
        // Range that contains nothing.
        let results = tree.range_scan(&key(200), &key(300));
        assert!(results.is_empty());
    }

    #[test]
    fn test_range_scan_single_element() {
        let mut tree = Btree::new();
        for i in 0..100u32 {
            tree.insert(&key(i), &val(i));
        }
        let results = tree.range_scan(&key(50), &key(51));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.as_slice(), &key(50));
    }

    #[test]
    fn test_range_scan_entire_tree() {
        let mut tree = Btree::new();
        let n = 500u32;
        for i in 0..n {
            tree.insert(&key(i), &val(i));
        }
        let results = tree.range_scan(&key(0), &key(u32::MAX));
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
            tree.insert(&key(i), &val(i));
        }
        for i in 0..n {
            assert_eq!(tree.find(&key(i)), Some(val(i).as_slice()));
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
            tree.insert(&key(k), &val(v));
            reference.insert(k, v);
        }

        for (&k, &v) in &reference {
            assert_eq!(tree.find(&key(k)), Some(val(v).as_slice()));
        }
        assert_eq!(count_keys(&tree.block_map, tree.root_block), reference.len());
        tree.verify();
    }

    // ---------- Invariant: total key count ----------

    #[test]
    fn test_key_count_after_mixed_operations() {
        let mut tree = Btree::new();
        let mut reference = HashMap::new();

        // Insert 0..1000.
        for i in 0..1000u32 {
            tree.insert(&key(i), &val(i));
            reference.insert(i, i);
        }
        assert_eq!(count_keys(&tree.block_map, tree.root_block), 1000);

        // Overwrite 500..600 — count should not change.
        for i in 500..600u32 {
            tree.insert(&key(i), &val(i + 9999));
            reference.insert(i, i + 9999);
        }
        assert_eq!(count_keys(&tree.block_map, tree.root_block), 1000);

        // Insert new keys 1000..2000.
        for i in 1000..2000u32 {
            tree.insert(&key(i), &val(i));
            reference.insert(i, i);
        }
        assert_eq!(count_keys(&tree.block_map, tree.root_block), 2000);

        // Verify all values.
        for (&k, &v) in &reference {
            assert_eq!(tree.find(&key(k)), Some(val(v).as_slice()));
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
            assert_eq!(
                d, first,
                "leaf {i} is at depth {d}, expected {first}"
            );
        }
        first
    }

    #[test]
    fn test_balance_sequential_inserts() {
        let mut tree = Btree::new();
        // Insert enough keys to create multiple levels.
        for i in 0..5000u32 {
            tree.insert(&key(i), &val(i));
        }
        let depth = assert_balanced(&tree);
        assert!(depth >= 2, "expected multi-level tree, got depth {depth}");
        tree.verify();
    }

    #[test]
    fn test_balance_reverse_inserts() {
        let mut tree = Btree::new();
        for i in (0..5000u32).rev() {
            tree.insert(&key(i), &val(i));
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
            tree.insert(&key(k), &val(k));
        }
        assert_balanced(&tree);
        tree.verify();
    }

    #[test]
    fn test_balance_after_overwrites() {
        let mut tree = Btree::new();
        for i in 0..2000u32 {
            tree.insert(&key(i), &val(i));
        }
        // Overwrite doesn't change tree structure, balance must hold.
        for i in 0..2000u32 {
            tree.insert(&key(i), &val(i + 99999));
        }
        assert_balanced(&tree);
        tree.verify();
    }

    #[test]
    fn test_balance_at_each_split_level() {
        // Insert one key at a time and verify balance after every insert.
        let mut tree = Btree::new();
        for i in 0..2000u32 {
            tree.insert(&key(i), &val(i));
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
                tree.insert(&key(i), &val(i));
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
        assert!(prev_depth <= 5, "unexpectedly deep tree: height={prev_depth}");
        tree.verify();
    }

    #[test]
    fn test_balance_leaf_fill_ratio() {
        // Every leaf (except possibly the last split remainder) should have
        // at least MAX_ENTRIES/2 keys. This is the standard B-tree invariant.
        let mut tree = Btree::new();
        for i in 0..5000u32 {
            tree.insert(&key(i), &val(i));
        }

        let min_keys = crate::block_btree::MAX_ENTRIES / 2;
        check_leaf_fill(&tree.block_map, tree.root_block, min_keys);
        tree.verify();
    }

    fn check_leaf_fill(
        block_map: &HashMap<u64, BtreeNodeRaw>,
        block_nr: u64,
        min_keys: usize,
    ) {
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
            tree.insert(&key(i), &val(i));
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
}
