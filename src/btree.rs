use crate::block_btree::BtreeNodeRaw;

pub struct BtreeNode {
    /// The actual data, stored here after being read from disk
    raw: Box<BtreeNodeRaw>,

    /// Block number of this node on the block device
    block_number: u64,

    /// Whether this node has been modified and needs to be written back (newly allocated block in COW scenario)
    dirty: bool,

    /// We may not need a separate reference count, but this flag lets the cache know someone is reading
    locked: bool,
}