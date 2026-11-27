//! Storage of the nodes of an A* search.
//!
//! The nodes live in a [`NodeArena`] and are referred to by [`NodeIndex`].
//! The open and closed lists only store these indices.
//!
//! The integer type of the indices is chosen with the type parameter `I` (see [`ArenaIndex`]).
//! It defaults to [`u32`], which keeps the open and closed lists small, but limits the arena to `u32::MAX` nodes.
//! Searches that need more nodes use [`u64`].

use std::{
    fmt::{Debug, Display},
    hash::Hash,
};

use crate::reset::Reset;

mod sealed {
    pub trait Sealed {}

    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
}

/// The integer type of a [`NodeIndex`].
///
/// It is implemented for [`u32`] (the default) and [`u64`], and for [`u16`], which is mainly useful to test how
/// a search handles a full arena.
/// An arena with index type `I` holds at most [`ArenaIndex::CAPACITY`] nodes at the same time.
///
/// This trait is sealed, so it cannot be implemented outside of this crate.
pub trait ArenaIndex:
    sealed::Sealed + Copy + Eq + Ord + Hash + Debug + Display + Send + Sync + 'static
{
    /// The name of the type, e.g. `"u32"`.
    const NAME: &'static str;

    /// The maximum number of nodes in an arena with this index type.
    ///
    /// This is the maximum value of the type (as `usize`, saturated if `usize` is smaller),
    /// such that the largest index is one less than the maximum value.
    const CAPACITY: usize;

    /// Converts a slot number below [`Self::CAPACITY`] into an index.
    fn from_usize(index: usize) -> Self;

    /// Converts the index into a slot number.
    fn as_usize(self) -> usize;
}

macro_rules! impl_arena_index {
    ($($type:ty),*) => {
        $(
            impl ArenaIndex for $type {
                const NAME: &'static str = stringify!($type);

                const CAPACITY: usize = if <$type>::MAX as u128 > usize::MAX as u128 {
                    usize::MAX
                } else {
                    <$type>::MAX as usize
                };

                #[inline]
                fn from_usize(index: usize) -> Self {
                    debug_assert!(index < Self::CAPACITY);
                    index as $type
                }

                #[inline]
                fn as_usize(self) -> usize {
                    self as usize
                }
            }
        )*
    };
}

impl_arena_index!(u16, u32, u64);

/// The index of a node in a [`NodeArena`].
///
/// Its integer type `I` is [`u32`] by default, see [`ArenaIndex`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeIndex<I: ArenaIndex = u32>(I);

impl<I: ArenaIndex> NodeIndex<I> {
    /// Returns the index as `usize`.
    #[inline]
    pub fn as_usize(self) -> usize {
        self.0.as_usize()
    }
}

impl<I: ArenaIndex> Display for NodeIndex<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The error of inserting a node into a full [`NodeArena`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
#[error(
    "the A* node arena is full: it can hold at most {capacity} nodes with {index_type} node indices"
)]
pub struct ArenaFullError {
    /// The maximum number of nodes in the arena, see [`ArenaIndex::CAPACITY`].
    pub capacity: usize,
    /// The name of the index type of the arena, see [`ArenaIndex::NAME`].
    pub index_type: &'static str,
}

impl ArenaFullError {
    /// Creates the error for an arena with index type `I`.
    #[cold]
    #[inline(never)]
    pub fn new<I: ArenaIndex>() -> Self {
        Self {
            capacity: I::CAPACITY,
            index_type: I::NAME,
        }
    }
}

/// The base-2 logarithm of the number of slots of a chunk.
const CHUNK_SIZE_LOG2: u32 = 16;
const CHUNK_SIZE: usize = 1 << CHUNK_SIZE_LOG2;

/// Stores the nodes of an A* search.
///
/// Nodes are stored in chunks of `2^16` slots, so growing a large arena never copies all nodes or needs twice the memory.
/// The first chunk grows like a [`Vec`] until it is full, so small searches allocate little.
/// The other chunks are allocated with their full capacity, but the operating system only maps the pages that are used.
/// Slots of removed nodes are reused by later insertions.
///
/// [`Reset::reset`] removes all nodes, but keeps the chunks for reuse.
///
/// The arena holds at most [`I::CAPACITY`](ArenaIndex::CAPACITY) nodes at the same time.
/// Inserting more returns an [`ArenaFullError`].
pub struct NodeArena<Node, I: ArenaIndex = u32> {
    /// Each chunk holds up to [`CHUNK_SIZE`] slots, and only the last chunk is not full.
    chunks: Vec<Vec<Option<Node>>>,
    /// The number of slots in use or free, i.e. the index of the next new slot.
    slot_count: usize,
    /// Slots whose node was removed.
    free_slots: Vec<NodeIndex<I>>,
}

impl<Node, I: ArenaIndex> NodeArena<Node, I> {
    /// Creates an empty arena without allocating.
    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
            slot_count: 0,
            free_slots: Vec::new(),
        }
    }

    /// Returns the number of nodes in the arena.
    pub fn len(&self) -> usize {
        self.slot_count - self.free_slots.len()
    }

    /// Returns true if the arena contains no nodes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the chunk and the offset in the chunk of a slot.
    #[inline]
    fn locate(index: usize) -> (usize, usize) {
        (index >> CHUNK_SIZE_LOG2, index & (CHUNK_SIZE - 1))
    }

    /// Inserts a node and returns its index.
    ///
    /// Returns an [`ArenaFullError`] (and drops the node) if the arena already contains
    /// [`I::CAPACITY`](ArenaIndex::CAPACITY) nodes.
    #[inline]
    pub fn insert(&mut self, node: Node) -> Result<NodeIndex<I>, ArenaFullError> {
        if let Some(index) = self.free_slots.pop() {
            let slot = self.slot_mut(index);
            debug_assert!(slot.is_none());
            *slot = Some(node);
            return Ok(index);
        }

        if self.slot_count >= I::CAPACITY {
            return Err(ArenaFullError::new::<I>());
        }
        let index = NodeIndex(I::from_usize(self.slot_count));
        let (chunk, offset) = Self::locate(self.slot_count);
        if chunk == self.chunks.len() {
            // The first chunk grows on demand, the others get their full size at once.
            self.chunks.push(if chunk == 0 {
                Vec::new()
            } else {
                Vec::with_capacity(CHUNK_SIZE)
            });
        }
        let chunk = &mut self.chunks[chunk];
        debug_assert_eq!(chunk.len(), offset);
        chunk.push(Some(node));
        self.slot_count += 1;
        Ok(index)
    }

    #[inline]
    fn slot(&self, index: NodeIndex<I>) -> &Option<Node> {
        debug_assert!(index.as_usize() < self.slot_count);
        let (chunk, offset) = Self::locate(index.as_usize());
        &self.chunks[chunk][offset]
    }

    #[inline]
    fn slot_mut(&mut self, index: NodeIndex<I>) -> &mut Option<Node> {
        debug_assert!(index.as_usize() < self.slot_count);
        let (chunk, offset) = Self::locate(index.as_usize());
        &mut self.chunks[chunk][offset]
    }

    /// Returns the node at the given index.
    ///
    /// **Panics** if the node was removed or taken.
    #[inline]
    pub fn get(&self, index: NodeIndex<I>) -> &Node {
        self.slot(index)
            .as_ref()
            .unwrap_or_else(|| panic!("node {index} was removed from the arena"))
    }

    /// Returns the node at the given index mutably.
    ///
    /// **Panics** if the node was removed or taken.
    #[inline]
    pub fn get_mut(&mut self, index: NodeIndex<I>) -> &mut Node {
        self.slot_mut(index)
            .as_mut()
            .unwrap_or_else(|| panic!("node {index} was removed from the arena"))
    }

    /// Moves the node out of its slot, keeping the slot reserved.
    ///
    /// Afterwards, the slot must be filled again with [`Self::put_back`] or released with [`Self::release`].
    #[inline]
    pub fn take(&mut self, index: NodeIndex<I>) -> Node {
        self.slot_mut(index)
            .take()
            .unwrap_or_else(|| panic!("node {index} was removed from the arena"))
    }

    /// Puts a node into a slot emptied by [`Self::take`].
    #[inline]
    pub fn put_back(&mut self, index: NodeIndex<I>, node: Node) {
        let slot = self.slot_mut(index);
        debug_assert!(slot.is_none());
        *slot = Some(node);
    }

    /// Releases a slot emptied by [`Self::take`] for reuse.
    #[inline]
    pub fn release(&mut self, index: NodeIndex<I>) {
        debug_assert!(self.slot(index).is_none());
        self.free_slots.push(index);
    }

    /// Removes the node at the given index and releases its slot for reuse.
    #[inline]
    pub fn remove(&mut self, index: NodeIndex<I>) -> Node {
        let node = self.take(index);
        self.free_slots.push(index);
        node
    }
}

impl<Node, I: ArenaIndex> Default for NodeArena<Node, I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Node, I: ArenaIndex> Reset for NodeArena<Node, I> {
    fn reset(&mut self) {
        // Keep the chunks with their capacity for reuse.
        self.chunks.iter_mut().for_each(Vec::clear);
        self.slot_count = 0;
        self.free_slots.clear();
    }
}

impl<Node, I: ArenaIndex> std::ops::Index<NodeIndex<I>> for NodeArena<Node, I> {
    type Output = Node;

    #[inline]
    fn index(&self, index: NodeIndex<I>) -> &Self::Output {
        self.get(index)
    }
}

impl<Node, I: ArenaIndex> Debug for NodeArena<Node, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeArena")
            .field("index_type", &I::NAME)
            .field("len", &self.len())
            .field("slot_count", &self.slot_count)
            .field("chunk_count", &self.chunks.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use crate::reset::Reset;

    use super::{ArenaFullError, ArenaIndex, CHUNK_SIZE, NodeArena, NodeIndex};

    #[test]
    fn locate() {
        let mut expected = Vec::new();
        for chunk in 0..3 {
            for offset in 0..CHUNK_SIZE {
                expected.push((chunk, offset));
            }
        }
        for (index, expected) in expected.into_iter().enumerate() {
            assert_eq!(NodeArena::<()>::locate(index), expected, "index {index}");
        }
    }

    #[test]
    fn insert_remove_reuse() {
        let mut arena = NodeArena::<_>::new();
        let count = 2 * CHUNK_SIZE + 10_000;
        let indices: Vec<_> = (0..count).map(|i| arena.insert(i).unwrap()).collect();
        assert_eq!(arena.len(), count);
        for (i, &index) in indices.iter().enumerate() {
            assert_eq!(index, NodeIndex(i as u32));
            assert_eq!(arena[index], i);
        }

        assert_eq!(arena.remove(indices[17]), 17);
        let taken = arena.take(indices[42]);
        assert_eq!(taken, 42);
        arena.release(indices[42]);
        assert_eq!(arena.len(), count - 2);

        // Released slots are reused before new slots are added.
        let a = arena.insert(100_000).unwrap();
        let b = arena.insert(100_001).unwrap();
        assert_eq!([a, b], [indices[42], indices[17]]);
        assert_eq!(arena.insert(100_002), Ok(NodeIndex(count as u32)));

        let taken = arena.take(indices[5]);
        arena.put_back(indices[5], taken + 1);
        assert_eq!(arena[indices[5]], 6);
        *arena.get_mut(indices[6]) += 1;
        assert_eq!(arena[indices[6]], 7);

        let chunk_capacities: Vec<_> = arena.chunks.iter().map(Vec::capacity).collect();
        arena.reset();
        assert!(arena.is_empty());
        assert_eq!(arena.insert(5), Ok(NodeIndex(0)));
        assert_eq!(
            arena.chunks.iter().map(Vec::capacity).collect::<Vec<_>>(),
            chunk_capacities
        );
    }

    #[test]
    fn capacities() {
        assert_eq!(<u16 as ArenaIndex>::CAPACITY, 65_535);
        assert_eq!(<u32 as ArenaIndex>::CAPACITY, 4_294_967_295);
        assert_eq!(<u64 as ArenaIndex>::CAPACITY, usize::MAX);
        assert_eq!(<u32 as ArenaIndex>::NAME, "u32");
    }

    /// A full arena returns an error instead of panicking, and accepts nodes again once a slot is free.
    #[test]
    fn full_arena() {
        let mut arena = NodeArena::<usize, u16>::new();
        let capacity = <u16 as ArenaIndex>::CAPACITY;
        let indices: Vec<_> = (0..capacity).map(|i| arena.insert(i).unwrap()).collect();
        assert_eq!(arena.len(), capacity);
        assert_eq!(indices.last(), Some(&NodeIndex(u16::MAX - 1)));

        let error = arena.insert(capacity).unwrap_err();
        assert_eq!(
            error,
            ArenaFullError {
                capacity: 65_535,
                index_type: "u16"
            }
        );
        assert_eq!(
            error.to_string(),
            "the A* node arena is full: it can hold at most 65535 nodes with u16 node indices"
        );
        assert_eq!(arena.len(), capacity);

        assert_eq!(arena.remove(indices[7]), 7);
        assert_eq!(arena.insert(capacity), Ok(indices[7]));
        assert!(arena.insert(capacity + 1).is_err());

        arena.reset();
        assert_eq!(arena.insert(1), Ok(NodeIndex(0)));
    }

    #[test]
    #[should_panic]
    fn get_removed() {
        let mut arena = NodeArena::<_>::new();
        let index = arena.insert(1).unwrap();
        arena.remove(index);
        let _ = arena[index];
    }
}
