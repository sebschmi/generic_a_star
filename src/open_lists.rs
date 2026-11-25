use crate::{
    AStarNode,
    arena::{NodeArena, NodeIndex},
    reset::Reset,
};

pub mod binary_heap;
pub mod linear_heap;

/// The open list for the A* algorithm.
///
/// This is a priority queue of the indices of nodes stored in a [`NodeArena`].
/// It must sort nodes with [`AStarNodeComparator`](crate::comparator::AStarNodeComparator).
/// The nodes of all indices in the open list stay in the arena until they are popped.
pub trait AStarOpenList<Node: AStarNode>: Reset {
    /// Create a new empty open list.
    fn new() -> Self;

    /// Returns the number of nodes in the open list.
    fn len(&self) -> usize;

    /// Returns true if the open list is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Add the node with the given index to the priority queue.
    fn push(&mut self, index: NodeIndex, nodes: &NodeArena<Node>);

    /// Removes and returns the index of a minimum node from the priority queue.
    ///
    /// Returns `None` if the priority queue is empty.
    fn pop_min(&mut self, nodes: &NodeArena<Node>) -> Option<NodeIndex>;
}
