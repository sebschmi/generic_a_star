use std::marker::PhantomData;

use compare::Compare;

use crate::{
    AStarNode,
    arena::{ArenaIndex, NodeArena, NodeIndex},
    comparator::AStarNodeComparator,
    open_lists::AStarOpenList,
    reset::Reset,
};

/// A binary heap of node indices, ordered by [`AStarNodeComparator`] on the nodes.
///
/// It uses the same algorithm as [`std::collections::BinaryHeap`] (and `binary_heap_plus`),
/// so it pops nodes in the same order as a binary heap of the nodes themselves.
///
/// `I` is the index type of the arena, see [`ArenaIndex`].
#[derive(Debug)]
pub struct BinaryHeap<Node, I: ArenaIndex = u32> {
    data: Vec<NodeIndex<I>>,
    phantom_data: PhantomData<fn() -> Node>,
}

impl<Node: AStarNode, I: ArenaIndex> BinaryHeap<Node, I> {
    /// Returns true if the node at `a` is ordered at most as high as the node at `b`.
    #[inline]
    fn le(nodes: &NodeArena<Node, I>, a: NodeIndex<I>, b: NodeIndex<I>) -> bool {
        AStarNodeComparator.compares_le(&nodes[a], &nodes[b])
    }

    /// Moves the element at `pos` up towards the root, and returns its new position.
    #[inline]
    fn sift_up(&mut self, nodes: &NodeArena<Node, I>, start: usize, mut pos: usize) -> usize {
        let element = self.data[pos];
        // Look up the moving node only once.
        let element_node = &nodes[element];
        while pos > start {
            let parent = (pos - 1) / 2;
            if AStarNodeComparator.compares_le(element_node, &nodes[self.data[parent]]) {
                break;
            }
            self.data[pos] = self.data[parent];
            pos = parent;
        }
        self.data[pos] = element;
        pos
    }

    /// Moves the element at `pos` all the way down, and then up to its place.
    #[inline]
    fn sift_down_to_bottom(&mut self, nodes: &NodeArena<Node, I>, mut pos: usize) {
        let end = self.data.len();
        let start = pos;
        let element = self.data[pos];

        let mut child = 2 * pos + 1;
        while child <= end.saturating_sub(2) {
            child += Self::le(nodes, self.data[child], self.data[child + 1]) as usize;
            self.data[pos] = self.data[child];
            pos = child;
            child = 2 * pos + 1;
        }

        if child == end - 1 {
            self.data[pos] = self.data[child];
            pos = child;
        }
        self.data[pos] = element;

        self.sift_up(nodes, start, pos);
    }
}

impl<Node: AStarNode, I: ArenaIndex> AStarOpenList<Node, I> for BinaryHeap<Node, I> {
    #[inline]
    fn new() -> Self {
        Self {
            data: Vec::new(),
            phantom_data: PhantomData,
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn push(&mut self, index: NodeIndex<I>, nodes: &NodeArena<Node, I>) {
        let old_len = self.data.len();
        self.data.push(index);
        self.sift_up(nodes, 0, old_len);
    }

    #[inline]
    fn pop_min(&mut self, nodes: &NodeArena<Node, I>) -> Option<NodeIndex<I>> {
        self.data.pop().map(|mut item| {
            if !self.data.is_empty() {
                std::mem::swap(&mut item, &mut self.data[0]);
                self.sift_down_to_bottom(nodes, 0);
            }
            item
        })
    }
}

impl<Node, I: ArenaIndex> Reset for BinaryHeap<Node, I> {
    #[inline]
    fn reset(&mut self) {
        self.data.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Display;

    use crate::{
        AStarNode, arena::NodeArena, comparator::AStarNodeComparator, cost::U64Cost,
        open_lists::AStarOpenList, reset::Reset,
    };

    use super::BinaryHeap;

    /// Ordered only by cost, so nodes with equal cost are ties whose order depends on the heap algorithm.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Node {
        cost: U64Cost,
        score: usize,
        id: usize,
    }

    impl Ord for Node {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.cost.cmp(&other.cost)
        }
    }

    impl PartialOrd for Node {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Display for Node {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}: {}", self.id, self.cost)
        }
    }

    impl AStarNode for Node {
        type Identifier = usize;
        type EdgeType = ();
        type Cost = U64Cost;

        fn identifier(&self) -> &Self::Identifier {
            &self.id
        }

        fn cost(&self) -> Self::Cost {
            self.cost
        }

        fn a_star_lower_bound(&self) -> Self::Cost {
            0u64.into()
        }

        fn secondary_maximisable_score(&self) -> usize {
            self.score
        }

        fn predecessor(&self) -> Option<&Self::Identifier> {
            None
        }

        fn predecessor_edge_type(&self) -> Option<Self::EdgeType> {
            None
        }
    }

    /// The heap pops the nodes in exactly the same order as `binary_heap_plus` (which copies the standard library),
    /// including the order of ties.
    #[test]
    fn same_order_as_binary_heap_plus() {
        let mut state = 7u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };

        let mut heap = BinaryHeap::<_>::new();
        let mut arena = NodeArena::new();
        for round in 0..3 {
            let mut model =
                binary_heap_plus::BinaryHeap::from_vec_cmp(Vec::new(), AStarNodeComparator);
            for id in 0..5000 {
                let node = Node {
                    cost: (next() % 30).into(),
                    score: (next() % 3) as usize,
                    id,
                };
                heap.push(arena.insert(node), &arena);
                model.push(node);
                assert_eq!(heap.len(), model.len());

                if next() % 3 != 0 {
                    let index = heap.pop_min(&arena).unwrap();
                    assert_eq!(Some(arena.remove(index)), model.pop(), "round {round}");
                }
            }

            while let Some(index) = heap.pop_min(&arena) {
                assert_eq!(Some(arena.remove(index)), model.pop(), "round {round}");
            }
            assert!(model.is_empty());
            heap.reset();
            arena.reset();
        }
    }
}
