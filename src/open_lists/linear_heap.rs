use std::collections::VecDeque;

use num_traits::{Bounded, Zero};

use crate::{
    AStarNode,
    arena::{NodeArena, NodeIndex},
    cost::AStarCost,
    open_lists::AStarOpenList,
    reset::Reset,
};

/// An open list for integer costs that stores one bucket of nodes per cost value.
///
/// Buckets are never freed, but kept with their capacity, also over [`Reset::reset`].
/// Emptied buckets below the current minimum cost are reused if a node with a lower cost is pushed,
/// and after a reset, each bucket is reused for the same cost again.
/// Like this, the buckets are allocated once and then only grown.
/// Freeing them would return the memory of large buckets to the operating system,
/// and page-fault it in again when the bucket is allocated anew.
pub struct LinearHeap<Node: AStarNode> {
    /// The bucket at index `i` contains the nodes with cost `cost_offset + i`.
    heap: VecDeque<Vec<NodeIndex>>,
    /// All buckets before this index are empty.
    front: usize,
    cost_offset: Node::Cost,
    len: usize,
}

impl<Node: AStarNode> AStarOpenList<Node> for LinearHeap<Node> {
    #[inline]
    fn new() -> Self {
        Self {
            heap: Default::default(),
            front: 0,
            cost_offset: Zero::zero(),
            len: 0,
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, node: NodeIndex, nodes: &NodeArena<Node>) {
        let cost = nodes[node].cost();
        assert_ne!(cost, Node::Cost::max_value());

        if cost < self.cost_offset {
            let prefix_len = (self.cost_offset - cost).as_usize();
            self.heap.reserve(prefix_len);
            for _ in 0..prefix_len {
                self.heap.push_front(Vec::new());
            }
            self.cost_offset = cost;
            self.front = 0;

            self.heap[0].push(node);
        } else if cost >= self.cost_offset + Node::Cost::from_usize(self.heap.len()) {
            let suffix_len = (cost + Node::Cost::from(1u8)
                - (self.cost_offset + Node::Cost::from_usize(self.heap.len())))
            .as_usize();
            self.heap.reserve(suffix_len);
            for _ in 0..suffix_len {
                self.heap.push_back(Vec::new());
            }

            self.heap.back_mut().unwrap().push(node);
        } else {
            let index = (cost - self.cost_offset).as_usize();
            self.heap[index].push(node);
            self.front = self.front.min(index);
        }

        self.len += 1;
    }

    fn pop_min(&mut self, _nodes: &NodeArena<Node>) -> Option<NodeIndex> {
        while let Some(front) = self.heap.get_mut(self.front) {
            if front.is_empty() {
                // Lazily advance the front of the heap, keeping the empty bucket for reuse.
                self.front += 1;
            } else {
                self.len -= 1;
                return Some(front.swap_remove(front.len() / 2));
            }
        }

        None
    }
}

impl<Node: AStarNode> Reset for LinearHeap<Node> {
    #[inline]
    fn reset(&mut self) {
        // Keep the buckets with their capacity for reuse.
        self.heap.iter_mut().for_each(Vec::clear);
        self.front = 0;
        self.cost_offset = Zero::zero();
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, fmt::Display};

    use crate::{
        AStarNode,
        arena::NodeArena,
        cost::{AStarCost, U64Cost},
        open_lists::AStarOpenList,
        reset::Reset,
    };

    use super::LinearHeap;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    struct Node {
        cost: U64Cost,
        id: usize,
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
            0
        }

        fn predecessor(&self) -> Option<&Self::Identifier> {
            None
        }

        fn predecessor_edge_type(&self) -> Option<Self::EdgeType> {
            None
        }
    }

    /// A simple model of the linear heap: one bucket per cost, popping the middle element of the cheapest bucket.
    #[derive(Default)]
    struct Model(BTreeMap<u64, Vec<Node>>);

    impl Model {
        fn push(&mut self, node: Node) {
            self.0.entry(node.cost.as_u64()).or_default().push(node);
        }

        fn pop_min(&mut self) -> Option<Node> {
            let mut entry = self.0.first_entry()?;
            let bucket = entry.get_mut();
            let node = bucket.swap_remove(bucket.len() / 2);
            if bucket.is_empty() {
                entry.remove();
            }
            Some(node)
        }
    }

    /// Pushes nodes with pseudo-random costs, and pops some of them in between, then pops the rest.
    /// Costs may be lower than the cost of the last popped node.
    /// Checks the popped nodes against the model.
    fn run(heap: &mut LinearHeap<Node>, seed: u64) {
        let mut arena = NodeArena::new();
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };

        let mut model = Model::default();
        let mut last_cost = 0;
        for id in 0..5000 {
            let cost = (last_cost + next() % 20).saturating_sub(3);
            let node = Node {
                cost: cost.into(),
                id,
            };
            heap.push(arena.insert(node), &arena);
            model.push(node);
            assert_eq!(heap.len(), model.0.values().map(Vec::len).sum::<usize>());

            if next() % 3 != 0 {
                let node = heap.pop_min(&arena).map(|index| arena.remove(index));
                assert_eq!(node, model.pop_min());
                last_cost = node.unwrap().cost.as_u64();
            }
        }

        while let Some(index) = heap.pop_min(&arena) {
            assert_eq!(Some(arena.remove(index)), model.pop_min());
        }
        assert_eq!(model.pop_min(), None);
        assert_eq!(heap.len(), 0);
    }

    fn capacity(heap: &LinearHeap<Node>) -> usize {
        heap.heap.iter().map(Vec::capacity).sum()
    }

    #[test]
    fn same_order_as_model() {
        for seed in 0..10 {
            run(&mut LinearHeap::new(), seed);
        }
    }

    #[test]
    fn reset_keeps_buckets() {
        let mut heap = LinearHeap::new();
        run(&mut heap, 0);
        let buckets = heap.heap.len();
        let capacity = capacity(&heap);
        assert!(capacity > 0);

        // Emptied buckets are kept.
        heap.reset();
        assert_eq!(heap.len(), 0);
        assert_eq!(heap.pop_min(&NodeArena::new()), None);
        assert_eq!(heap.heap.len(), buckets);
        assert_eq!(capacity, self::capacity(&heap));

        // Reusing the heap gives the same results as a fresh heap, and running the same input again
        // does not allocate more buckets.
        run(&mut heap, 0);
        assert_eq!(heap.heap.len(), buckets);
        assert_eq!(capacity, self::capacity(&heap));
        for seed in 1..10 {
            heap.reset();
            run(&mut heap, seed);
        }
    }
}
