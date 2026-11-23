//! Tests of the search with a small explicit graph.

use std::fmt::Display;

use crate::{
    AStar, AStarContext, AStarIdentifier, AStarNode, AStarResult, cost::U64Cost, reset::Reset,
};

impl AStarIdentifier for usize {}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Node {
    identifier: usize,
    predecessor: Option<usize>,
    cost: U64Cost,
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
        write!(f, "{}: {}", self.identifier, self.cost)
    }
}

impl AStarNode for Node {
    type Identifier = usize;
    type EdgeType = ();
    type Cost = U64Cost;

    fn identifier(&self) -> &Self::Identifier {
        &self.identifier
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
        self.predecessor.as_ref()
    }

    fn predecessor_edge_type(&self) -> Option<Self::EdgeType> {
        self.predecessor.map(|_| ())
    }
}

/// A graph given as weighted edges `(from, to, cost)`, with root `0`.
struct Context {
    edges: Vec<(usize, usize, u64)>,
    target: usize,
    cost_limit: Option<u64>,
    label_setting: bool,
}

impl Reset for Context {
    fn reset(&mut self) {}
}

impl AStarContext for Context {
    type Node = Node;

    fn create_root(&self) -> Self::Node {
        Node {
            identifier: 0,
            predecessor: None,
            cost: 0u64.into(),
        }
    }

    fn generate_successors(&mut self, node: &Self::Node, output: &mut impl Extend<Self::Node>) {
        output.extend(
            self.edges
                .iter()
                .filter(|(from, _, _)| *from == node.identifier)
                .map(|&(_, to, cost)| Node {
                    identifier: to,
                    predecessor: Some(node.identifier),
                    cost: node.cost + U64Cost::from(cost),
                }),
        );
    }

    fn is_target(&self, node: &Self::Node) -> bool {
        node.identifier == self.target
    }

    fn cost_limit(&self) -> Option<U64Cost> {
        self.cost_limit.map(Into::into)
    }

    fn memory_limit(&self) -> Option<usize> {
        None
    }

    fn is_label_setting(&self) -> bool {
        self.label_setting
    }
}

fn search(cost_limit: Option<u64>, label_setting: bool) -> AStarResult<usize, U64Cost> {
    // The target 1 has cost 3, the dead end 2 has cost 4.
    let mut a_star = AStar::<_>::new(Context {
        edges: vec![(0, 1, 3), (0, 2, 4)],
        target: 1,
        cost_limit,
        label_setting,
    });
    a_star.initialise();
    a_star.search()
}

/// The cost limit is inclusive: a target whose cost equals the limit is found,
/// also if the search is label-correcting and the limit pruned a node after the target was found.
#[test]
fn cost_limit_is_inclusive() {
    for label_setting in [true, false] {
        for cost_limit in [None, Some(3), Some(4), Some(5)] {
            assert_eq!(
                search(cost_limit, label_setting),
                AStarResult::FoundTarget {
                    identifier: 1,
                    cost: 3u64.into(),
                },
                "cost_limit: {cost_limit:?}, label_setting: {label_setting}",
            );
        }

        assert_eq!(
            search(Some(2), label_setting),
            AStarResult::ExceededCostLimit {
                cost_limit: 2u64.into(),
            },
            "label_setting: {label_setting}",
        );
    }
}
