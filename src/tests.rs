//! Tests of the search with a small explicit graph.

use std::fmt::Display;

use crate::{
    AStar, AStarContext, AStarError, AStarIdentifier, AStarNode, AStarResult, AStarState,
    arena::{ArenaFullError, ArenaIndex},
    closed_lists::{AStarClosedList, HashClosedList},
    cost::{AStarCost, U64Cost},
    open_lists::{AStarOpenList, linear_heap::LinearHeap},
    reset::Reset,
};

impl AStarIdentifier for usize {}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Node {
    identifier: usize,
    predecessor: Option<usize>,
    cost: U64Cost,
    lower_bound: U64Cost,
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.cost + self.lower_bound).cmp(&(other.cost + other.lower_bound))
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
        self.lower_bound
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
    /// The outgoing edges `(to, cost)` of each node.
    successors: Vec<Vec<(usize, u64)>>,
    /// The A* lower bound of each node, or zero if the node has no entry.
    lower_bounds: Vec<u64>,
    target: usize,
    cost_limit: Option<u64>,
    label_setting: bool,
}

impl Context {
    fn new(edges: Vec<(usize, usize, u64)>, target: usize) -> Self {
        let mut successors = Vec::new();
        for (from, to, cost) in edges {
            if successors.len() <= from {
                successors.resize_with(from + 1, Vec::new);
            }
            successors[from].push((to, cost));
        }
        Self {
            successors,
            lower_bounds: Vec::new(),
            target,
            cost_limit: None,
            label_setting: true,
        }
    }

    fn lower_bound(&self, identifier: usize) -> U64Cost {
        self.lower_bounds
            .get(identifier)
            .copied()
            .unwrap_or(0)
            .into()
    }
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
            lower_bound: self.lower_bound(0),
        }
    }

    fn generate_successors(&mut self, node: &Self::Node, output: &mut impl Extend<Self::Node>) {
        output.extend(
            self.successors
                .get(node.identifier)
                .into_iter()
                .flatten()
                .map(|&(to, cost)| Node {
                    identifier: to,
                    predecessor: Some(node.identifier),
                    cost: node.cost + U64Cost::from(cost),
                    lower_bound: self.lower_bound(to),
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
        cost_limit,
        label_setting,
        ..Context::new(vec![(0, 1, 3), (0, 2, 4)], 1)
    });
    a_star.initialise();
    a_star.search().unwrap()
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

/// Successors that are already closed are not pushed into the open list,
/// but they are still counted as opened and suboptimal nodes.
#[test]
fn closed_successors_are_not_pushed() {
    for label_setting in [true, false] {
        // Nodes are closed in the order 0, 1, 2, 3, 4.
        // Closing 2 generates 1 again, which is already closed, and closing 3 generates 2 again.
        // Without skipping them when generated, the second copy of 1 (cost 52) would still be in the open list
        // when the target is found.
        let mut a_star = AStar::<_>::new(Context {
            label_setting,
            ..Context::new(
                vec![
                    (0, 1, 1),
                    (0, 2, 2),
                    (1, 3, 2),
                    (2, 1, 50),
                    (3, 2, 1),
                    (3, 4, 10),
                ],
                4,
            )
        });
        a_star.initialise();
        assert_eq!(
            a_star.search().unwrap(),
            AStarResult::FoundTarget {
                identifier: 4,
                cost: 13u64.into(),
            }
        );
        assert_eq!(a_star.open_list.len(), 0, "label_setting: {label_setting}");
        let counters = a_star.performance_counters();
        assert_eq!(counters.opened_nodes, 6, "label_setting: {label_setting}");
        assert_eq!(counters.suboptimal_opened_nodes, 2);
        assert_eq!(counters.closed_nodes, 5);
        assert_eq!(a_star.reconstruct_path().len(), 3);
    }
}

/// A small deterministic pseudo random number generator (xorshift64*).
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

/// Returns a random graph with `node_count` nodes (possibly with cycles and parallel edges).
fn random_graph(random: &mut Random, node_count: usize) -> Vec<(usize, usize, u64)> {
    let edge_count = random.below(4 * node_count as u64) as usize;
    (0..edge_count)
        .map(|_| {
            (
                random.below(node_count as u64) as usize,
                random.below(node_count as u64) as usize,
                random.below(10),
            )
        })
        .collect()
}

/// Shortest distances from `source` (Bellman-Ford), `None` if unreachable.
fn distances(edges: &[(usize, usize, u64)], node_count: usize, source: usize) -> Vec<Option<u64>> {
    let mut distances = vec![None; node_count];
    distances[source] = Some(0);
    for _ in 0..node_count {
        for &(from, to, cost) in edges {
            if let Some(distance) = distances[from] {
                let distance = distance + cost;
                if distances[to].is_none_or(|existing| distance < existing) {
                    distances[to] = Some(distance);
                }
            }
        }
    }
    distances
}

/// Every node in the arena is either open or closed: dropped nodes release their slots.
fn check_no_leaked_nodes<
    I: ArenaIndex,
    ClosedList: AStarClosedList<Node, I>,
    OpenList: AStarOpenList<Node, I>,
>(
    a_star: &AStar<Context, I, ClosedList, OpenList>,
    case: &str,
) {
    assert_eq!(
        a_star.nodes.len(),
        a_star.closed_list.len() + a_star.open_list.len(),
        "{case}"
    );
}

/// Checks that a search finds a target at the shortest distance and that backtracking reproduces that cost.
fn check_search<
    I: ArenaIndex,
    ClosedList: AStarClosedList<Node, I>,
    OpenList: AStarOpenList<Node, I>,
>(
    a_star: &mut AStar<Context, I, ClosedList, OpenList>,
    distance: Option<u64>,
    case: &str,
) {
    a_star.initialise();
    let result = a_star.search().unwrap();
    check_no_leaked_nodes(a_star, case);
    match distance {
        Some(distance) => {
            assert_eq!(
                result,
                AStarResult::FoundTarget {
                    identifier: a_star.context().target,
                    cost: distance.into(),
                },
                "{case}"
            );
            let path_cost: u64 = a_star
                .backtrack_with_costs()
                .next()
                .map(|(_, cost)| cost.as_primitive())
                .unwrap_or(0);
            assert_eq!(path_cost, distance, "{case}");
        }
        None => assert_eq!(result, AStarResult::NoTarget, "{case}"),
    }
}

/// Model-based test: random graphs, label setting with consistent lower bounds (zero and exact),
/// and label correcting with admissible but inconsistent lower bounds.
#[test]
fn random_graphs_find_shortest_paths() {
    let mut random = Random(0x1234_5678_9abc_def1);
    for case in 0..2000 {
        let node_count = 1 + random.below(12) as usize;
        let edges = random_graph(&mut random, node_count);
        let target = random.below(node_count as u64) as usize;
        let distance = distances(&edges, node_count, 0)[target];

        // Exact distances to the target, computed on the reversed graph.
        // Nodes that cannot reach the target get a large lower bound, which keeps the lower bounds consistent.
        let reversed: Vec<_> = edges.iter().map(|&(f, t, c)| (t, f, c)).collect();
        let exact: Vec<u64> = distances(&reversed, node_count, target)
            .into_iter()
            .map(|d| d.unwrap_or(1000))
            .collect();

        for (label_setting, lower_bounds) in [
            (true, Vec::new()),
            (true, exact.clone()),
            (false, Vec::new()),
            (false, exact.iter().map(|&d| random.below(d + 1)).collect()),
        ] {
            let mut a_star = AStar::<_>::new(Context {
                lower_bounds: lower_bounds.clone(),
                label_setting,
                ..Context::new(edges.clone(), target)
            });
            check_search(
                &mut a_star,
                distance,
                &format!(
                    "case {case}: edges {edges:?}, target {target}, label_setting {label_setting}, lower_bounds {lower_bounds:?}"
                ),
            );

            // Searching again after a reset with reused buffers gives the same result.
            a_star.reset();
            check_search(&mut a_star, distance, &format!("case {case} after reset"));

            // Searching with reused buffers gives the same result.
            let mut a_star = AStar::new_with_buffers(
                Context {
                    lower_bounds: lower_bounds.clone(),
                    label_setting,
                    ..Context::new(edges.clone(), target)
                },
                a_star.into_buffers(),
            );
            check_search(&mut a_star, distance, &format!("case {case} with buffers"));

            // The linear heap gives the same costs.
            let mut a_star = AStar::<_, u32, HashClosedList, LinearHeap<_>>::new(Context {
                lower_bounds: lower_bounds.clone(),
                label_setting,
                ..Context::new(edges.clone(), target)
            });
            check_search(
                &mut a_star,
                distance,
                &format!("case {case} with linear heap"),
            );

            // The other index types give the same costs.
            let mut a_star = AStar::<_, u64>::new(Context {
                lower_bounds: lower_bounds.clone(),
                label_setting,
                ..Context::new(edges.clone(), target)
            });
            check_search(&mut a_star, distance, &format!("case {case} with u64"));
            let mut a_star = AStar::<_, u16, HashClosedList<_>, LinearHeap<_, _>>::new(Context {
                lower_bounds: lower_bounds.clone(),
                label_setting,
                ..Context::new(edges.clone(), target)
            });
            check_search(&mut a_star, distance, &format!("case {case} with u16"));
        }
    }
}

/// Model-based test of a resumed search, as done by tsalign's windowed aligners:
/// after finding the target, the search is continued until the first node more expensive than a cost limit.
/// Then every node within the cost limit must be closed with its shortest distance,
/// and every closed node must have a valid path to the root.
#[test]
fn resumed_search_closes_all_nodes_within_cost_limit() {
    let mut random = Random(0x0fed_cba9_8765_4321);
    for case in 0..2000 {
        let node_count = 1 + random.below(12) as usize;
        let edges = random_graph(&mut random, node_count);
        let target = random.below(node_count as u64) as usize;
        let distances = distances(&edges, node_count, 0);
        let mut a_star = AStar::<_>::new(Context::new(edges.clone(), target));
        a_star.initialise();
        let first_result = a_star.search().unwrap();
        let case = format!("case {case}: edges {edges:?}, target {target}");
        let Some(target_distance) = distances[target] else {
            assert_eq!(first_result, AStarResult::NoTarget, "{case}");
            continue;
        };
        assert_eq!(first_result.cost(), target_distance.into(), "{case}");

        let cost_limit = target_distance + random.below(10);
        a_star
            .search_until_with_target_policy(|_, node| node.cost > cost_limit.into(), true)
            .unwrap();

        for (identifier, distance) in distances.iter().enumerate() {
            if let Some(distance) = *distance
                && distance <= cost_limit
            {
                let node = a_star.closed_node(&identifier).unwrap_or_else(|| {
                    panic!("{case}: node {identifier} with distance {distance} is not closed")
                });
                assert_eq!(node.cost, distance.into(), "{case}");
            }
        }
        check_no_leaked_nodes(&a_star, &case);
        for node in a_star.iter_closed_nodes() {
            assert_eq!(
                Some(node.cost.as_primitive()),
                distances[node.identifier],
                "{case}: closed node {node}"
            );
            assert!(
                a_star.backtrack_from(&node.identifier).unwrap().count() < node_count,
                "{case}"
            );
        }
    }
}

/// A search that needs more nodes at the same time than its index type allows fails with an error
/// instead of panicking, and the same search with a larger index type succeeds.
#[test]
fn full_arena_is_an_error() {
    // The root has more successors than an arena with u16 indices can hold, and the target is behind the last one.
    let successor_count = <u16 as ArenaIndex>::CAPACITY + 10;
    let mut edges: Vec<_> = (1..=successor_count).map(|to| (0, to, 1)).collect();
    edges.push((successor_count, successor_count + 1, 1));
    let target = successor_count + 1;

    for label_setting in [true, false] {
        let context = || Context {
            label_setting,
            ..Context::new(edges.clone(), target)
        };

        let mut a_star = AStar::<_, u16>::new(context());
        a_star.initialise();
        let expected_error = AStarError::ArenaFull(ArenaFullError {
            capacity: 65_535,
            index_type: "u16",
        });
        assert_eq!(a_star.search(), Err(expected_error.clone()));
        assert_eq!(
            a_star.state(),
            &AStarState::Failed {
                error: expected_error.clone()
            }
        );
        assert_eq!(
            expected_error.to_string(),
            "the A* node arena is full: it can hold at most 65535 nodes with u16 node indices"
        );
        // All successors that fit were opened.
        assert_eq!(a_star.performance_counters().opened_nodes, 65_534);

        // The linear heap behaves the same.
        let mut a_star = AStar::<_, u16, HashClosedList<_>, LinearHeap<_, _>>::new(context());
        a_star.initialise();
        assert_eq!(a_star.search(), Err(expected_error));

        for result in [
            {
                let mut a_star = AStar::<_>::new(context());
                a_star.initialise();
                a_star.search()
            },
            {
                let mut a_star = AStar::<_, u64>::new(context());
                a_star.initialise();
                a_star.search()
            },
        ] {
            assert_eq!(
                result,
                Ok(AStarResult::FoundTarget {
                    identifier: target,
                    cost: 2u64.into(),
                })
            );
        }
    }
}
