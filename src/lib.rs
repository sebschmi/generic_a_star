#![forbid(clippy::mod_module_files)]

use std::{
    fmt::{Debug, Display},
    hash::Hash,
    marker::PhantomData,
    ops::ControlFlow,
};

use cost::AStarCost;
use log::trace;
use num_traits::{Bounded, Zero};
use reset::Reset;

use crate::{
    arena::{ArenaFullError, ArenaIndex, NodeArena},
    closed_lists::{AStarClosedList, HashClosedList},
    open_lists::{AStarOpenList, binary_heap::BinaryHeap},
};

pub mod arena;
pub mod closed_lists;
pub mod comparator;
pub mod cost;
pub mod open_lists;
pub mod reset;
#[cfg(test)]
mod tests;

const DEBUG_ASTAR: bool = false;

/// An identifier for nodes in the A* graph.
pub trait AStarIdentifier: Debug + Clone + Eq + Hash {}

/// A node of the A* graph.
/// The node must implement [`Ord`], ordering it by its cost plus A* cost, ascending.
/// The graph defined by the node type must be cycle-free.
pub trait AStarNode: Sized + Ord + Debug + Display {
    /// A unique identifier of the node.
    ///
    /// For example, in case of traditional edit distance, this would be the tuple (i, j) indicating which alignment matrix cell this node belongs to.
    type Identifier: AStarIdentifier;

    /// The type collecting possible edge types.
    ///
    /// These are used when backtracking a solution.
    type EdgeType: Debug;

    type Cost: AStarCost;

    /// Returns the identifier of this node.
    fn identifier(&self) -> &Self::Identifier;

    /// Returns the cost of this node.
    ///
    /// This is the cost measured from the root node, and does NOT include the A* lower bound.
    fn cost(&self) -> Self::Cost;

    /// Returns the A* lower bound of this node.
    fn a_star_lower_bound(&self) -> Self::Cost;

    /// Returns a score that is used to order nodes of the same cost.
    ///
    /// This score should be maximised, which is done via complete search.
    fn secondary_maximisable_score(&self) -> usize;

    /// Returns the identifier of the predecessor of this node.
    fn predecessor(&self) -> Option<&Self::Identifier>;

    /// Returns the edge type used to reach this node from the predecessor, or `None` if this is a root node.
    fn predecessor_edge_type(&self) -> Option<Self::EdgeType>;

    /// Returns the size of a node in bytes, including any owned pointees.
    ///
    /// The default implementation uses [`std::mem::size_of()`] to determine the size.
    fn required_memory() -> usize {
        std::mem::size_of::<Self>()
    }
}

pub trait AStarContext: Reset {
    /// The node type used by the A* algorithm.
    type Node: AStarNode;

    /// Create the root node of the A* graph.
    fn create_root(&self) -> Self::Node;

    /// Generate the successors of this node.
    fn generate_successors(&mut self, node: &Self::Node, output: &mut impl Extend<Self::Node>);

    /// Returns true if this node is a target node of the A* graph.
    fn is_target(&self, node: &Self::Node) -> bool;

    /// Returns the maximum cost that the target node is allowed to have.
    ///
    /// If no target is found with this cost or lower, then [`AStarResult::ExceededCostLimit`] is returned.
    fn cost_limit(&self) -> Option<<Self::Node as AStarNode>::Cost>;

    /// An approximate memory limit for the aligner in bytes.
    ///
    /// If it is exceeded, then [`AStarResult::ExceededMemoryLimit`] is returned
    fn memory_limit(&self) -> Option<usize>;

    /// Returns true if the nodes are generated in a label-setting manner.
    ///
    /// Label setting means that once a node has been closed, it will never be opened at a smaller cost.
    /// On the contrary, if this is set to false, then nodes are allowed to be closed multiple times.
    /// This results in a worse performance, but allows for example to handle negative costs.
    ///
    /// This method returns `true` in its default implementation.
    fn is_label_setting(&self) -> bool {
        true
    }
}

#[derive(Debug, Default)]
pub struct AStarPerformanceCounters {
    pub opened_nodes: usize,
    /// Opened nodes that do not have optimal costs.
    pub suboptimal_opened_nodes: usize,
    pub closed_nodes: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub enum AStarState<NodeIdentifier, Cost> {
    /// The algorithm was just created or reset.
    Empty,
    /// The algorithm was just initialised.
    Init,
    /// The algorithm is searching for a target node.
    Searching,
    /// The algorithm terminated.
    Terminated {
        result: AStarResult<NodeIdentifier, Cost>,
    },
    /// The algorithm failed with an error.
    ///
    /// The search cannot be continued.
    Failed { error: AStarError },
}

/// The return type of [`AStar::search`] and its variants:
/// the outcome of the search, or an error if the search could not be completed.
pub type AStarSearchResult<Node> =
    Result<AStarResult<<Node as AStarNode>::Identifier, <Node as AStarNode>::Cost>, AStarError>;

/// An error that stops an A* search.
///
/// Unlike the outcomes in [`AStarResult`], an error means that the search could not be completed as requested,
/// so it is returned as the [`Err`] variant of [`AStar::search`] and its variants.
#[derive(Debug, Clone, PartialEq, Eq, Hash, thiserror::Error)]
#[non_exhaustive]
pub enum AStarError {
    /// The [`NodeArena`] of the search is full.
    ///
    /// The search needs more nodes at the same time than its [index type](ArenaIndex) allows.
    /// Use a larger index type, e.g. `u64` instead of the default `u32`.
    #[error(transparent)]
    ArenaFull(#[from] ArenaFullError),
}

/// The A* algorithm.
///
/// All nodes are stored in a [`NodeArena`], and the open and closed lists refer to them by [`NodeIndex`](arena::NodeIndex).
/// Nodes that are dropped from the search (because they are dominated or exceed the cost limit) free their slot for reuse.
///
/// `I` is the integer type of the node indices, see [`ArenaIndex`].
/// It limits how many nodes can be stored at the same time:
/// the default [`u32`] allows `u32::MAX` nodes, and [`u64`] removes the limit, but needs more memory for the open and closed lists.
#[derive(Debug)]
pub struct AStar<
    Context: AStarContext,
    I: ArenaIndex = u32,
    ClosedList: AStarClosedList<Context::Node, I> = HashClosedList<I>,
    OpenList: AStarOpenList<Context::Node, I> = BinaryHeap<<Context as AStarContext>::Node, I>,
> {
    state: AStarState<<Context::Node as AStarNode>::Identifier, <Context::Node as AStarNode>::Cost>,
    context: Context,
    nodes: NodeArena<Context::Node, I>,
    closed_list: ClosedList,
    open_list: OpenList,
    performance_counters: AStarPerformanceCounters,
}

/// The memory of an [`AStar`] without its context, to be reused by another search.
#[derive(Debug)]
pub struct AStarBuffers<
    Node: AStarNode,
    I: ArenaIndex = u32,
    ClosedList: AStarClosedList<Node, I> = HashClosedList<I>,
    OpenList: AStarOpenList<Node, I> = BinaryHeap<Node, I>,
> {
    nodes: NodeArena<Node, I>,
    closed_list: ClosedList,
    open_list: OpenList,
    phantom_data: PhantomData<Node>,
}

/// Inserts generated successors into the arena and the open list,
/// unless they exceed the cost limit or are dominated by the closed list.
struct SuccessorSink<
    'a,
    Node: AStarNode,
    I: ArenaIndex,
    ClosedList: AStarClosedList<Node, I>,
    OpenList: AStarOpenList<Node, I>,
> {
    nodes: &'a mut NodeArena<Node, I>,
    closed_list: &'a ClosedList,
    open_list: &'a mut OpenList,
    cost_limit: Node::Cost,
    is_label_setting: bool,
    applied_cost_limit: bool,
    pushed: usize,
    skipped: usize,
    /// Set if a successor did not fit into the arena.
    /// Then all further successors are dropped.
    arena_full: Option<ArenaFullError>,
}

impl<
    Node: AStarNode,
    I: ArenaIndex,
    ClosedList: AStarClosedList<Node, I>,
    OpenList: AStarOpenList<Node, I>,
> Extend<Node> for SuccessorSink<'_, Node, I, ClosedList, OpenList>
{
    #[inline]
    fn extend<T: IntoIterator<Item = Node>>(&mut self, iter: T) {
        if self.arena_full.is_some() {
            return;
        }

        // Internal iteration (`try_for_each`) is faster than a `for` loop for chained and flattened iterators.
        let _ = iter.into_iter().try_for_each(|successor| {
            if successor.cost() + successor.a_star_lower_bound() > self.cost_limit {
                self.applied_cost_limit = true;
                return ControlFlow::Continue(());
            }

            // Drop successors that the closed list already dominates.
            // They would be skipped when popped anyway, so this only saves the work of pushing and popping them.
            // A closed node is never replaced by a worse one, so a successor that is skipped here would also be skipped when popped.
            if self
                .closed_list
                .can_skip_node(&successor, self.nodes, self.is_label_setting)
            {
                self.skipped += 1;
                return ControlFlow::Continue(());
            }

            match self.nodes.insert(successor) {
                Ok(index) => {
                    self.open_list.push(index, self.nodes);
                    self.pushed += 1;
                    ControlFlow::Continue(())
                }
                Err(error) => {
                    self.arena_full = Some(error);
                    ControlFlow::Break(())
                }
            }
        });
    }
}

#[derive(Debug, Clone, Ord, PartialOrd, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "astar_result_type"))]
pub enum AStarResult<NodeIdentifier, Cost> {
    /// The algorithm has found a target node.
    FoundTarget {
        #[cfg_attr(feature = "serde", serde(skip))]
        identifier: NodeIdentifier,
        cost: Cost,
    },

    /// The algorithm terminated before finding a target because the cost limit was reached.
    ExceededCostLimit { cost_limit: Cost },

    /// The algorithm termianted before finding a target because the memory limit was reached.
    ExceededMemoryLimit {
        /// The maximum cost reached before reaching the memory limit.
        max_cost: Cost,
    },

    /// The algorithm terminated, but did not find a target.
    #[default]
    NoTarget,
}

struct BacktrackingIterator<
    'a_star,
    Context: AStarContext,
    I: ArenaIndex,
    ClosedList: AStarClosedList<Context::Node, I>,
    OpenList: AStarOpenList<Context::Node, I>,
> {
    a_star: &'a_star AStar<Context, I, ClosedList, OpenList>,
    current: <Context::Node as AStarNode>::Identifier,
}

struct BacktrackingIteratorWithCost<
    'a_star,
    Context: AStarContext,
    I: ArenaIndex,
    ClosedList: AStarClosedList<Context::Node, I>,
    OpenList: AStarOpenList<Context::Node, I>,
> {
    a_star: &'a_star AStar<Context, I, ClosedList, OpenList>,
    current: <Context::Node as AStarNode>::Identifier,
}

impl<
    Context: AStarContext,
    I: ArenaIndex,
    ClosedList: AStarClosedList<Context::Node, I>,
    OpenList: AStarOpenList<Context::Node, I>,
> AStar<Context, I, ClosedList, OpenList>
{
    pub fn new(context: Context) -> Self {
        Self {
            state: AStarState::Empty,
            context,
            nodes: NodeArena::new(),
            closed_list: ClosedList::new(),
            open_list: OpenList::new(),
            performance_counters: Default::default(),
        }
    }

    pub fn new_with_buffers(
        context: Context,
        mut buffers: AStarBuffers<Context::Node, I, ClosedList, OpenList>,
    ) -> Self {
        buffers.nodes.reset();
        buffers.closed_list.reset();
        buffers.open_list.reset();
        Self {
            state: AStarState::Empty,
            context,
            nodes: buffers.nodes,
            closed_list: buffers.closed_list,
            open_list: buffers.open_list,
            performance_counters: Default::default(),
        }
    }

    pub fn state(
        &self,
    ) -> &AStarState<<Context::Node as AStarNode>::Identifier, <Context::Node as AStarNode>::Cost>
    {
        &self.state
    }

    pub fn context(&self) -> &Context {
        &self.context
    }

    pub fn context_mut(&mut self) -> &mut Context {
        &mut self.context
    }

    pub fn into_context(self) -> Context {
        self.context
    }

    pub fn into_buffers(self) -> AStarBuffers<Context::Node, I, ClosedList, OpenList> {
        AStarBuffers {
            nodes: self.nodes,
            closed_list: self.closed_list,
            open_list: self.open_list,
            phantom_data: Default::default(),
        }
    }

    pub fn closed_node(
        &self,
        node_identifier: &<Context::Node as AStarNode>::Identifier,
    ) -> Option<&Context::Node> {
        self.closed_list.get(node_identifier, &self.nodes)
    }

    pub fn iter_closed_nodes(&self) -> impl Iterator<Item = &Context::Node> {
        self.closed_list.iter(&self.nodes)
    }

    pub fn performance_counters(&self) -> &AStarPerformanceCounters {
        &self.performance_counters
    }

    pub fn reset(&mut self) {
        self.state = AStarState::Empty;
        self.context.reset();
        self.nodes.reset();
        self.closed_list.reset();
        self.open_list.reset();
        self.performance_counters = Default::default();
    }

    pub fn initialise(&mut self) {
        self.initialise_with(|context| context.create_root());
    }

    pub fn initialise_with(&mut self, node: impl FnOnce(&Context) -> Context::Node) {
        assert_eq!(self.state, AStarState::Empty);

        self.state = AStarState::Init;
        // The arena is empty, so it has room for the root.
        let index = self
            .nodes
            .insert(node(&self.context))
            .expect("an empty arena has room for a node");
        self.open_list.push(index, &self.nodes);
    }

    /// Searches until a target node (see [`AStarContext::is_target`]) is closed.
    ///
    /// Returns the outcome of the search, see [`AStarResult`].
    ///
    /// # Errors
    ///
    /// Returns [`AStarError::ArenaFull`] if the search needs more nodes at the same time than the index type `I` allows.
    /// Then the search cannot be continued.
    pub fn search(&mut self) -> AStarSearchResult<Context::Node> {
        self.search_until(|context, node| context.is_target(node))
    }

    /// Like [`Self::search`], but with a custom target condition.
    ///
    /// # Errors
    ///
    /// See [`Self::search`].
    pub fn search_until(
        &mut self,
        is_target: impl FnMut(&Context, &Context::Node) -> bool,
    ) -> AStarSearchResult<Context::Node> {
        self.search_until_with_target_policy(is_target, false)
    }

    /// Like [`Self::search_until`], but if `abort_on_first_target` is set, then the search also returns the first
    /// target it finds when it is label-correcting.
    ///
    /// # Errors
    ///
    /// See [`Self::search`].
    pub fn search_until_with_target_policy(
        &mut self,
        mut is_target: impl FnMut(&Context, &Context::Node) -> bool,
        abort_on_first_target: bool,
    ) -> AStarSearchResult<Context::Node> {
        assert!(matches!(
            self.state,
            AStarState::Init | AStarState::Searching | AStarState::Terminated { .. }
        ));

        let cost_limit = self
            .context
            .cost_limit()
            .unwrap_or(<Context::Node as AStarNode>::Cost::max_value());
        let mut applied_cost_limit = false;
        let memory_limit = self.context.memory_limit().unwrap_or(usize::MAX);
        // The factor of 2.3 is determined empirically.
        let node_count_limit =
            (memory_limit as f64 / Context::Node::required_memory() as f64 / 2.3).round() as usize;

        if self.open_list.is_empty() {
            return Ok(AStarResult::NoTarget);
        }

        self.state = AStarState::Searching;

        let mut last_node = None;
        let mut target_identifier = None;
        let mut target_cost = <Context::Node as AStarNode>::Cost::max_value();
        let mut target_secondary_maximisable_score = 0;

        loop {
            let Some(node_index) = self.open_list.pop_min(&self.nodes) else {
                if last_node.is_none() {
                    unreachable!("Open list was empty.");
                };
                if target_identifier.is_some() {
                    // If we found a target, then we can break the loop and return it normally.
                    // This branch is taken if the search is label-correcting, and all nodes have costs at most the lowest target cost.
                    // The cost limit only pruned nodes more expensive than the limit, and hence more expensive than the target,
                    // so the target is optimal even if the cost limit was applied.
                    break;
                } else if applied_cost_limit {
                    self.state = AStarState::Terminated {
                        result: AStarResult::ExceededCostLimit { cost_limit },
                    };
                    return Ok(AStarResult::ExceededCostLimit { cost_limit });
                } else {
                    self.state = AStarState::Terminated {
                        result: AStarResult::NoTarget,
                    };
                    return Ok(AStarResult::NoTarget);
                }
            };
            // The node is moved out of the arena while it is processed, and its slot is either filled again
            // (if the node is closed) or released.
            let node = self.nodes.take(node_index);

            // Check cost limit.
            // Nodes are ordered by cost plus lower bound.
            if node.cost() + node.a_star_lower_bound() > cost_limit {
                self.nodes.release(node_index);
                self.state = AStarState::Terminated {
                    result: AStarResult::ExceededCostLimit { cost_limit },
                };
                return Ok(AStarResult::ExceededCostLimit { cost_limit });
            }

            // Check memory limit.
            if self.closed_list.len() + self.open_list.len() > node_count_limit {
                self.nodes.release(node_index);
                self.state = AStarState::Terminated {
                    result: AStarResult::ExceededMemoryLimit {
                        max_cost: node.cost(),
                    },
                };
                return Ok(AStarResult::ExceededMemoryLimit {
                    max_cost: node.cost(),
                });
            }

            // If label-correcting, abort when the first node more expensive than the cheapest target is visited.
            if node.cost() + node.a_star_lower_bound() > target_cost {
                debug_assert!(!self.context.is_label_setting());
                self.nodes.release(node_index);
                break;
            }

            last_node = Some(node.identifier().clone());
            let is_target = is_target(&self.context, &node);
            if DEBUG_ASTAR && is_target {
                trace!("Node {node} is target");
            }
            if is_target {
                //println!("Node {node} is target");
            } else {
                //println!("Node {node}");
            }
            debug_assert!(!is_target || node.a_star_lower_bound().is_zero());

            if self
                .closed_list
                .can_skip_node(&node, &self.nodes, self.context.is_label_setting())
            {
                //println!("Skipping node {node}.");
                self.performance_counters.suboptimal_opened_nodes += 1;
                let existing = self
                    .closed_list
                    .get(node.identifier(), &self.nodes)
                    .unwrap();
                let existing_cost = existing.cost();
                let existing_secondary_maximisable_score = existing.secondary_maximisable_score();

                if is_target
                    && (node.cost() < target_cost.min(existing_cost)
                        || (node.cost() == target_cost.min(existing_cost)
                            && node.secondary_maximisable_score()
                                > target_secondary_maximisable_score
                                    .max(existing_secondary_maximisable_score)))
                {
                    if DEBUG_ASTAR {
                        trace!("Updating target to {node}");
                    }
                    //println!("Updating target to {node}");
                    target_identifier = Some(node.identifier().clone());
                    target_cost = node.cost();
                    target_secondary_maximisable_score = node.secondary_maximisable_score();

                    if self.context.is_label_setting() || abort_on_first_target {
                        if DEBUG_ASTAR {
                            trace!("Context is label setting, so we return the first target found");
                        }
                        self.nodes.put_back(node_index, node);
                        let had_previous_visit =
                            self.closed_list.insert(node_index, &mut self.nodes);
                        self.performance_counters.closed_nodes += 1;
                        debug_assert!(
                            !had_previous_visit || !self.context.is_label_setting(),
                            "Visited node again even though we are label setting",
                        );
                        break;
                    }
                } else if is_target
                    && (existing_cost < target_cost
                        || (existing_cost == target_cost
                            && node.secondary_maximisable_score()
                                > existing_secondary_maximisable_score))
                {
                    let node = self
                        .closed_list
                        .get(node.identifier(), &self.nodes)
                        .unwrap();
                    // Set target to existing node if it is better.
                    if DEBUG_ASTAR {
                        trace!("Updating target to {node}");
                    }
                    //println!("Updating target to {node}");
                    target_identifier = Some(node.identifier().clone());
                    target_cost = node.cost();
                    target_secondary_maximisable_score = node.secondary_maximisable_score();

                    if self.context.is_label_setting() || abort_on_first_target {
                        if DEBUG_ASTAR {
                            trace!("Context is label setting, so we return the first target found");
                        }
                        self.performance_counters.closed_nodes += 1;
                        self.nodes.release(node_index);
                        break;
                    }
                }

                if DEBUG_ASTAR {
                    trace!("Skipping node {node}");
                }
                self.nodes.release(node_index);
                continue;
            }

            let mut successors = SuccessorSink {
                nodes: &mut self.nodes,
                closed_list: &self.closed_list,
                open_list: &mut self.open_list,
                cost_limit,
                is_label_setting: self.context.is_label_setting(),
                applied_cost_limit: false,
                pushed: 0,
                skipped: 0,
                arena_full: None,
            };
            self.context.generate_successors(&node, &mut successors);
            applied_cost_limit = applied_cost_limit || successors.applied_cost_limit;
            // Skipped successors count as opened and suboptimal, like they did when they were skipped after popping them.
            self.performance_counters.opened_nodes += successors.pushed + successors.skipped;
            self.performance_counters.suboptimal_opened_nodes += successors.skipped;

            if let Some(error) = successors.arena_full {
                self.nodes.release(node_index);
                let error = AStarError::from(error);
                self.state = AStarState::Failed {
                    error: error.clone(),
                };
                return Err(error);
            }

            if is_target
                && (node.cost() < target_cost
                    || (node.cost() == target_cost
                        && node.secondary_maximisable_score() > target_secondary_maximisable_score))
            {
                if DEBUG_ASTAR {
                    trace!("Updating target to {node}");
                }
                //println!("Updating target to {node}");
                target_identifier = Some(node.identifier().clone());
                target_cost = node.cost();
                target_secondary_maximisable_score = node.secondary_maximisable_score();

                if self.context.is_label_setting() || abort_on_first_target {
                    if DEBUG_ASTAR {
                        trace!("Context is label setting, so we return the first target found");
                    }
                    self.nodes.put_back(node_index, node);
                    let had_previous_visit = self.closed_list.insert(node_index, &mut self.nodes);
                    self.performance_counters.closed_nodes += 1;
                    debug_assert!(!had_previous_visit || !self.context.is_label_setting());
                    break;
                }
            }

            self.nodes.put_back(node_index, node);
            let had_previous_visit = self.closed_list.insert(node_index, &mut self.nodes);
            self.performance_counters.closed_nodes += 1;
            debug_assert!(
                !had_previous_visit || !self.context.is_label_setting(),
                "Node was visited previously: {}",
                self.nodes[node_index],
            );
        }

        let Some(target_identifier) = target_identifier else {
            debug_assert!(!self.context.is_label_setting());
            self.state = AStarState::Terminated {
                result: AStarResult::NoTarget,
            };
            return Ok(AStarResult::NoTarget);
        };

        let cost = self.closed_node(&target_identifier).unwrap().cost();
        debug_assert_eq!(
            cost,
            target_cost,
            "Target node has lower cost than target_cost:\nnode: {}\ntarget_cost: {target_cost}",
            self.closed_node(&target_identifier).unwrap(),
        );
        self.state = AStarState::Terminated {
            result: AStarResult::FoundTarget {
                identifier: target_identifier.clone(),
                cost,
            },
        };
        Ok(AStarResult::FoundTarget {
            identifier: target_identifier,
            cost,
        })
    }

    pub fn backtrack(
        &self,
    ) -> impl use<'_, Context, I, ClosedList, OpenList>
    + Iterator<Item = <Context::Node as AStarNode>::EdgeType> {
        let AStarState::Terminated {
            result: AStarResult::FoundTarget { identifier, .. },
        } = &self.state
        else {
            panic!("Cannot backtrack since no target was found.")
        };

        self.backtrack_from(identifier).unwrap()
    }

    /// Backtrack from the target node to a root node.
    ///
    /// The elements of the iterator are a pair of an edge and the cost of the node that is reached by the edge.
    /// The cost of the first node is never returned.
    pub fn backtrack_with_costs(
        &self,
    ) -> impl use<'_, Context, I, ClosedList, OpenList>
    + Iterator<
        Item = (
            <Context::Node as AStarNode>::EdgeType,
            <Context::Node as AStarNode>::Cost,
        ),
    > {
        let AStarState::Terminated {
            result: AStarResult::FoundTarget { identifier, .. },
        } = &self.state
        else {
            panic!("Cannot backtrack since no target was found.")
        };

        self.backtrack_with_costs_from(identifier).unwrap()
    }

    pub fn backtrack_from(
        &self,
        identifier: &<Context::Node as AStarNode>::Identifier,
    ) -> Option<
        impl use<'_, Context, I, ClosedList, OpenList>
        + Iterator<Item = <Context::Node as AStarNode>::EdgeType>,
    > {
        if self
            .closed_list
            .contains_identifier(identifier, &self.nodes)
        {
            Some(BacktrackingIterator {
                a_star: self,
                current: identifier.clone(),
            })
        } else {
            None
        }
    }

    /// Backtrack from a node to a root node.
    ///
    /// The elements of the iterator are a pair of an edge and the cost of the node that is reached by the edge.
    /// The cost of the first node is never returned.
    #[allow(clippy::type_complexity)]
    pub fn backtrack_with_costs_from(
        &self,
        identifier: &<Context::Node as AStarNode>::Identifier,
    ) -> Option<
        impl use<'_, Context, I, ClosedList, OpenList>
        + Iterator<
            Item = (
                <Context::Node as AStarNode>::EdgeType,
                <Context::Node as AStarNode>::Cost,
            ),
        >,
    > {
        if self
            .closed_list
            .contains_identifier(identifier, &self.nodes)
        {
            Some(BacktrackingIteratorWithCost {
                a_star: self,
                current: identifier.clone(),
            })
        } else {
            None
        }
    }

    /// Reconstruct the path from a root node to the target node.
    pub fn reconstruct_path(&self) -> Vec<<Context::Node as AStarNode>::EdgeType> {
        let AStarState::Terminated {
            result: AStarResult::FoundTarget { .. },
        } = &self.state
        else {
            panic!("Cannot reconstruct path since no target was found.")
        };

        let mut result = self.backtrack().collect::<Vec<_>>();
        result.reverse();
        result
    }
}

impl<NodeIdentifier, Cost: Copy> AStarResult<NodeIdentifier, Cost> {
    /// Returns the maximum cost of closed nodes reached during alignment.
    ///
    /// **Panics** if `self` is [`AStarResult::NoTarget`].
    pub fn cost(&self) -> Cost {
        match self {
            Self::FoundTarget { cost, .. } => *cost,
            Self::ExceededCostLimit { cost_limit } => *cost_limit,
            Self::ExceededMemoryLimit { max_cost } => *max_cost,
            Self::NoTarget => panic!("AStarResult has no costs"),
        }
    }

    pub fn without_node_identifier(&self) -> AStarResult<(), Cost> {
        match *self {
            Self::FoundTarget { cost, .. } => AStarResult::FoundTarget {
                identifier: (),
                cost,
            },
            Self::ExceededCostLimit { cost_limit } => AStarResult::ExceededCostLimit { cost_limit },
            Self::ExceededMemoryLimit { max_cost } => AStarResult::ExceededMemoryLimit { max_cost },
            Self::NoTarget => AStarResult::NoTarget,
        }
    }
}

impl<NodeIdentifier: Clone, Cost> AStarResult<NodeIdentifier, Cost> {
    pub fn transform_cost<TargetCost>(
        &self,
        transform: impl Fn(&Cost) -> TargetCost,
    ) -> AStarResult<NodeIdentifier, TargetCost> {
        match self {
            AStarResult::FoundTarget { identifier, cost } => AStarResult::FoundTarget {
                identifier: identifier.clone(),
                cost: transform(cost),
            },
            AStarResult::ExceededCostLimit { cost_limit } => AStarResult::ExceededCostLimit {
                cost_limit: transform(cost_limit),
            },
            AStarResult::ExceededMemoryLimit { max_cost } => AStarResult::ExceededMemoryLimit {
                max_cost: transform(max_cost),
            },
            AStarResult::NoTarget => AStarResult::NoTarget,
        }
    }
}

impl<
    Context: AStarContext,
    I: ArenaIndex,
    ClosedList: AStarClosedList<Context::Node, I>,
    OpenList: AStarOpenList<Context::Node, I>,
> Iterator for BacktrackingIterator<'_, Context, I, ClosedList, OpenList>
{
    type Item = <Context::Node as AStarNode>::EdgeType;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.a_star.closed_node(&self.current).unwrap();

        if let Some(predecessor) = current.predecessor().cloned() {
            let predecessor_edge_type = current.predecessor_edge_type().unwrap();
            self.current = predecessor;
            Some(predecessor_edge_type)
        } else {
            None
        }
    }
}

impl<
    Context: AStarContext,
    I: ArenaIndex,
    ClosedList: AStarClosedList<Context::Node, I>,
    OpenList: AStarOpenList<Context::Node, I>,
> Iterator for BacktrackingIteratorWithCost<'_, Context, I, ClosedList, OpenList>
{
    type Item = (
        <Context::Node as AStarNode>::EdgeType,
        <Context::Node as AStarNode>::Cost,
    );

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.a_star.closed_node(&self.current).unwrap();
        let cost = current.cost();

        if let Some(predecessor) = current.predecessor().cloned() {
            let predecessor_edge_type = current.predecessor_edge_type().unwrap();
            self.current = predecessor;
            Some((predecessor_edge_type, cost))
        } else {
            None
        }
    }
}

impl<
    Node: AStarNode,
    I: ArenaIndex,
    ClosedList: AStarClosedList<Node, I>,
    OpenList: AStarOpenList<Node, I>,
> Default for AStarBuffers<Node, I, ClosedList, OpenList>
{
    fn default() -> Self {
        Self {
            nodes: NodeArena::new(),
            closed_list: ClosedList::new(),
            open_list: OpenList::new(),
            phantom_data: Default::default(),
        }
    }
}

impl<NodeIdentifier, Cost: Display> Display for AStarResult<NodeIdentifier, Cost> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AStarResult::FoundTarget { cost, .. } => write!(f, "Reached target with cost {cost}"),
            AStarResult::ExceededCostLimit { cost_limit } => {
                write!(f, "Exceeded cost limit of {cost_limit}")
            }
            AStarResult::ExceededMemoryLimit { max_cost } => write!(
                f,
                "Exceeded memory limit, but reached a maximum cost of {max_cost}"
            ),
            AStarResult::NoTarget => write!(f, "Found no target"),
        }
    }
}

impl<T: AStarNode> AStarNode for Box<T> {
    type Identifier = <T as AStarNode>::Identifier;

    type EdgeType = <T as AStarNode>::EdgeType;

    type Cost = <T as AStarNode>::Cost;

    #[inline]
    fn identifier(&self) -> &Self::Identifier {
        <T as AStarNode>::identifier(self)
    }

    #[inline]
    fn cost(&self) -> Self::Cost {
        <T as AStarNode>::cost(self)
    }

    #[inline]
    fn a_star_lower_bound(&self) -> Self::Cost {
        <T as AStarNode>::a_star_lower_bound(self)
    }

    #[inline]
    fn secondary_maximisable_score(&self) -> usize {
        <T as AStarNode>::secondary_maximisable_score(self)
    }

    #[inline]
    fn predecessor(&self) -> Option<&Self::Identifier> {
        <T as AStarNode>::predecessor(self)
    }

    #[inline]
    fn predecessor_edge_type(&self) -> Option<Self::EdgeType> {
        <T as AStarNode>::predecessor_edge_type(self)
    }

    #[inline]
    fn required_memory() -> usize {
        <T as AStarNode>::required_memory() + std::mem::size_of::<Box<()>>()
    }
}
