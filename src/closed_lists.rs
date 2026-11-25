use std::{cmp::Ordering, hash::BuildHasher};

use compare::Compare;
use hashbrown::HashTable;
use rustc_hash::FxBuildHasher;

use crate::{
    AStarNode,
    arena::{NodeArena, NodeIndex},
    comparator::AStarNodeComparator,
    reset::Reset,
};

/// The closed list for the A* algorithm.
///
/// It maps node identifiers to closed nodes.
/// Nodes are inserted by their index in the [`NodeArena`] of the search.
/// The closed list can either keep them in the arena and store their index (like [`HashClosedList`]),
/// or move them out of the arena into its own storage.
pub trait AStarClosedList<Node: AStarNode>: Reset {
    /// Create a new empty closed list.
    fn new() -> Self;

    /// Returns the number of nodes in the closed list.
    fn len(&self) -> usize;

    /// Returns true if the closed list is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert the node with the given index into the closed list, under the identifier of the node.
    ///
    /// If a node is already mapped to the identifier, then the closed list keeps the better of the two nodes
    /// (or the inserted one, if it does not compare nodes).
    /// The closed list removes the node it does not keep from the arena.
    ///
    /// Returns true if a node was already mapped to the identifier.
    fn insert(&mut self, index: NodeIndex, nodes: &mut NodeArena<Node>) -> bool;

    /// Return the node specified by `identifier`.
    ///
    /// Returns `None` if no node with the given identifier exists.
    fn get<'nodes>(
        &'nodes self,
        identifier: &<Node as AStarNode>::Identifier,
        nodes: &'nodes NodeArena<Node>,
    ) -> Option<&'nodes Node>;

    /// Returns true if the given identifier is mapped to a node.
    fn contains_identifier(
        &self,
        identifier: &<Node as AStarNode>::Identifier,
        nodes: &NodeArena<Node>,
    ) -> bool {
        self.get(identifier, nodes).is_some()
    }

    /// Returns an iterator over the nodes in the closed list.
    fn iter<'nodes>(
        &'nodes self,
        nodes: &'nodes NodeArena<Node>,
    ) -> impl Iterator<Item = &'nodes Node>
    where
        Node: 'nodes;

    /// Returns true if the closed list contains a node that is at least as good as `node`,
    /// such that `node` does not need to be expanded.
    fn can_skip_node(&self, node: &Node, nodes: &NodeArena<Node>, is_label_setting: bool) -> bool {
        if let Some(previous_visit) = self.get(node.identifier(), nodes) {
            if is_label_setting {
                // In label-setting mode, if we have already visited the node, we now must be visiting it with a higher or equal cost.
                debug_assert!(
                    previous_visit.cost() + previous_visit.a_star_lower_bound()
                        <= node.cost() + node.a_star_lower_bound(),
                    "Revisiting node {} (previous: {}) at lower costs:\n{}",
                    node,
                    previous_visit,
                    {
                        use std::fmt::Write;
                        let mut previous_visit = previous_visit;
                        let mut node = node;
                        let mut out = String::new();

                        // The predecessor of a node may not be closed yet, if the node is checked when it is generated.
                        writeln!(out, "previous_visit:").unwrap();
                        while let Some(predecessor) = previous_visit.predecessor() {
                            writeln!(out, "{previous_visit}").unwrap();
                            let Some(predecessor) = self.get(predecessor, nodes) else {
                                writeln!(out, "predecessor {predecessor:?} (not closed)").unwrap();
                                break;
                            };
                            previous_visit = predecessor;
                        }

                        writeln!(out, "\nnode:").unwrap();
                        while let Some(predecessor) = node.predecessor() {
                            writeln!(out, "{node}").unwrap();
                            let Some(predecessor) = self.get(predecessor, nodes) else {
                                writeln!(out, "predecessor {predecessor:?} (not closed)").unwrap();
                                break;
                            };
                            node = predecessor;
                        }

                        out
                    }
                );

                true
            } else {
                // If we are label-correcting, we may still find a better node later on.
                // Skip if equal or worse.
                AStarNodeComparator.compare(node, previous_visit) != Ordering::Greater
            }
        } else {
            false
        }
    }
}

/// A closed list that is a hash table of node indices.
///
/// The identifiers are not stored in the table, but read from the nodes in the arena,
/// so each closed node costs only the four bytes of its index (plus the table overhead).
/// The identifiers are hashed with [`FxBuildHasher`].
#[derive(Debug, Clone)]
pub struct HashClosedList {
    table: HashTable<NodeIndex>,
}

#[inline]
fn hash_identifier<Identifier: std::hash::Hash>(identifier: &Identifier) -> u64 {
    FxBuildHasher.hash_one(identifier)
}

impl<Node: AStarNode> AStarClosedList<Node> for HashClosedList {
    #[inline]
    fn new() -> Self {
        Self {
            table: HashTable::new(),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.table.len()
    }

    /// Like [`HashMap::insert`](std::collections::HashMap::insert),
    /// this always replaces an existing node (and removes it from the arena).
    #[inline]
    fn insert(&mut self, index: NodeIndex, nodes: &mut NodeArena<Node>) -> bool {
        let previous = {
            let nodes = &*nodes;
            let identifier = nodes[index].identifier();
            match self.table.entry(
                hash_identifier(identifier),
                |&existing| nodes[existing].identifier() == identifier,
                |&existing| hash_identifier(nodes[existing].identifier()),
            ) {
                hashbrown::hash_table::Entry::Occupied(mut entry) => {
                    Some(std::mem::replace(entry.get_mut(), index))
                }
                hashbrown::hash_table::Entry::Vacant(entry) => {
                    entry.insert(index);
                    None
                }
            }
        };

        if let Some(previous) = previous {
            nodes.remove(previous);
            true
        } else {
            false
        }
    }

    #[inline]
    fn get<'nodes>(
        &'nodes self,
        identifier: &<Node as AStarNode>::Identifier,
        nodes: &'nodes NodeArena<Node>,
    ) -> Option<&'nodes Node> {
        self.table
            .find(hash_identifier(identifier), |&existing| {
                nodes[existing].identifier() == identifier
            })
            .map(|&index| &nodes[index])
    }

    #[inline]
    fn iter<'nodes>(
        &'nodes self,
        nodes: &'nodes NodeArena<Node>,
    ) -> impl Iterator<Item = &'nodes Node>
    where
        Node: 'nodes,
    {
        self.table.iter().map(|&index| &nodes[index])
    }
}

impl Reset for HashClosedList {
    #[inline]
    fn reset(&mut self) {
        self.table.clear();
    }
}
