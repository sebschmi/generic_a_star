# Generic A*

A generic implementation of the A* algorithm.
A useful building block for many dynamic programming problems.

## Node storage and index width

All nodes of a search live in an arena (`arena::NodeArena`), and the open and closed lists store only node indices.
The integer type of the indices is the type parameter `I` of `AStar` (and of the arena and the lists).
It is `u32` by default, which keeps the lists small but allows at most `u32::MAX` nodes at the same time.
Use `AStar<Context, u64>` for searches that need more nodes.

If the arena is full, the search does not panic, but returns `Err(AStarError::ArenaFull(_))`.
