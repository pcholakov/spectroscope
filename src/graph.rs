//! Directed graphs with labelled edges, and the cycle queries checkers need:
//! strongly connected components, and the shortest cycle through a node.
//!
//! Every query takes an `include` predicate over edge labels, so one graph can
//! answer for several consistency models, each of which counts a different set
//! of edge kinds. Results depend only on node numbering and edge insertion
//! order, never on hashing.

use std::collections::{BTreeMap, VecDeque};

/// A directed multigraph with nodes `0..node_count()` and labelled edges.
#[derive(Debug)]
pub(crate) struct Digraph<E> {
    /// Outgoing edges of each node, in insertion order.
    out: Vec<Vec<(usize, E)>>,
}

/// One edge of a path: source, target and label.
pub(crate) type PathEdge<'g, E> = (usize, usize, &'g E);

impl<E> Digraph<E> {
    /// A graph with `node_count` nodes and no edges.
    pub(crate) fn new(node_count: usize) -> Self {
        Self {
            out: (0..node_count).map(|_| Vec::new()).collect(),
        }
    }

    /// Add a node with no edges, returning its number.
    pub(crate) fn add_node(&mut self) -> usize {
        self.out.push(Vec::new());
        self.out.len() - 1
    }

    pub(crate) fn add_edge(&mut self, source: usize, target: usize, label: E) {
        assert!(
            target < self.out.len(),
            "edge target {target} is not a node"
        );
        self.out[source].push((target, label));
    }

    /// The strongly connected components that contain a cycle, counting only
    /// edges whose label satisfies `include`. Each component is sorted, and
    /// components are ordered by their smallest node.
    ///
    /// Tarjan's algorithm, iterative so deep graphs cannot overflow the stack.
    /// Cost: O(nodes + edges).
    pub(crate) fn cyclic_components(&self, include: impl Fn(&E) -> bool) -> Vec<Vec<usize>> {
        let node_count = self.out.len();
        let mut visit_order: Vec<Option<usize>> = vec![None; node_count];
        // The smallest visit order reachable through the DFS subtree plus one
        // edge to a node still on `stack`.
        let mut lowlink = vec![0; node_count];
        let mut on_stack = vec![false; node_count];
        let mut stack = Vec::new();
        let mut next_visit = 0;
        let mut components = Vec::new();

        for root in 0..node_count {
            if visit_order[root].is_some() {
                continue;
            }
            // DFS frames: a node and the position of its next edge to explore.
            let mut frames = vec![(root, 0)];
            visit_order[root] = Some(next_visit);
            lowlink[root] = next_visit;
            next_visit += 1;
            stack.push(root);
            on_stack[root] = true;

            while let Some(frame) = frames.last_mut() {
                let (node, edge_position) = *frame;
                if let Some((target, label)) = self.out[node].get(edge_position) {
                    frame.1 += 1;
                    if !include(label) {
                        continue;
                    }
                    match visit_order[*target] {
                        None => {
                            visit_order[*target] = Some(next_visit);
                            lowlink[*target] = next_visit;
                            next_visit += 1;
                            stack.push(*target);
                            on_stack[*target] = true;
                            frames.push((*target, 0));
                        }
                        Some(target_order) if on_stack[*target] => {
                            lowlink[node] = lowlink[node].min(target_order);
                        }
                        Some(_) => {}
                    }
                    continue;
                }

                frames.pop();
                if let Some(&(parent, _)) = frames.last() {
                    lowlink[parent] = lowlink[parent].min(lowlink[node]);
                }
                if Some(lowlink[node]) == visit_order[node] {
                    let mut component = Vec::new();
                    while let Some(member) = stack.pop() {
                        on_stack[member] = false;
                        component.push(member);
                        if member == node {
                            break;
                        }
                    }
                    if component.len() > 1 {
                        component.sort_unstable();
                        components.push(component);
                    }
                }
            }
        }
        assert!(
            stack.is_empty(),
            "every visited node belongs to a component"
        );
        components.sort_unstable_by_key(|component| component[0]);
        components
    }

    /// A cheapest cycle through `start` that stays within `component` (sorted),
    /// as edges from `start` back to `start`. `cost` prices each edge, and
    /// `None` excludes it. `None` if there is no such cycle.
    ///
    /// 0-1 breadth-first search, so among equally cheap cycles the one found
    /// through earlier-inserted edges wins.
    /// Cost: O(edges within the component · log nodes).
    pub(crate) fn shortest_cycle(
        &self,
        start: usize,
        component: &[usize],
        cost: impl Fn(&E) -> Option<Cost>,
    ) -> Option<Vec<PathEdge<'_, E>>> {
        let mut distance: BTreeMap<usize, usize> = BTreeMap::from([(start, 0)]);
        let mut predecessor: BTreeMap<usize, (usize, &E)> = BTreeMap::new();
        // The cheapest edge closing a cycle so far, with the cycle's total cost.
        let mut closing: Option<(usize, usize, &E)> = None;
        let mut queue = VecDeque::from([(start, 0)]);
        while let Some((node, node_distance)) = queue.pop_front() {
            let stale = distance
                .get(&node)
                .is_some_and(|&best| best < node_distance);
            let no_better = closing.is_some_and(|(total, _, _)| node_distance >= total);
            if stale || no_better {
                continue;
            }
            for (target, label) in &self.out[node] {
                let Some(edge_cost) = cost(label) else {
                    continue;
                };
                if component.binary_search(target).is_err() {
                    continue;
                }
                let target_distance = node_distance + edge_cost.units();
                if *target == start {
                    if closing.is_none_or(|(total, _, _)| target_distance < total) {
                        closing = Some((target_distance, node, label));
                    }
                } else if distance
                    .get(target)
                    .is_none_or(|&best| target_distance < best)
                {
                    distance.insert(*target, target_distance);
                    predecessor.insert(*target, (node, label));
                    match edge_cost {
                        Cost::Free => queue.push_front((*target, target_distance)),
                        Cost::Step => queue.push_back((*target, target_distance)),
                    }
                }
            }
        }

        let (_, last, last_label) = closing?;
        let mut path = vec![(last, start, last_label)];
        let mut current = last;
        while current != start {
            let (previous, previous_label) = predecessor[&current];
            path.push((previous, current, previous_label));
            current = previous;
        }
        path.reverse();
        Some(path)
    }
}

/// What an edge adds to a cycle's length in [`Digraph::shortest_cycle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cost {
    /// The edge is part of a longer step and adds nothing on its own.
    Free,
    /// The edge completes one step.
    Step,
}

impl Cost {
    fn units(self) -> usize {
        match self {
            Self::Free => 0,
            Self::Step => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The example from the SCC explanation: T1 → T2 → T3 → T1, and T3 → T4.
    fn example() -> Digraph<char> {
        let mut graph = Digraph::new(4);
        graph.add_edge(0, 1, 'a');
        graph.add_edge(1, 2, 'b');
        graph.add_edge(2, 0, 'c');
        graph.add_edge(2, 3, 'd');
        graph
    }

    #[test]
    fn test_cyclic_components_ignore_acyclic_nodes_and_excluded_edges() {
        let graph = example();
        assert_eq!(graph.cyclic_components(|_| true), vec![vec![0, 1, 2]]);
        assert_eq!(
            graph.cyclic_components(|label| *label != 'c'),
            Vec::<Vec<usize>>::new()
        );
    }

    #[test]
    fn test_shortest_cycle_prefers_fewer_edges() {
        let mut graph = example();
        graph.add_edge(1, 0, 'e');
        let component = graph.cyclic_components(|_| true).remove(0);
        let cycle = graph
            .shortest_cycle(0, &component, |_| Some(Cost::Step))
            .unwrap();
        assert_eq!(cycle, vec![(0, 1, &'a'), (1, 0, &'e')]);
    }

    /// Free edges do not count: a cycle with more edges but fewer steps wins.
    #[test]
    fn test_shortest_cycle_counts_only_steps() {
        let mut graph = example();
        // A three-edge route 0 → 3 → 2 → 0 whose first two edges are one step.
        graph.add_edge(0, 3, 'f');
        graph.add_edge(3, 2, 'g');
        let component = graph.cyclic_components(|_| true).remove(0);
        let cost = |label: &char| match label {
            'f' => Some(Cost::Free),
            _ => Some(Cost::Step),
        };
        let cycle = graph.shortest_cycle(0, &component, cost).unwrap();
        assert_eq!(cycle, vec![(0, 3, &'f'), (3, 2, &'g'), (2, 0, &'c')]);
    }

    /// Deep chains would overflow a recursive Tarjan.
    #[test]
    fn test_cyclic_components_on_a_long_cycle() {
        let node_count = 1_000_000;
        let mut graph = Digraph::new(node_count);
        for node in 0..node_count {
            graph.add_edge(node, (node + 1) % node_count, ());
        }
        let components = graph.cyclic_components(|()| true);
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), node_count);
    }
}
