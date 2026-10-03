//! The dependency graph of a list-append history, and the cycles that rule out
//! a consistency model.
//!
//! Nodes are the transactions that may have committed: committed ones, and
//! indeterminate ones. Failed transactions have no effects and are left out.
//! Edges follow Elle's inference for list-append:
//!
//! - A key's version order is its longest committed read. An append no read
//!   observed must come after every observed value, since a read returns every
//!   value appended before its last one.
//! - `ww`: the writer of the value just before an append precedes the appender.
//! - `wr`: the writer of a read's last value precedes the reader.
//! - `rw`: a reader of the state just before an append precedes the appender.
//!
//! Appends no read observed all follow a key's final observed state, so every
//! transaction that read or wrote that state precedes every such appender.
//! Rather than one edge per pair, these orderings go through shared nodes: a
//! hub from the transactions that are only sources to every such appender, a
//! second hub from the sources that are also such appenders to the rest, and a
//! ring among those. Every transaction reaches exactly the transactions it
//! would through direct edges, never itself, with O(transactions) edges.
//!
//! Process order links consecutive transactions of one process. Real-time order
//! uses a chain of time nodes, one per committed completion in history order:
//! each completion links to its time node, each time node to the next, and the
//! latest time node before an invocation links to the invoked transaction. So
//! a transaction reaches another through time nodes exactly when it completed
//! before the other was invoked, with O(transactions) edges. An indeterminate
//! transaction never completes, so real time gives it no outgoing edges.
//!
//! Each strongly connected component is reported as one shortest cycle, named by
//! the dependencies on it. Elle instead searches each component for cycles of
//! its weakest anomaly first, so it may name the same component differently.

use std::collections::BTreeMap;

use super::txns::{Outcome, Txns};
use super::{MicroOp, TxnRef};
use crate::graph::{Cost, Digraph, PathEdge};
use crate::history::Pid;

/// A consistency model the checker can verify. Variants are ordered from
/// weakest to strongest; each allows a subset of the histories the previous
/// one allows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConsistencyModel {
    /// Transactions appear to execute in some total order consistent with
    /// write-write, write-read and read-write dependencies.
    Serializable,
    /// Serializable, with each process's transactions in the order it ran them.
    StrongSessionSerializable,
    /// Serializable, with every transaction ordered after all transactions that
    /// completed before it was invoked.
    StrictSerializable,
}

/// Why one transaction must precede another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dependency<K, V> {
    /// `ww`: `to` appended `appended` to `key` after `from` appended
    /// `previous`. If a read observed `appended`, `previous` is the value just
    /// before it in the order reads revealed. If none did, `previous` is the
    /// key's last observed value, and other unobserved appends may lie between.
    WriteWrite {
        /// The key.
        key: K,
        /// The value `from` appended.
        previous: V,
        /// The value `to` appended.
        appended: V,
    },
    /// `wr`: `to` read `key`, and the last value it saw was `from`'s append of
    /// `value`.
    WriteRead {
        /// The key.
        key: K,
        /// `from`'s value, last in the list `to` read.
        value: V,
    },
    /// `rw`: `from` read `key` in a state that `to`'s append of `appended`
    /// came after, so it did not observe that append. `read` is the last value
    /// `from` saw, or `None` if it read the empty list. If a read observed
    /// `appended`, that state is the one just before it; if none did, it is the
    /// key's final observed state.
    ReadWrite {
        /// The key.
        key: K,
        /// The last value `from` read.
        read: Option<V>,
        /// The value `to` appended.
        appended: V,
    },
    /// Both ran on the same process, `from` first.
    Process,
    /// `from` completed (at op `completed_index`) before `to` was invoked (at op
    /// `invoked_index`).
    RealTime {
        /// Index of `from`'s completion.
        completed_index: usize,
        /// Index of `to`'s invocation.
        invoked_index: usize,
    },
}

/// One edge of a [`Cycle`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step<K, V> {
    /// The transaction that must come first.
    pub from: TxnRef,
    /// The transaction that must come second.
    pub to: TxnRef,
    /// What forces the order.
    pub dependency: Dependency<K, V>,
}

/// Dependencies that cannot all hold. Each step's `to` is the next step's
/// `from`, and the last step's `to` is the first step's `from`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cycle<K, V> {
    /// The anomaly the steps' dependencies make this cycle.
    pub anomaly: CycleAnomaly,
    /// The steps, in cycle order.
    pub steps: Vec<Step<K, V>>,
}

/// Elle's classification of a dependency cycle, after Adya's phenomena. All
/// dependencies here are on items, not predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CycleAnomaly {
    /// The class the data dependencies alone determine.
    pub data: DataCycle,
    /// The strongest ordering edge in the cycle, if it has any: real time if
    /// any step is real time, otherwise process if any step is.
    pub order: Option<OrderEdge>,
}

/// A cycle's class by its write-write, write-read and read-write dependencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataCycle {
    /// Write cycle: only write-write dependencies.
    G0,
    /// Circular information flow: write-read dependencies, possibly with
    /// write-write ones, and no read-write dependency.
    G1c,
    /// Single anti-dependency: exactly one read-write dependency.
    GSingle,
    /// Two or more read-write dependencies, none consecutive in the cycle.
    GNonadjacent,
    /// Two or more read-write dependencies, some consecutive.
    G2,
}

/// An ordering dependency that is not a data dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderEdge {
    /// [`Dependency::Process`].
    Process,
    /// [`Dependency::RealTime`].
    RealTime,
}

impl CycleAnomaly {
    /// Elle's name for the anomaly, such as `"G-single-item-realtime"`.
    #[must_use]
    pub fn name(&self) -> &'static str {
        use DataCycle::{G0, G1c, G2, GNonadjacent, GSingle};
        use OrderEdge::{Process, RealTime};
        match (self.data, self.order) {
            (G0, None) => "G0",
            (G0, Some(Process)) => "G0-process",
            (G0, Some(RealTime)) => "G0-realtime",
            (G1c, None) => "G1c",
            (G1c, Some(Process)) => "G1c-process",
            (G1c, Some(RealTime)) => "G1c-realtime",
            (GSingle, None) => "G-single-item",
            (GSingle, Some(Process)) => "G-single-item-process",
            (GSingle, Some(RealTime)) => "G-single-item-realtime",
            (GNonadjacent, None) => "G-nonadjacent-item",
            (GNonadjacent, Some(Process)) => "G-nonadjacent-item-process",
            (GNonadjacent, Some(RealTime)) => "G-nonadjacent-item-realtime",
            (G2, None) => "G2-item",
            (G2, Some(Process)) => "G2-item-process",
            (G2, Some(RealTime)) => "G2-item-realtime",
        }
    }

    /// Classifies a cycle's steps as Elle does.
    fn of<K, V>(steps: &[Step<K, V>]) -> Self {
        let count = |kind: fn(&Dependency<K, V>) -> bool| {
            steps.iter().filter(|step| kind(&step.dependency)).count()
        };
        let is_read_write =
            |dependency: &Dependency<K, V>| matches!(dependency, Dependency::ReadWrite { .. });
        let read_writes = count(is_read_write);
        let write_reads = count(|d| matches!(d, Dependency::WriteRead { .. }));
        // Process and real-time order both follow invocation order, so a cycle
        // always contains a data dependency; with no rw or wr, that is ww.
        assert!(
            read_writes + write_reads + count(|d| matches!(d, Dependency::WriteWrite { .. })) > 0,
            "a cycle needs a data dependency"
        );
        let adjacent_read_writes = steps
            .iter()
            .zip(steps.iter().cycle().skip(1))
            .any(|(a, b)| is_read_write(&a.dependency) && is_read_write(&b.dependency));
        let data = match read_writes {
            1 => DataCycle::GSingle,
            2.. if adjacent_read_writes => DataCycle::G2,
            2.. => DataCycle::GNonadjacent,
            0 if write_reads > 0 => DataCycle::G1c,
            0 => DataCycle::G0,
        };
        let order = if count(|d| matches!(d, Dependency::RealTime { .. })) > 0 {
            Some(OrderEdge::RealTime)
        } else if count(|d| matches!(d, Dependency::Process)) > 0 {
            Some(OrderEdge::Process)
        } else {
            None
        };
        Self { data, order }
    }
}

/// A graph edge. Real time is a chain through time nodes; a [`Cycle`] reports
/// each run through them as one [`Dependency::RealTime`] step.
#[derive(Debug)]
enum Edge<'a, K, V> {
    WriteWrite {
        key: &'a K,
        previous: &'a V,
        appended: &'a V,
    },
    WriteRead {
        key: &'a K,
        value: &'a V,
    },
    ReadWrite {
        key: &'a K,
        read: Option<&'a V>,
        appended: &'a V,
    },
    Process,
    /// From a transaction that read or wrote a key's final observed state to a
    /// hub standing for "after that state".
    FinalState {
        role: FinalRole<'a, V>,
    },
    /// From a final-state hub to a transaction whose append no read observed.
    AfterFinalState {
        key: &'a K,
        appended: &'a V,
    },
    /// From a transaction to the time node of its completion at `index`.
    Completion {
        index: usize,
    },
    /// From one time node to the next.
    Tick,
    /// From the latest time node before an invocation at `index` to its transaction.
    Invocation {
        index: usize,
    },
}

/// How a transaction relates to a key's final observed state.
#[derive(Debug)]
enum FinalRole<'a, V> {
    /// It read the state, whose last value is `Some` or which is empty.
    Reader(Option<&'a V>),
    /// It appended the state's last value.
    Writer(&'a V),
}

// Derived impls would require `V: Copy`, but a role only holds references.
impl<V> Clone for FinalRole<'_, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V> Copy for FinalRole<'_, V> {}

impl<'a, V> FinalRole<'a, V> {
    /// The dependency from a transaction in this role to one that appended
    /// `appended` after the state.
    fn edge<K>(&self, key: &'a K, appended: &'a V) -> Edge<'a, K, V> {
        match *self {
            Self::Reader(read) => Edge::ReadWrite {
                key,
                read,
                appended,
            },
            Self::Writer(previous) => Edge::WriteWrite {
                key,
                previous,
                appended,
            },
        }
    }

    /// The reported dependency from a transaction in this role to one that
    /// appended `appended` to `key` after the state.
    fn dependency<K: Clone>(self, key: &K, appended: &V) -> Dependency<K, V>
    where
        V: Clone,
    {
        match self {
            Self::Reader(read) => Dependency::ReadWrite {
                key: key.clone(),
                read: read.cloned(),
                appended: appended.clone(),
            },
            Self::Writer(previous) => Dependency::WriteWrite {
                key: key.clone(),
                previous: previous.clone(),
                appended: appended.clone(),
            },
        }
    }
}

impl<K, V> Edge<'_, K, V> {
    /// What the edge adds to a cycle's length under `model`, or `None` if the
    /// model does not count it. A run through time nodes is one step.
    fn cost(&self, model: ConsistencyModel) -> Option<Cost> {
        if !self.counts_for(model) {
            return None;
        }
        match self {
            Self::Completion { .. } | Self::Tick | Self::FinalState { .. } => Some(Cost::Free),
            _ => Some(Cost::Step),
        }
    }

    fn counts_for(&self, model: ConsistencyModel) -> bool {
        match self {
            Self::WriteWrite { .. }
            | Self::WriteRead { .. }
            | Self::ReadWrite { .. }
            | Self::FinalState { .. }
            | Self::AfterFinalState { .. } => true,
            Self::Process => model == ConsistencyModel::StrongSessionSerializable,
            Self::Completion { .. } | Self::Tick | Self::Invocation { .. } => {
                model == ConsistencyModel::StrictSerializable
            }
        }
    }
}

/// One shortest cycle per strongly connected component of the dependency graph
/// under `model`, through the component's earliest-invoked transaction.
pub(super) fn cycles<K, V>(txns: &Txns<'_, K, V>, model: ConsistencyModel) -> Vec<Cycle<K, V>>
where
    K: Clone + Ord,
    V: Clone + Ord,
{
    let graph = build(txns);
    let include = |edge: &Edge<'_, K, V>| edge.counts_for(model);
    graph
        .cyclic_components(include)
        .into_iter()
        .filter_map(|component| {
            // Time nodes are numbered after transactions, and the time chain
            // alone is acyclic, so a cyclic component starts with a transaction.
            let cost = |edge: &Edge<'_, K, V>| edge.cost(model);
            let path = graph.shortest_cycle(component[0], &component, cost)?;
            Some(explain(txns, &path))
        })
        .collect()
}

/// Nodes `0..txns.records.len()` are transactions; time nodes follow.
fn build<'a, K: Ord, V: Ord>(txns: &Txns<'a, K, V>) -> Digraph<Edge<'a, K, V>> {
    let mut graph = Digraph::new(txns.records.len());
    add_data_dependencies(txns, &mut graph);
    add_process_order(txns, &mut graph);
    add_real_time_order(txns, &mut graph);
    graph
}

/// Each key's version order and readers, as the dependency inference sees them.
struct VersionIndex<'a, K, V> {
    /// Each key's longest committed read.
    orders: BTreeMap<&'a K, &'a [V]>,
    /// Position of each observed value in its key's order. With duplicate
    /// elements, the first occurrence.
    positions: BTreeMap<(&'a K, &'a V), usize>,
    /// Committed readers of each key, grouped by the last value they read, each
    /// listed once per group in invocation order.
    readers: BTreeMap<(&'a K, Option<&'a V>), Vec<usize>>,
}

impl<'a, K: Ord, V: Ord> VersionIndex<'a, K, V> {
    fn new(txns: &Txns<'a, K, V>) -> Self {
        let orders: BTreeMap<&K, &[V]> = txns
            .observed_lists
            .iter()
            .filter_map(|(key, lists)| Some((*key, lists.last()?.0)))
            .collect();
        let mut positions = BTreeMap::new();
        for (key, order) in &orders {
            for (index, value) in order.iter().enumerate() {
                positions.entry((*key, value)).or_insert(index);
            }
        }
        let mut readers: BTreeMap<(&K, Option<&V>), Vec<usize>> = BTreeMap::new();
        for (position, record) in txns.records.iter().enumerate() {
            for micro_op in record.micro_ops {
                if let MicroOp::Read {
                    key,
                    list: Some(list),
                } = micro_op
                {
                    let group = readers.entry((key, list.last())).or_default();
                    if group.last() != Some(&position) {
                        group.push(position);
                    }
                }
            }
        }
        Self {
            orders,
            positions,
            readers,
        }
    }

    fn readers_of(&self, key: &'a K, last: Option<&'a V>) -> &[usize] {
        self.readers.get(&(key, last)).map_or(&[], Vec::as_slice)
    }
}

/// The transaction that appended `value` to `key`, if it may have committed.
fn possible_appender<K: Ord, V: Ord>(txns: &Txns<'_, K, V>, key: &K, value: &V) -> Option<usize> {
    txns.appends
        .get(&(key, value))
        .map(|append| append.txn)
        .filter(|&txn| txns.records[txn].outcome != Outcome::Aborted)
}

fn add_data_dependencies<'a, K: Ord, V: Ord>(
    txns: &Txns<'a, K, V>,
    graph: &mut Digraph<Edge<'a, K, V>>,
) {
    let index = VersionIndex::new(txns);
    // Per key, each transaction with an append no read observed, and its
    // first such value.
    let mut unobserved: BTreeMap<&K, BTreeMap<usize, &V>> = BTreeMap::new();

    for (position, record) in txns.records.iter().enumerate() {
        if record.outcome == Outcome::Aborted {
            continue;
        }
        for micro_op in record.micro_ops {
            match micro_op {
                MicroOp::Read {
                    key,
                    list: Some(list),
                } => {
                    if let Some(value) = list.last()
                        && let Some(writer) = possible_appender(txns, key, value)
                        && writer != position
                    {
                        graph.add_edge(writer, position, Edge::WriteRead { key, value });
                    }
                }
                MicroOp::Append { key, value } => {
                    let Some(&version) = index.positions.get(&(key, value)) else {
                        unobserved
                            .entry(key)
                            .or_default()
                            .entry(position)
                            .or_insert(value);
                        continue;
                    };
                    let previous = version.checked_sub(1).map(|i| &index.orders[key][i]);
                    if let Some(previous) = previous
                        && let Some(writer) = possible_appender(txns, key, previous)
                        && writer != position
                    {
                        let role = FinalRole::Writer(previous);
                        graph.add_edge(writer, position, role.edge(key, value));
                    }
                    for &reader in index.readers_of(key, previous) {
                        if reader != position {
                            let role = FinalRole::Reader(previous);
                            graph.add_edge(reader, position, role.edge(key, value));
                        }
                    }
                }
                MicroOp::Read { list: None, .. } => {}
            }
        }
    }

    for (key, appenders) in unobserved {
        add_after_final_state(txns, &index, graph, key, &appenders);
    }
}

/// Orders every transaction that read or wrote `key`'s final observed state
/// before every transaction in `appenders`, other than itself, through hubs
/// and a ring; see the module docs.
fn add_after_final_state<'a, K: Ord, V: Ord>(
    txns: &Txns<'a, K, V>,
    index: &VersionIndex<'a, K, V>,
    graph: &mut Digraph<Edge<'a, K, V>>,
    key: &'a K,
    appenders: &BTreeMap<usize, &'a V>,
) {
    let last = index.orders.get(key).and_then(|order| order.last());
    let mut sources: BTreeMap<usize, FinalRole<'a, V>> = index
        .readers_of(key, last)
        .iter()
        .map(|&reader| (reader, FinalRole::Reader(last)))
        .collect();
    // A transaction that both wrote and read the state is reported as its writer.
    if let Some(last) = last
        && let Some(writer) = possible_appender(txns, key, last)
    {
        sources.insert(writer, FinalRole::Writer(last));
    }
    let (both, only_sources): (Vec<_>, Vec<_>) = sources
        .iter()
        .partition(|(source, _)| appenders.contains_key(source));
    let only_appenders: Vec<(&usize, &&V)> = appenders
        .iter()
        .filter(|(appender, _)| !sources.contains_key(appender))
        .collect();

    if !only_sources.is_empty() {
        let all_appenders = appenders.iter().map(|(&t, &v)| (t, v));
        add_hub(graph, key, &only_sources, all_appenders);
    }
    if !both.is_empty() && !only_appenders.is_empty() {
        let rest = only_appenders.iter().map(|&(&t, &v)| (t, v));
        add_hub(graph, key, &both, rest);
    }
    if both.len() > 1 {
        for (i, &(&source, role)) in both.iter().enumerate() {
            let (&next, _) = both[(i + 1) % both.len()];
            graph.add_edge(source, next, role.edge(key, appenders[&next]));
        }
    }
}

/// A new node with an edge from every source and to every appender.
fn add_hub<'a, K, V>(
    graph: &mut Digraph<Edge<'a, K, V>>,
    key: &'a K,
    sources: &[(&usize, &FinalRole<'a, V>)],
    appenders: impl Iterator<Item = (usize, &'a V)>,
) {
    let hub = graph.add_node();
    for &(&source, &role) in sources {
        graph.add_edge(source, hub, Edge::FinalState { role });
    }
    for (appender, appended) in appenders {
        graph.add_edge(hub, appender, Edge::AfterFinalState { key, appended });
    }
}

fn add_process_order<K, V>(txns: &Txns<'_, K, V>, graph: &mut Digraph<Edge<'_, K, V>>) {
    let mut latest: BTreeMap<Pid, usize> = BTreeMap::new();
    for (position, record) in txns.records.iter().enumerate() {
        if record.outcome == Outcome::Aborted {
            continue;
        }
        if let Some(previous) = latest.insert(record.process, position) {
            graph.add_edge(previous, position, Edge::Process);
        }
    }
}

fn add_real_time_order<K, V>(txns: &Txns<'_, K, V>, graph: &mut Digraph<Edge<'_, K, V>>) {
    // Invocations and committed completions, in history order.
    let mut events: Vec<(usize, usize, bool)> = Vec::new();
    for (position, record) in txns.records.iter().enumerate() {
        if record.outcome == Outcome::Aborted {
            continue;
        }
        events.push((record.reference.invoke_index, position, false));
        if record.outcome == Outcome::Committed
            && let Some(index) = record.reference.complete_index
        {
            events.push((index, position, true));
        }
    }
    events.sort_unstable();

    let mut latest_time_node: Option<usize> = None;
    for (index, position, completed) in events {
        if completed {
            let time_node = graph.add_node();
            graph.add_edge(position, time_node, Edge::Completion { index });
            if let Some(previous) = latest_time_node {
                graph.add_edge(previous, time_node, Edge::Tick);
            }
            latest_time_node = Some(time_node);
        } else if let Some(time_node) = latest_time_node {
            graph.add_edge(time_node, position, Edge::Invocation { index });
        }
    }
}

/// Turns a graph cycle into steps between transactions, merging each run through
/// time nodes into one real-time step.
fn explain<K: Clone, V: Clone>(
    txns: &Txns<'_, K, V>,
    path: &[PathEdge<'_, Edge<'_, K, V>>],
) -> Cycle<K, V> {
    let reference = |node: usize| txns.records[node].reference;
    let mut steps = Vec::new();
    let mut completed: Option<(usize, usize)> = None;
    let mut final_state: Option<(usize, &FinalRole<'_, V>)> = None;
    for &(source, target, edge) in path {
        let dependency = match edge {
            Edge::WriteWrite {
                key,
                previous,
                appended,
            } => Dependency::WriteWrite {
                key: (*key).clone(),
                previous: (*previous).clone(),
                appended: (*appended).clone(),
            },
            Edge::WriteRead { key, value } => Dependency::WriteRead {
                key: (*key).clone(),
                value: (*value).clone(),
            },
            Edge::ReadWrite {
                key,
                read,
                appended,
            } => Dependency::ReadWrite {
                key: (*key).clone(),
                read: read.map(Clone::clone),
                appended: (*appended).clone(),
            },
            Edge::Process => Dependency::Process,
            Edge::FinalState { role, .. } => {
                final_state = Some((source, role));
                continue;
            }
            Edge::AfterFinalState { key, appended } => {
                let Some((from, role)) = final_state.take() else {
                    unreachable!("a final-state hub is entered only from a source");
                };
                steps.push(Step {
                    from: reference(from),
                    to: reference(target),
                    dependency: role.dependency(*key, *appended),
                });
                continue;
            }
            Edge::Completion { index } => {
                completed = Some((source, *index));
                continue;
            }
            Edge::Tick => continue,
            Edge::Invocation { index } => {
                let Some((from, completed_index)) = completed.take() else {
                    unreachable!("a time node is entered only from a completion");
                };
                steps.push(Step {
                    from: reference(from),
                    to: reference(target),
                    dependency: Dependency::RealTime {
                        completed_index,
                        invoked_index: *index,
                    },
                });
                continue;
            }
        };
        steps.push(Step {
            from: reference(source),
            to: reference(target),
            dependency,
        });
    }
    Cycle {
        anomaly: CycleAnomaly::of(&steps),
        steps,
    }
}
