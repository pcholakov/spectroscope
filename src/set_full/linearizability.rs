//! Exact linearizability check for grow-only set histories.
//!
//! An element's add takes effect at a single point: no earlier than the first
//! invocation of a non-failed add of it, and no later than the first `Ok`
//! completion. A read takes effect between its invocation and completion. The
//! history is linearizable iff these points can be chosen so every read returns
//! exactly the elements whose adds precede it, which holds iff:
//!
//! 1. Distinct read results form a chain under `⊆` (the set only grows, so all
//!    readers see its states in one order), and
//! 2. no event in the chain's ordering completed before an event it must
//!    follow was invoked.
//!
//! The chain orders events into layers: elements first observed at chain step
//! `i`, then the reads that returned step `i`, ..., then elements with an `Ok`
//! add that no read observed. Every event must follow every event in an earlier
//! layer, through at most one intermediate event, so condition 2 is a sweep that
//! tracks the latest invocation seen in earlier layers.

use std::hash::Hash;

use ahash::{HashMap, HashMapExt};

use crate::history::{History, OpFn, OpType, OpValue};

/// An event in a set history that takes effect at a single point in time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event<T> {
    /// The point at which the element first became part of the set.
    Add(T),
    /// A successful read, identified by its invocation and completion indices.
    Read {
        invoke_index: usize,
        complete_index: usize,
    },
}

/// Why one event must take effect before another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// `before` completed (at `completed_index`) before `after` was invoked
    /// (at `invoked_index`). For an add, these are its earliest `Ok` completion
    /// and its earliest non-failed invocation.
    RealTime {
        completed_index: usize,
        invoked_index: usize,
    },
    /// `after` is a read that returned `before`'s element.
    Observed,
    /// `before` is a read that did not return `after`'s element.
    Missed,
}

/// A required ordering between two events.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Precedence<T> {
    pub before: Event<T>,
    pub after: Event<T>,
    pub reason: Reason,
}

/// Orderings that cannot all hold, proving a history is not linearizable.
///
/// Each edge's `after` is the next edge's `before`, and the last edge's `after`
/// is the first edge's `before`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrecedenceCycle<T> {
    pub edges: Vec<Precedence<T>>,
}

/// Returns a cycle proving `history` is not linearizable, or `None` if no
/// ordering forbids it. Elements observed without a non-failed add impose no
/// ordering here; the caller rejects them separately.
pub(super) fn find_violation<T>(history: &History<T>) -> Option<PrecedenceCycle<T>>
where
    T: Clone + Eq + Hash + Ord,
{
    let model = Model::new(history);
    model
        .find_divergent_reads()
        .or_else(|| model.find_real_time_violation())
}

/// When an element's add can take effect, as op indices. `start` is `None` when
/// no non-failed add was invoked, and `end` is `None` when no add completed `Ok`.
#[derive(Debug, Clone, Copy, Default)]
struct AddWindow {
    start: Option<usize>,
    end: Option<usize>,
}

#[derive(Debug)]
struct Read {
    invoke_index: usize,
    complete_index: usize,
}

/// Reads that returned the same set state, and the elements that state adds
/// to the previous one in the chain.
#[derive(Debug)]
struct ChainStep<'a, T> {
    state: Vec<&'a T>,
    reads: Vec<usize>,
    fresh: Vec<&'a T>,
}

/// An event in the layered order: an element's add, or a read by its position
/// in [`Model::reads`].
#[derive(Debug)]
enum Node<'a, T> {
    Add(&'a T),
    Read(usize),
}

// Derived impls would require `T: Copy`, but a node only holds a reference.
impl<T> Clone for Node<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Node<'_, T> {}

struct Model<'a, T> {
    windows: HashMap<&'a T, AddWindow>,
    reads: Vec<Read>,
    /// Distinct read results, ordered by size.
    steps: Vec<ChainStep<'a, T>>,
}

impl<'a, T> Model<'a, T>
where
    T: Clone + Eq + Hash + Ord,
{
    fn new(history: &'a History<T>) -> Self {
        let mut windows: HashMap<&T, AddWindow> = HashMap::new();
        let mut reads = Vec::new();
        let mut states: HashMap<Vec<&T>, Vec<usize>> = HashMap::new();

        for (pos, op) in history.ops().iter().enumerate() {
            match (op.f, op.op_type, &op.value) {
                (OpFn::Add, OpType::Invoke, OpValue::Single(element)) => {
                    let window = windows.entry(element).or_default();
                    match history.completion(pos).map(|c| (c.op_type, c.index)) {
                        Some((OpType::Fail, _)) => {}
                        outcome => {
                            window.start = window.start.or(Some(op.index));
                            if let Some((OpType::Ok, completed)) = outcome {
                                window.end =
                                    Some(window.end.map_or(completed, |e| e.min(completed)));
                            }
                        }
                    }
                }
                (OpFn::Read, OpType::Ok, value) => {
                    let mut state: Vec<&T> = match value {
                        OpValue::Set(s) => s.iter().collect(),
                        OpValue::Vec(v) => v.iter().collect(),
                        _ => unreachable!("History validates read results"),
                    };
                    state.sort_unstable();
                    state.dedup();
                    let invocation = history
                        .invocation(pos)
                        .expect("History pairs every completion with its invocation");
                    states.entry(state).or_default().push(reads.len());
                    reads.push(Read {
                        invoke_index: invocation.index,
                        complete_index: op.index,
                    });
                }
                _ => {}
            }
        }

        let mut steps: Vec<ChainStep<T>> = states
            .into_iter()
            .map(|(state, reads)| ChainStep {
                state,
                reads,
                fresh: Vec::new(),
            })
            .collect();
        steps.sort_unstable_by(|a, b| (a.state.len(), &a.state).cmp(&(b.state.len(), &b.state)));
        for i in 0..steps.len() {
            let previous: &[&T] = if i == 0 { &[] } else { &steps[i - 1].state };
            steps[i].fresh = sorted_difference(&steps[i].state, previous);
        }

        Self {
            windows,
            reads,
            steps,
        }
    }

    /// Two reads whose results are not `⊆`-comparable force each one's extra
    /// element to be added both before and after the other.
    fn find_divergent_reads(&self) -> Option<PrecedenceCycle<T>> {
        let (smaller, larger) = self
            .steps
            .windows(2)
            .map(|pair| (&pair[0], &pair[1]))
            .find(|(smaller, larger)| {
                !sorted_difference(&smaller.state, &larger.state).is_empty()
            })?;
        let only_smaller = sorted_difference(&smaller.state, &larger.state)[0];
        let only_larger = sorted_difference(&larger.state, &smaller.state)[0];
        let first = self.read_event(smaller.reads[0]);
        let second = self.read_event(larger.reads[0]);
        Some(PrecedenceCycle {
            edges: vec![
                edge(
                    Event::Add(only_smaller.clone()),
                    first.clone(),
                    Reason::Observed,
                ),
                edge(first, Event::Add(only_larger.clone()), Reason::Missed),
                edge(
                    Event::Add(only_larger.clone()),
                    second.clone(),
                    Reason::Observed,
                ),
                edge(second, Event::Add(only_smaller.clone()), Reason::Missed),
            ],
        })
    }

    /// Sweeps the chain's layers, looking for an event that completed before
    /// some event in an earlier layer was invoked. Assumes the reads form a chain.
    fn find_real_time_violation(&self) -> Option<PrecedenceCycle<T>> {
        let largest: &[&T] = self.steps.last().map_or(&[], |s| &s.state);
        let mut unobserved: Vec<&T> = self
            .windows
            .iter()
            .filter(|(element, window)| {
                window.end.is_some() && largest.binary_search(element).is_err()
            })
            .map(|(element, _)| *element)
            .collect();
        unobserved.sort_unstable();

        // The latest invocation among earlier layers, with its node and chain step.
        let mut latest: Option<(usize, Node<T>, usize)> = None;
        let layers = self.steps.iter().enumerate().flat_map(|(i, step)| {
            [
                (
                    i,
                    step.fresh.iter().map(|e| Node::Add(*e)).collect::<Vec<_>>(),
                ),
                (i, step.reads.iter().map(|r| Node::Read(*r)).collect()),
            ]
        });
        let final_layer = (
            self.steps.len(),
            unobserved.into_iter().map(Node::Add).collect(),
        );

        for (step, nodes) in layers.chain(std::iter::once(final_layer)) {
            for &node in &nodes {
                let (_, end) = self.bounds(node);
                if let (Some((invoked, earlier, earlier_step)), Some(completed)) = (latest, end)
                    && invoked > completed
                {
                    return Some(self.witness(earlier, earlier_step, node, completed, invoked));
                }
            }
            for &node in &nodes {
                if let (Some(start), _) = self.bounds(node)
                    && latest.is_none_or(|(invoked, _, _)| start > invoked)
                {
                    latest = Some((start, node, step));
                }
            }
        }
        None
    }

    /// Builds the cycle through `earlier` (from chain step `earlier_step`) and
    /// `later`, which completed at `completed` before `earlier` was invoked at `invoked`.
    fn witness(
        &self,
        earlier: Node<T>,
        earlier_step: usize,
        later: Node<T>,
        completed: usize,
        invoked: usize,
    ) -> PrecedenceCycle<T> {
        let mut edges = match (earlier, later) {
            // The read observed an element whose add had not been invoked yet.
            (Node::Add(_), Node::Read(_)) => {
                vec![edge(
                    self.event(earlier),
                    self.event(later),
                    Reason::Observed,
                )]
            }
            // The read missed an element whose add had already completed.
            (Node::Read(_), Node::Add(_)) => {
                vec![edge(self.event(earlier), self.event(later), Reason::Missed)]
            }
            // The later read observed an element the earlier one missed.
            (Node::Read(_), Node::Read(_)) => {
                let element = Event::Add(self.steps[earlier_step + 1].fresh[0].clone());
                vec![
                    edge(self.event(earlier), element.clone(), Reason::Missed),
                    edge(element, self.event(later), Reason::Observed),
                ]
            }
            // A read observed the earlier element but missed the later one.
            (Node::Add(_), Node::Add(_)) => {
                let read = self.read_event(self.steps[earlier_step].reads[0]);
                vec![
                    edge(self.event(earlier), read.clone(), Reason::Observed),
                    edge(read, self.event(later), Reason::Missed),
                ]
            }
        };
        edges.push(edge(
            self.event(later),
            self.event(earlier),
            Reason::RealTime {
                completed_index: completed,
                invoked_index: invoked,
            },
        ));
        PrecedenceCycle { edges }
    }

    /// The node's earliest invocation and completion; `None` means unbounded.
    fn bounds(&self, node: Node<T>) -> (Option<usize>, Option<usize>) {
        match node {
            Node::Add(element) => {
                let window = self.windows.get(element).copied().unwrap_or_default();
                (window.start, window.end)
            }
            Node::Read(r) => (
                Some(self.reads[r].invoke_index),
                Some(self.reads[r].complete_index),
            ),
        }
    }

    fn event(&self, node: Node<T>) -> Event<T> {
        match node {
            Node::Add(element) => Event::Add(element.clone()),
            Node::Read(r) => self.read_event(r),
        }
    }

    fn read_event(&self, read: usize) -> Event<T> {
        Event::Read {
            invoke_index: self.reads[read].invoke_index,
            complete_index: self.reads[read].complete_index,
        }
    }
}

fn edge<T>(before: Event<T>, after: Event<T>, reason: Reason) -> Precedence<T> {
    Precedence {
        before,
        after,
        reason,
    }
}

/// Elements of sorted `a` that are not in sorted `b`.
fn sorted_difference<'a, T: Ord>(a: &[&'a T], b: &[&'a T]) -> Vec<&'a T> {
    let mut rest = b.iter().copied().peekable();
    a.iter()
        .copied()
        .filter(|&x| {
            while rest.next_if(|&y| y < x).is_some() {}
            rest.peek().is_none_or(|&y| y != x)
        })
        .collect()
}
