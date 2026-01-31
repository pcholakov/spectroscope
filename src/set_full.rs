//! Set-full linearizability checker.
//!
//! A rigorous set analysis for histories with `:add` and `:read` operations.
//! For each element added to the set, we track:
//!
//! - **known time**: when the element was first confirmed to exist (add completed or first read observed it)
//! - **stable time**: when the element became permanently visible in all subsequent reads
//! - **lost time**: when the element disappeared after being observed
//!
//! Elements can have three outcomes:
//! - **stable**: visible in all reads after being known
//! - **lost**: was known but then disappeared
//! - **never-read**: no read began after the element was known
//!
//! When `linearizable` is enabled, an element must appear immediately in the next read
//! after being added (zero stable latency), otherwise it's considered "stale" and the
//! check fails.
//!
//! Derived from Jepsen's set-full checker, licensed under EPL-1.0.

use std::collections::BTreeMap;
use std::hash::Hash;

use ahash::{HashMap, HashMapExt};

use crate::history::{History, Op, OpFn, OpType, OpValue};

/// Options for the set-full checker.
#[derive(Debug, Clone, Default)]
pub struct SetFullOptions {
    /// If true, require zero stable latency (immediate visibility after add).
    pub linearizable: bool,
}

/// Outcome for a single element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementOutcome {
    /// Element is visible in all reads after being known.
    Stable,
    /// Element was known but then disappeared.
    Lost,
    /// No read began after the element was known.
    NeverRead,
}

/// Detailed result for a single element.
#[derive(Debug, Clone)]
pub struct ElementResult<T> {
    pub element: T,
    pub outcome: ElementOutcome,
    /// Latency in milliseconds until the element became stable (if stable).
    pub stable_latency_ms: Option<u64>,
    /// Latency in milliseconds until the element was lost (if lost).
    pub lost_latency_ms: Option<u64>,
}

/// Validity status of the check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Validity {
    Valid,
    Invalid,
    Unknown,
}

/// Result of the set-full check.
#[derive(Debug, Clone)]
pub struct SetFullResult<T> {
    pub valid: Validity,
    pub attempt_count: usize,
    pub stable_count: usize,
    pub lost_count: usize,
    pub never_read_count: usize,
    pub stale_count: usize,
    pub duplicated_count: usize,
    /// Elements that were lost.
    pub lost: Vec<T>,
    /// Elements that were never read.
    pub never_read: Vec<T>,
    /// Elements that were stale (non-zero stable latency).
    pub stale: Vec<T>,
    /// Elements with duplicates and their max multiplicity.
    pub duplicated: BTreeMap<T, usize>,
    /// Latency percentiles for stable elements (0, 0.5, 0.95, 0.99, 1.0) in ms.
    pub stable_latencies: Option<BTreeMap<String, u64>>,
    /// Latency percentiles for lost elements in ms.
    pub lost_latencies: Option<BTreeMap<String, u64>>,
}

/// Tracks the state of a single element during analysis.
#[derive(Debug, Clone)]
struct ElementState<T> {
    element: T,
    /// The operation that first confirmed this element exists.
    known: Option<OpRef>,
    /// The most recent read invocation that observed this element.
    last_present: Option<OpRef>,
    /// The most recent read invocation that did not observe this element.
    last_absent: Option<OpRef>,
}

/// Reference to an operation (just the fields we need).
#[derive(Debug, Clone)]
struct OpRef {
    index: usize,
    time: Option<u64>,
}

impl<T: Clone> ElementState<T> {
    fn new(element: T) -> Self {
        Self {
            element,
            known: None,
            last_present: None,
            last_absent: None,
        }
    }

    /// Process an add completion.
    fn on_add_ok(&mut self, op: &Op<T>) {
        if self.known.is_none() {
            self.known = Some(OpRef {
                index: op.index,
                time: op.time,
            });
        }
    }

    /// Process a read that observed this element.
    fn on_read_present(&mut self, inv: &Op<T>, completion: &Op<T>) {
        // If we see it in a read before the add completes, we know it exists
        if self.known.is_none() {
            self.known = Some(OpRef {
                index: completion.index,
                time: completion.time,
            });
        }

        // Track the most recent read invocation that saw us
        if self
            .last_present
            .as_ref()
            .is_none_or(|lp| lp.index < inv.index)
        {
            self.last_present = Some(OpRef {
                index: inv.index,
                time: inv.time,
            });
        }
    }

    /// Process a read that did not observe this element.
    fn on_read_absent(&mut self, inv: &Op<T>) {
        if self
            .last_absent
            .as_ref()
            .is_none_or(|la| la.index < inv.index)
        {
            self.last_absent = Some(OpRef {
                index: inv.index,
                time: inv.time,
            });
        }
    }

    /// Compute the final result for this element.
    fn into_result(self) -> ElementResult<T> {
        let last_present_idx = self.last_present.as_ref().map_or(-1, |p| p.index as i64);
        let last_absent_idx = self.last_absent.as_ref().map_or(-1, |a| a.index as i64);
        let known_idx = self.known.as_ref().map_or(-1, |k| k.index as i64);

        // Stable if we have a present observation more recent than any absent observation
        let stable = self.last_present.is_some() && last_present_idx > last_absent_idx;

        // Lost if:
        // 1. Element was known
        // 2. Most recent observation is absent (not present)
        // 3. The absent observation is after the known time
        let lost = self.known.is_some()
            && self.last_absent.is_some()
            && last_absent_idx > last_present_idx
            && last_absent_idx > known_idx;

        let outcome = if stable {
            ElementOutcome::Stable
        } else if lost {
            ElementOutcome::Lost
        } else {
            ElementOutcome::NeverRead
        };

        // Calculate latencies
        let known_time = self.known.as_ref().and_then(|k| k.time);

        let stable_latency_ms = if stable {
            let stable_time = if let Some(ref la) = self.last_absent {
                la.time.map(|t| t + 1)
            } else {
                Some(0)
            };
            match (stable_time, known_time) {
                (Some(st), Some(kt)) => Some(st.saturating_sub(kt) / 1_000_000),
                _ => Some(0),
            }
        } else {
            None
        };

        let lost_latency_ms = if lost {
            let lost_time = if let Some(ref lp) = self.last_present {
                lp.time.map(|t| t + 1)
            } else {
                Some(0)
            };
            match (lost_time, known_time) {
                (Some(lt), Some(kt)) => Some(lt.saturating_sub(kt) / 1_000_000),
                _ => Some(0),
            }
        } else {
            None
        };

        ElementResult {
            element: self.element,
            outcome,
            stable_latency_ms,
            lost_latency_ms,
        }
    }
}

/// The set-full checker.
#[derive(Debug, Clone, Default)]
pub struct SetFullChecker {
    pub options: SetFullOptions,
}

impl SetFullChecker {
    pub fn new(options: SetFullOptions) -> Self {
        Self { options }
    }

    pub fn linearizable() -> Self {
        Self::new(SetFullOptions { linearizable: true })
    }

    /// Check a history for set consistency.
    pub fn check<T>(&self, history: &History<T>) -> SetFullResult<T>
    where
        T: Clone + Eq + Hash + Ord,
    {
        let mut elements: HashMap<T, ElementState<T>> = HashMap::new();
        let duplicates: BTreeMap<T, usize> = BTreeMap::new();

        for op in history.ops() {
            match op.f {
                OpFn::Add => {
                    if let OpValue::Single(ref v) = op.value {
                        match op.op_type {
                            OpType::Invoke => {
                                // Start tracking this element
                                elements
                                    .entry(v.clone())
                                    .or_insert_with(|| ElementState::new(v.clone()));
                            }
                            OpType::Ok => {
                                // Mark as known
                                if let Some(state) = elements.get_mut(v) {
                                    state.on_add_ok(op);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                OpFn::Read => {
                    if op.op_type == OpType::Ok {
                        if let OpValue::Set(ref read_set) = op.value {
                            // Find the invocation for this read
                            let inv = match history.invocation(op) {
                                Some(i) => i,
                                None => continue,
                            };

                            // Check for duplicates (would need to track multiplicities
                            // in the actual read value - simplified here)
                            // In a real implementation, we'd receive a Vec and count duplicates

                            // Update all tracked elements
                            for (elem, state) in elements.iter_mut() {
                                if read_set.contains(elem) {
                                    state.on_read_present(inv, op);
                                } else {
                                    state.on_read_absent(inv);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Compute results
        let mut results: Vec<ElementResult<T>> = elements
            .into_values()
            .map(|state| state.into_result())
            .collect();
        results.sort_by(|a, b| a.element.cmp(&b.element));

        let mut stable = Vec::new();
        let mut lost = Vec::new();
        let mut never_read = Vec::new();
        let mut stable_latencies = Vec::new();
        let mut lost_latencies = Vec::new();

        for r in &results {
            match r.outcome {
                ElementOutcome::Stable => {
                    stable.push(r.element.clone());
                    if let Some(lat) = r.stable_latency_ms {
                        stable_latencies.push(lat);
                    }
                }
                ElementOutcome::Lost => {
                    lost.push(r.element.clone());
                    if let Some(lat) = r.lost_latency_ms {
                        lost_latencies.push(lat);
                    }
                }
                ElementOutcome::NeverRead => {
                    never_read.push(r.element.clone());
                }
            }
        }

        // Stale elements: stable but with non-zero latency
        let stale: Vec<T> = results
            .iter()
            .filter(|r| r.outcome == ElementOutcome::Stable && r.stable_latency_ms.unwrap_or(0) > 0)
            .map(|r| r.element.clone())
            .collect();

        // Determine validity
        let valid = if !lost.is_empty()
            || (self.options.linearizable && !stale.is_empty())
            || !duplicates.is_empty()
        {
            Validity::Invalid
        } else if stable.is_empty() {
            Validity::Unknown
        } else {
            Validity::Valid
        };

        SetFullResult {
            valid,
            attempt_count: results.len(),
            stable_count: stable.len(),
            lost_count: lost.len(),
            never_read_count: never_read.len(),
            stale_count: stale.len(),
            duplicated_count: duplicates.len(),
            lost,
            never_read,
            stale,
            duplicated: duplicates,
            stable_latencies: percentiles(&stable_latencies),
            lost_latencies: percentiles(&lost_latencies),
        }
    }
}

/// Compute percentile distribution.
fn percentiles(values: &[u64]) -> Option<BTreeMap<String, u64>> {
    if values.is_empty() {
        return None;
    }

    let mut sorted: Vec<u64> = values.to_vec();
    sorted.sort_unstable();

    let n = sorted.len();
    let points = [
        (0.0, "0"),
        (0.5, "0.5"),
        (0.95, "0.95"),
        (0.99, "0.99"),
        (1.0, "1"),
    ];

    let mut result = BTreeMap::new();
    for (p, label) in points {
        let idx = ((n as f64 * p).floor() as usize).min(n - 1);
        result.insert(label.to_string(), sorted[idx]);
    }

    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::{HashSet, HashSetExt};

    #[test]
    fn test_simple_add_read_stable() {
        // Add 1, then read and see it
        let mut history = History::new();
        history.push(Op {
            index: 0,
            op_type: OpType::Invoke,
            f: OpFn::Add,
            value: OpValue::Single(1),
            time: Some(0),
            process: 0,
        });
        history.push(Op {
            index: 1,
            op_type: OpType::Ok,
            f: OpFn::Add,
            value: OpValue::Single(1),
            time: Some(1_000_000_000),
            process: 0,
        });
        history.push(Op {
            index: 2,
            op_type: OpType::Invoke,
            f: OpFn::Read,
            value: OpValue::None,
            time: Some(2_000_000_000),
            process: 0,
        });
        history.push(Op {
            index: 3,
            op_type: OpType::Ok,
            f: OpFn::Read,
            value: OpValue::Set(HashSet::from_iter([1])),
            time: Some(3_000_000_000),
            process: 0,
        });

        let checker = SetFullChecker::linearizable();
        let result = checker.check(&history);

        assert_eq!(result.valid, Validity::Valid);
        assert_eq!(result.stable_count, 1);
        assert_eq!(result.lost_count, 0);
    }

    #[test]
    fn test_lost_element() {
        // Add 1, read and see it, then read and don't see it
        let mut history = History::new();
        // Add invoke
        history.push(Op {
            index: 0,
            op_type: OpType::Invoke,
            f: OpFn::Add,
            value: OpValue::Single(1),
            time: Some(0),
            process: 0,
        });
        // Add ok
        history.push(Op {
            index: 1,
            op_type: OpType::Ok,
            f: OpFn::Add,
            value: OpValue::Single(1),
            time: Some(1_000_000_000),
            process: 0,
        });
        // Read invoke (sees it)
        history.push(Op {
            index: 2,
            op_type: OpType::Invoke,
            f: OpFn::Read,
            value: OpValue::None,
            time: Some(2_000_000_000),
            process: 0,
        });
        history.push(Op {
            index: 3,
            op_type: OpType::Ok,
            f: OpFn::Read,
            value: OpValue::Set(HashSet::from_iter([1])),
            time: Some(3_000_000_000),
            process: 0,
        });
        // Read invoke (doesn't see it - lost!)
        history.push(Op {
            index: 4,
            op_type: OpType::Invoke,
            f: OpFn::Read,
            value: OpValue::None,
            time: Some(4_000_000_000),
            process: 0,
        });
        history.push(Op {
            index: 5,
            op_type: OpType::Ok,
            f: OpFn::Read,
            value: OpValue::Set(HashSet::new()),
            time: Some(5_000_000_000),
            process: 0,
        });

        let checker = SetFullChecker::new(SetFullOptions::default());
        let result = checker.check(&history);

        assert_eq!(result.valid, Validity::Invalid);
        assert_eq!(result.lost_count, 1);
        assert!(result.lost.contains(&1));
    }
}
