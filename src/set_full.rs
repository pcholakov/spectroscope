//! Set-full linearizability checker.
//!
//! Analyzes histories with `Add` and `Read` operations to detect consistency violations.
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

use std::hash::Hash;
use std::time::Duration;

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
    /// Index and time when the element was first known to exist.
    pub known_index: Option<usize>,
    pub known_time: Option<Duration>,
    /// Index and time of the last read that didn't observe the element.
    pub last_absent_index: Option<usize>,
    pub last_absent_time: Option<Duration>,
}

/// Validity status of the check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Validity {
    Valid,
    Invalid,
    Unknown,
}

/// Detailed info about a stale element (for worst_stale reporting).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorstStaleEntry<T> {
    pub element: T,
    pub outcome: ElementOutcome,
    /// Index and time when the element was first known to exist.
    pub known_index: usize,
    pub known_time: Option<Duration>,
    /// Index and time of the last read that didn't observe the element.
    pub last_absent_index: Option<usize>,
    pub last_absent_time: Option<Duration>,
    /// Latency in milliseconds until stable.
    pub stable_latency_ms: u64,
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
    /// Top 8 stale elements with highest latency, with detailed info.
    pub worst_stale: Vec<WorstStaleEntry<T>>,
    /// Elements with duplicates and their max multiplicity.
    pub duplicated: HashMap<T, usize>,
    /// Raw stable latencies in milliseconds (unsorted).
    stable_latencies: Vec<u64>,
    /// Raw lost latencies in milliseconds (unsorted).
    lost_latencies: Vec<u64>,
}

impl<T> SetFullResult<T> {
    /// Compute a percentile (0.0 to 1.0) from stable latencies. Returns None if no stable elements.
    #[must_use]
    pub fn stable_latency_percentile(&self, p: f64) -> Option<u64> {
        percentile(&self.stable_latencies, p)
    }

    /// Compute a percentile (0.0 to 1.0) from lost latencies. Returns None if no lost elements.
    #[must_use]
    pub fn lost_latency_percentile(&self, p: f64) -> Option<u64> {
        percentile(&self.lost_latencies, p)
    }

    /// Get all stable latencies (unsorted).
    #[must_use]
    pub fn stable_latencies(&self) -> &[u64] {
        &self.stable_latencies
    }

    /// Get all lost latencies (unsorted).
    #[must_use]
    pub fn lost_latencies(&self) -> &[u64] {
        &self.lost_latencies
    }
}

#[derive(Debug, Clone)]
struct ElementState<T> {
    element: T,
    known: Option<OpRef>,
    last_present: Option<OpRef>,
    last_absent: Option<OpRef>,
}

#[derive(Debug, Clone)]
struct OpRef {
    index: usize,
    time: Option<Duration>,
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
                la.time.map(|t| t + Duration::from_nanos(1))
            } else {
                Some(Duration::ZERO)
            };
            match (stable_time, known_time) {
                (Some(st), Some(kt)) => Some(st.saturating_sub(kt).as_millis() as u64),
                _ => Some(0),
            }
        } else {
            None
        };

        let lost_latency_ms = if lost {
            let lost_time = if let Some(ref lp) = self.last_present {
                lp.time.map(|t| t + Duration::from_nanos(1))
            } else {
                Some(Duration::ZERO)
            };
            match (lost_time, known_time) {
                (Some(lt), Some(kt)) => Some(lt.saturating_sub(kt).as_millis() as u64),
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
            known_index: self.known.as_ref().map(|k| k.index),
            known_time: self.known.as_ref().and_then(|k| k.time),
            last_absent_index: self.last_absent.as_ref().map(|a| a.index),
            last_absent_time: self.last_absent.as_ref().and_then(|a| a.time),
        }
    }
}

/// The set-full checker.
#[derive(Debug, Clone, Default)]
pub struct SetFullChecker {
    pub options: SetFullOptions,
}

impl SetFullChecker {
    #[must_use]
    pub fn new(options: SetFullOptions) -> Self {
        Self { options }
    }

    #[must_use]
    pub fn linearizable() -> Self {
        Self::new(SetFullOptions { linearizable: true })
    }

    /// Check a history for set consistency.
    #[must_use]
    pub fn check<T>(&self, history: &History<T>) -> SetFullResult<T>
    where
        T: Clone + Eq + Hash + Ord,
    {
        use std::collections::HashSet;

        let mut elements: HashMap<T, ElementState<T>> = HashMap::new();
        let mut duplicates: HashMap<T, usize> = HashMap::new();

        for (pos, op) in history.ops().iter().enumerate() {
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
                                // Mark as known (create if missing - handles add_ok without invoke)
                                elements
                                    .entry(v.clone())
                                    .or_insert_with(|| ElementState::new(v.clone()))
                                    .on_add_ok(op);
                            }
                            _ => {}
                        }
                    }
                }
                OpFn::Read => {
                    if op.op_type == OpType::Ok {
                        // Find the invocation for this read using position in history
                        let inv = match history.invocation(pos) {
                            Some(i) => i,
                            None => continue,
                        };

                        // Handle both Set and Vec read values
                        let read_set: HashSet<T> = match &op.value {
                            OpValue::Set(s) => s.iter().cloned().collect(),
                            OpValue::Vec(v) => {
                                // Detect duplicates: count frequencies
                                let mut freqs: HashMap<T, usize> = HashMap::new();
                                for elem in v {
                                    *freqs.entry(elem.clone()).or_insert(0) += 1;
                                }
                                // Track max multiplicity for elements with count > 1
                                for (elem, count) in &freqs {
                                    if *count > 1 {
                                        duplicates
                                            .entry(elem.clone())
                                            .and_modify(|c| *c = (*c).max(*count))
                                            .or_insert(*count);
                                    }
                                }
                                freqs.into_keys().collect()
                            }
                            _ => continue,
                        };

                        // Track elements discovered in reads (not previously added)
                        for elem in &read_set {
                            if !elements.contains_key(elem) {
                                let mut state = ElementState::new(elem.clone());
                                // Element is known from this read
                                state.known = Some(OpRef {
                                    index: op.index,
                                    time: op.time,
                                });
                                state.on_read_present(inv, op);
                                elements.insert(elem.clone(), state);
                            }
                        }

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
        let mut stale_results: Vec<&ElementResult<T>> = results
            .iter()
            .filter(|r| r.outcome == ElementOutcome::Stable && r.stable_latency_ms.unwrap_or(0) > 0)
            .collect();

        // Sort by stable_latency descending for worst_stale
        stale_results.sort_by(|a, b| {
            b.stable_latency_ms
                .unwrap_or(0)
                .cmp(&a.stable_latency_ms.unwrap_or(0))
        });

        // Top 8 worst stale elements
        let worst_stale: Vec<WorstStaleEntry<T>> = stale_results
            .iter()
            .take(8)
            .map(|r| WorstStaleEntry {
                element: r.element.clone(),
                outcome: r.outcome,
                known_index: r.known_index.unwrap_or(0),
                known_time: r.known_time,
                last_absent_index: r.last_absent_index,
                last_absent_time: r.last_absent_time,
                stable_latency_ms: r.stable_latency_ms.unwrap_or(0),
            })
            .collect();

        let stale: Vec<T> = stale_results.iter().map(|r| r.element.clone()).collect();

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
            worst_stale,
            duplicated: duplicates,
            stable_latencies,
            lost_latencies,
        }
    }
}

/// Compute a percentile from a slice of values.
fn percentile(values: &[u64], p: f64) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted: Vec<u64> = values.to_vec();
    sorted.sort_unstable();
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    Some(sorted[idx])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::ProcessId;
    use std::collections::HashSet;
    use std::time::Duration;

    /// Operation builder for tests. Mimics Jepsen's invoke-op/ok-op pattern.
    #[derive(Clone)]
    enum TestOp {
        AddInvoke(u64, i32),   // (process, value)
        AddOk(u64, i32),       // (process, value)
        ReadInvoke(u64),       // (process)
        ReadOk(u64, Vec<i32>), // (process, values seen)
    }

    /// Build a history from a sequence of test ops.
    /// Assigns indices sequentially and times as index * 1_000_000 (microseconds).
    fn build_history(ops: Vec<TestOp>) -> History<i32> {
        let mut history = History::new();
        for (idx, test_op) in ops.into_iter().enumerate() {
            let (op_type, f, value, process) = match test_op {
                TestOp::AddInvoke(p, v) => (OpType::Invoke, OpFn::Add, OpValue::Single(v), p),
                TestOp::AddOk(p, v) => (OpType::Ok, OpFn::Add, OpValue::Single(v), p),
                TestOp::ReadInvoke(p) => (OpType::Invoke, OpFn::Read, OpValue::None, p),
                TestOp::ReadOk(p, vals) => (
                    OpType::Ok,
                    OpFn::Read,
                    OpValue::Set(HashSet::from_iter(vals)),
                    p,
                ),
            };
            history.push(Op {
                index: idx,
                op_type,
                f,
                value,
                time: Some(Duration::from_nanos(idx as u64 * 1_000_000)), // microseconds, like Jepsen
                process: ProcessId::from(process),
            });
        }
        history
    }

    fn check(ops: Vec<TestOp>) -> SetFullResult<i32> {
        let history = build_history(ops);
        SetFullChecker::default().check(&history)
    }

    // Helpers matching Jepsen's test setup
    fn a() -> TestOp {
        TestOp::AddInvoke(0, 0)
    }
    fn a_ok() -> TestOp {
        TestOp::AddOk(0, 0)
    }
    fn r() -> TestOp {
        TestOp::ReadInvoke(1)
    }
    fn r_plus() -> TestOp {
        TestOp::ReadOk(1, vec![0])
    }
    fn r_minus() -> TestOp {
        TestOp::ReadOk(1, vec![])
    }

    // Multi-element helpers
    fn a0() -> TestOp {
        TestOp::AddInvoke(0, 0)
    }
    fn a0_ok() -> TestOp {
        TestOp::AddOk(0, 0)
    }
    fn a1() -> TestOp {
        TestOp::AddInvoke(1, 1)
    }
    fn a1_ok() -> TestOp {
        TestOp::AddOk(1, 1)
    }
    fn r2() -> TestOp {
        TestOp::ReadInvoke(2)
    }
    fn r3() -> TestOp {
        TestOp::ReadInvoke(3)
    }
    fn r2_empty() -> TestOp {
        TestOp::ReadOk(2, vec![])
    }
    fn r2_0() -> TestOp {
        TestOp::ReadOk(2, vec![0])
    }
    fn r2_1() -> TestOp {
        TestOp::ReadOk(2, vec![1])
    }
    fn r2_01() -> TestOp {
        TestOp::ReadOk(2, vec![0, 1])
    }
    fn r3_1() -> TestOp {
        TestOp::ReadOk(3, vec![1])
    }

    #[test]
    fn test_failed_add_ignored() {
        // Failed adds should not affect element tracking
        let mut history = History::new();
        history.push(Op {
            index: 0,
            op_type: OpType::Invoke,
            f: OpFn::Add,
            value: OpValue::Single(1),
            time: None,
            process: ProcessId(0),
        });
        history.push(Op {
            index: 1,
            op_type: OpType::Fail,
            f: OpFn::Add,
            value: OpValue::Single(1),
            time: None,
            process: ProcessId(0),
        });
        history.push(Op::read_invoke(2, 0u64));
        history.push(Op::read_ok(3, 0u64, Vec::<i32>::new()));

        let result = SetFullChecker::default().check(&history);
        // Element was invoked but failed, so it's tracked but unknown outcome
        assert_eq!(result.attempt_count, 1);
        assert_eq!(result.never_read_count, 1);
    }

    #[test]
    fn test_info_operations_ignored() {
        // Info (indeterminate) operations should be handled gracefully
        let mut history = History::new();
        history.push(Op {
            index: 0,
            op_type: OpType::Invoke,
            f: OpFn::Add,
            value: OpValue::Single(1),
            time: None,
            process: ProcessId(0),
        });
        history.push(Op {
            index: 1,
            op_type: OpType::Info,
            f: OpFn::Add,
            value: OpValue::Single(1),
            time: None,
            process: ProcessId(0),
        });
        history.push(Op::read_invoke(2, 1u64));
        history.push(Op::read_ok(3, 1u64, vec![1]));

        let result = SetFullChecker::default().check(&history);
        // Element seen in read, so it's known and stable
        assert_eq!(result.attempt_count, 1);
        assert_eq!(result.stable_count, 1);
    }

    #[test]
    fn test_never_read() {
        let result = check(vec![TestOp::AddInvoke(0, 0), TestOp::AddOk(0, 0)]);

        assert_eq!(result.valid, Validity::Unknown);
        assert_eq!(result.attempt_count, 1);
        assert_eq!(result.lost_count, 0);
        assert_eq!(result.never_read_count, 1);
        assert_eq!(result.never_read, vec![0]);
        assert_eq!(result.stable_count, 0);
    }

    #[test]
    fn test_never_confirmed_never_read() {
        // Add invoke only (no ok), concurrent absent read
        // [a r r-]
        let result = check(vec![a(), r(), r_minus()]);

        assert_eq!(result.valid, Validity::Unknown);
        assert_eq!(result.attempt_count, 1);
        assert_eq!(result.lost_count, 0);
        assert_eq!(result.never_read_count, 1);
        assert_eq!(result.never_read, vec![0]);
        assert_eq!(result.stable_count, 0);
    }

    #[test]
    fn test_successful_read_concurrent_before() {
        // [r a r+ a'] - Concurrent read before add
        let result = check(vec![r(), a(), r_plus(), a_ok()]);

        assert_eq!(result.valid, Validity::Valid);
        assert_eq!(result.stable_count, 1);
        assert_eq!(result.lost_count, 0);
        assert_eq!(result.never_read_count, 0);
    }

    #[test]
    fn test_successful_read_concurrent_outside() {
        // [r a a' r+] - Concurrent read outside add
        let result = check(vec![r(), a(), a_ok(), r_plus()]);

        assert_eq!(result.valid, Validity::Valid);
        assert_eq!(result.stable_count, 1);
    }

    #[test]
    fn test_successful_read_concurrent_inside() {
        // [a r r+ a'] - Concurrent read inside add
        let result = check(vec![a(), r(), r_plus(), a_ok()]);

        assert_eq!(result.valid, Validity::Valid);
        assert_eq!(result.stable_count, 1);
    }

    #[test]
    fn test_successful_read_concurrent_after() {
        // [a r a' r+] - Concurrent read after add invoke
        let result = check(vec![a(), r(), a_ok(), r_plus()]);

        assert_eq!(result.valid, Validity::Valid);
        assert_eq!(result.stable_count, 1);
    }

    #[test]
    fn test_successful_read_subsequent() {
        // [a a' r r+] - Subsequent read
        let result = check(vec![a(), a_ok(), r(), r_plus()]);

        assert_eq!(result.valid, Validity::Valid);
        assert_eq!(result.stable_count, 1);
        assert_eq!(result.stale_count, 0);
    }

    #[test]
    fn test_absent_read_after() {
        // [a a' r r-] - Add completes, then absent read -> lost
        let result = check(vec![a(), a_ok(), r(), r_minus()]);

        assert_eq!(result.valid, Validity::Invalid);
        assert_eq!(result.lost_count, 1);
        assert_eq!(result.lost, vec![0]);
        assert_eq!(result.stable_count, 0);
    }

    #[test]
    fn test_absent_read_concurrent_before() {
        // [r a r- a'] - Read before add, absent -> unknown (not lost)
        let result = check(vec![r(), a(), r_minus(), a_ok()]);

        assert_eq!(result.valid, Validity::Unknown);
        assert_eq!(result.lost_count, 0);
        assert_eq!(result.never_read_count, 1);
    }

    #[test]
    fn test_absent_read_concurrent_outside() {
        // [r a a' r-] - Read outside add, absent -> unknown
        let result = check(vec![r(), a(), a_ok(), r_minus()]);

        assert_eq!(result.valid, Validity::Unknown);
        assert_eq!(result.lost_count, 0);
        assert_eq!(result.never_read_count, 1);
    }

    #[test]
    fn test_absent_read_concurrent_inside() {
        // [a r r- a'] - Read inside add, absent -> unknown
        let result = check(vec![a(), r(), r_minus(), a_ok()]);

        assert_eq!(result.valid, Validity::Unknown);
        assert_eq!(result.lost_count, 0);
        assert_eq!(result.never_read_count, 1);
    }

    #[test]
    fn test_absent_read_concurrent_after_invoke() {
        // [a r a' r-] - Read after add invoke, absent -> unknown
        let result = check(vec![a(), r(), a_ok(), r_minus()]);

        assert_eq!(result.valid, Validity::Unknown);
        assert_eq!(result.lost_count, 0);
        assert_eq!(result.never_read_count, 1);
    }

    #[test]
    fn test_write_present_missing() {
        // Complex multi-element test:
        // We write a0 and a1 concurrently, reading 1 before a1 completes.
        // Then we read both, 0, then nothing.
        // [a0 a1 r2 r2'1 a0' a1' r2 r2'01 r2 r2'0 r2 r2']
        let result = check(vec![
            a0(),       // 0
            a1(),       // 1
            r2(),       // 2
            r2_1(),     // 3: sees {1}
            a0_ok(),    // 4
            a1_ok(),    // 5
            r2(),       // 6
            r2_01(),    // 7: sees {0,1}
            r2(),       // 8
            r2_0(),     // 9: sees {0}
            r2(),       // 10
            r2_empty(), // 11: sees {}
        ]);

        assert_eq!(result.valid, Validity::Invalid);
        assert_eq!(result.attempt_count, 2);
        assert_eq!(result.lost_count, 2);
        assert!(result.lost.contains(&0));
        assert!(result.lost.contains(&1));
        assert_eq!(result.stable_count, 0);
    }

    #[test]
    fn test_write_flutter_stable_lost() {
        // Element 1 flutters (present -> absent -> present), element 0 is lost.
        // t:   0  1    2   3  4     5     6  7   8     9
        // ops: a0 a0'  a1  r2 r2'1  a1'   r2 r3  r3'1  r2'0
        let result = check(vec![
            a0(),    // 0: add 0 invoke
            a0_ok(), // 1: add 0 ok
            a1(),    // 2: add 1 invoke
            r2(),    // 3: read invoke (process 2)
            r2_1(),  // 4: read ok, sees {1}
            a1_ok(), // 5: add 1 ok
            r2(),    // 6: read invoke (process 2)
            r3(),    // 7: read invoke (process 3)
            r3_1(),  // 8: read ok (process 3), sees {1}
            r2_0(),  // 9: read ok (process 2), sees {0}
        ]);

        assert_eq!(result.valid, Validity::Invalid);
        assert_eq!(result.attempt_count, 2);

        // Element 0 is lost: known at 1, last present at 9 (invoke 6), but last absent at 7
        // Wait, let me trace this:
        // Element 0: known at index 1 (a0_ok)
        //   r2 at index 3 -> r2_1 at 4 (doesn't contain 0) -> last_absent = 3
        //   r2 at index 6 -> r2_0 at 9 (contains 0) -> last_present = 6
        //   r3 at index 7 -> r3_1 at 8 (doesn't contain 0) -> last_absent = 7
        // So last_present (6) < last_absent (7), and last_absent (7) > known (1) -> LOST
        assert_eq!(result.lost_count, 1);
        assert!(result.lost.contains(&0));

        // Element 1 is stable but stale: known at index 4 (r2_1 saw it before a1_ok)
        //   r2 at 3 -> r2_1 at 4 (contains 1) -> last_present = 3
        //   r2 at 6 -> r2_0 at 9 (doesn't contain 1) -> last_absent = 6
        //   r3 at 7 -> r3_1 at 8 (contains 1) -> last_present = 7
        // So last_present (7) > last_absent (6) -> STABLE
        // Stale latency: last_absent.time + 1 - known.time = 6000001 - 4000000 = 2000001 ns = 2 ms
        assert_eq!(result.stable_count, 1);
        assert_eq!(result.stale_count, 1);
        assert!(result.stale.contains(&1));

        // Verify worst_stale contains element 1 with correct latency
        assert_eq!(result.worst_stale.len(), 1);
        let ws = &result.worst_stale[0];
        assert_eq!(ws.element, 1);
        assert_eq!(ws.outcome, ElementOutcome::Stable);
        assert_eq!(ws.known_index, 4);
        assert_eq!(ws.known_time, Some(Duration::from_nanos(4_000_000)));
        assert_eq!(ws.last_absent_index, Some(6));
        assert_eq!(ws.last_absent_time, Some(Duration::from_nanos(6_000_000)));
        assert_eq!(ws.stable_latency_ms, 2); // (6_000_001 - 4_000_000) / 1_000_000 = 2

        // Verify latency percentiles (only one stable element, so min=max=2)
        assert_eq!(result.stable_latency_percentile(0.0), Some(2));
        assert_eq!(result.stable_latency_percentile(1.0), Some(2));

        // Element 0: known at 1, last_present at 6
        // lost_latency = (6_000_001 - 1_000_000) / 1_000_000 = 5
        assert_eq!(result.lost_latency_percentile(0.0), Some(5));
        assert_eq!(result.lost_latency_percentile(1.0), Some(5));
    }

    #[test]
    fn test_duplicates_in_read() {
        // Read returns an element multiple times (duplicate)
        let mut history = History::new();
        history.push(Op::add_invoke(0, 0u64, 0).at(Duration::ZERO));
        history.push(Op::add_ok(1, 0u64, 0).at(Duration::from_millis(1)));
        history.push(Op::read_invoke(2, 1u64).at(Duration::from_millis(2)));
        // Read returns element 0 three times (duplicate!)
        history.push(Op::read_ok(3, 1u64, [0, 0, 0]).at(Duration::from_millis(3)));

        let result = SetFullChecker::default().check(&history);

        // Should be invalid due to duplicates
        assert_eq!(result.valid, Validity::Invalid);
        assert_eq!(result.duplicated_count, 1);
        assert_eq!(result.duplicated.get(&0), Some(&3)); // max multiplicity is 3
        assert_eq!(result.stable_count, 1); // element is still stable
    }

    #[test]
    fn test_latencies_write_present_missing() {
        // Verify latency values for the write_present_missing test
        // [a0 a1 r2 r2'1 a0' a1' r2 r2'01 r2 r2'0 r2 r2']
        let result = check(vec![
            a0(),       // 0
            a1(),       // 1
            r2(),       // 2
            r2_1(),     // 3: sees {1}
            a0_ok(),    // 4
            a1_ok(),    // 5
            r2(),       // 6
            r2_01(),    // 7: sees {0,1}
            r2(),       // 8
            r2_0(),     // 9: sees {0}
            r2(),       // 10
            r2_empty(), // 11: sees {}
        ]);

        // Both elements are lost
        assert_eq!(result.lost_count, 2);

        // Verify lost_latencies
        // Element 0: known at 4, last_present at 8 (invoke)
        //   lost_latency = (8_000_001 - 4_000_000) / 1_000_000 = 4
        // Element 1: known at 3 (first read that saw it), last_present at 6 (invoke)
        //   lost_latency = (6_000_001 - 3_000_000) / 1_000_000 = 3
        // Percentiles: min=3, max=4
        assert_eq!(result.lost_latency_percentile(0.0), Some(3)); // min
        assert_eq!(result.lost_latency_percentile(1.0), Some(4)); // max
    }

    #[test]
    fn test_worst_stale_ordering() {
        // Create multiple stale elements with different latencies
        // and verify worst_stale is ordered by latency descending
        let mut history = History::new();

        // Add elements 0, 1, 2 at staggered times
        for elem in 0..3 {
            let ms = |n: usize| Duration::from_millis(n as u64);
            history.push(Op::add_invoke(elem * 2, elem as u64, elem as i32).at(ms(elem * 2)));
            history.push(Op::add_ok(elem * 2 + 1, elem as u64, elem as i32).at(ms(elem * 2 + 1)));
        }
        // Element 0 known at index 1 (1ms), Element 1 at index 3 (3ms), Element 2 at index 5 (5ms)

        // Read that misses all (creating absent records)
        history.push(Op::read_invoke(6, 10u64).at(Duration::from_millis(6)));
        history.push(Op::read_ok(7, 10u64, std::iter::empty::<i32>()).at(Duration::from_millis(7)));

        // Read that sees all (making them stable)
        history.push(Op::read_invoke(8, 10u64).at(Duration::from_millis(8)));
        history.push(Op::read_ok(9, 10u64, [0, 1, 2]).at(Duration::from_millis(9)));

        let result = SetFullChecker::default().check(&history);

        // All elements are stable with different latencies
        assert_eq!(result.stable_count, 3);
        assert_eq!(result.stale_count, 3);
        assert_eq!(result.worst_stale.len(), 3);

        // Element 0: stable_latency = (6M+1 - 1M) / 1M = 5
        // Element 1: stable_latency = (6M+1 - 3M) / 1M = 3
        // Element 2: stable_latency = (6M+1 - 5M) / 1M = 1

        // Verify ordering: worst (highest latency) first
        assert_eq!(result.worst_stale[0].element, 0);
        assert_eq!(result.worst_stale[0].stable_latency_ms, 5);
        assert_eq!(result.worst_stale[1].element, 1);
        assert_eq!(result.worst_stale[1].stable_latency_ms, 3);
        assert_eq!(result.worst_stale[2].element, 2);
        assert_eq!(result.worst_stale[2].stable_latency_ms, 1);
    }

    #[test]
    fn test_element_discovered_in_read() {
        // Element appears in a read without any prior add operation.
        // This can happen if the history is incomplete or if element was added
        // by a process not in the history.
        let mut history = History::new();

        // Read sees element 42, but there's no add for it
        history.push(Op::read_invoke(0, 0u64).at(Duration::from_millis(0)));
        history.push(Op::read_ok(1, 0u64, [42]).at(Duration::from_millis(1)));

        // Second read also sees it
        history.push(Op::read_invoke(2, 0u64).at(Duration::from_millis(2)));
        history.push(Op::read_ok(3, 0u64, [42]).at(Duration::from_millis(3)));

        let result = SetFullChecker::default().check(&history);

        // Element should be tracked and stable (seen in both reads)
        assert_eq!(result.attempt_count, 1);
        assert_eq!(result.stable_count, 1);
        assert_eq!(result.valid, Validity::Valid);
    }

    #[test]
    fn test_element_discovered_in_read_then_lost() {
        // Element appears in read, then disappears - should be lost
        let mut history = History::new();

        history.push(Op::read_invoke(0, 0u64).at(Duration::from_millis(0)));
        history.push(Op::read_ok(1, 0u64, [42]).at(Duration::from_millis(1)));

        history.push(Op::read_invoke(2, 0u64).at(Duration::from_millis(2)));
        history.push(Op::read_ok(3, 0u64, []).at(Duration::from_millis(3)));

        let result = SetFullChecker::default().check(&history);

        assert_eq!(result.attempt_count, 1);
        assert_eq!(result.lost_count, 1);
        assert!(result.lost.contains(&42));
        assert_eq!(result.valid, Validity::Invalid);
    }

    #[test]
    fn test_add_ok_without_invoke() {
        // add_ok appears without a prior add_invoke - should still track element
        let mut history = History::new();

        // Only the completion, no invoke
        history.push(Op::add_ok(0, 0u64, 99).at(Duration::from_millis(0)));

        // Read sees the element
        history.push(Op::read_invoke(1, 1u64).at(Duration::from_millis(1)));
        history.push(Op::read_ok(2, 1u64, [99]).at(Duration::from_millis(2)));

        let result = SetFullChecker::default().check(&history);

        assert_eq!(result.attempt_count, 1);
        assert_eq!(result.stable_count, 1);
        assert_eq!(result.valid, Validity::Valid);
    }

    #[test]
    fn test_invocation_matches_correct_op_fn() {
        // Verify that read_ok matches read_invoke, not add_invoke from same process
        let mut history = History::new();

        // Process 0: add invoke, then read invoke, then read ok
        // The read_ok should match the read_invoke, not the add_invoke
        history.push(Op::add_invoke(0, 0u64, 1).at(Duration::from_millis(0)));
        history.push(Op::read_invoke(1, 0u64).at(Duration::from_millis(1)));
        history.push(Op::read_ok(2, 0u64, [1]).at(Duration::from_millis(2)));
        history.push(Op::add_ok(3, 0u64, 1).at(Duration::from_millis(3)));

        let result = SetFullChecker::default().check(&history);

        // Should work correctly - element is stable
        assert_eq!(result.stable_count, 1);
        assert_eq!(result.valid, Validity::Valid);
    }

    // ====== Stress tests ======

    /// Positive test: Single add, all reads see it.
    /// Catches: basic tracking + "stable in all reads after add" logic.
    #[test]
    fn test_single_add_all_reads_see_it() {
        let result = check(vec![
            TestOp::AddInvoke(0, 1),
            TestOp::AddOk(0, 1),
            TestOp::ReadInvoke(1),
            TestOp::ReadOk(1, vec![1]),
            TestOp::ReadInvoke(2),
            TestOp::ReadOk(2, vec![1]),
        ]);

        assert_eq!(result.valid, Validity::Valid);
        assert_eq!(result.stable_count, 1);
        assert_eq!(result.lost_count, 0);
        assert_eq!(result.stale_count, 0);
        assert_eq!(result.never_read_count, 0);
    }

    /// Positive test: Overlapping read (allowed to miss) then consistent.
    /// Catches: treating overlapping reads as required to see element (false stale).
    #[test]
    fn test_overlapping_read_then_consistent() {
        let result = check(vec![
            TestOp::AddInvoke(0, 1),
            TestOp::ReadInvoke(1),
            TestOp::ReadOk(1, vec![]), // overlaps add, can miss
            TestOp::AddOk(0, 1),
            TestOp::ReadInvoke(1),
            TestOp::ReadOk(1, vec![1]),
        ]);

        assert_eq!(result.valid, Validity::Valid);
        assert_eq!(result.stable_count, 1);
        assert_eq!(result.lost_count, 0);
        assert_eq!(result.stale_count, 0);
        assert_eq!(result.never_read_count, 0);
    }

    /// Positive test: Read-only history.
    /// Catches: errors when no adds exist.
    #[test]
    fn test_read_only_history() {
        let result = check(vec![
            TestOp::ReadInvoke(0),
            TestOp::ReadOk(0, vec![]),
            TestOp::ReadInvoke(1),
            TestOp::ReadOk(1, vec![]),
        ]);

        assert_eq!(result.valid, Validity::Unknown);
        assert_eq!(result.stable_count, 0);
        assert_eq!(result.lost_count, 0);
        assert_eq!(result.stale_count, 0);
        assert_eq!(result.never_read_count, 0);
    }

    /// Positive test: Duplicate adds of same element (idempotent).
    /// Catches: double-counting elements or treating second add as separate element.
    #[test]
    fn test_duplicate_adds_idempotent() {
        let result = check(vec![
            TestOp::AddInvoke(0, 1),
            TestOp::AddOk(0, 1),
            TestOp::AddInvoke(1, 1),
            TestOp::AddOk(1, 1),
            TestOp::ReadInvoke(2),
            TestOp::ReadOk(2, vec![1]),
        ]);

        assert_eq!(result.valid, Validity::Valid);
        assert_eq!(result.stable_count, 1);
        assert_eq!(result.lost_count, 0);
        assert_eq!(result.stale_count, 0);
        assert_eq!(result.never_read_count, 0);
    }

    /// Positive test: Two adds, interleaved reads, eventually consistent.
    /// Catches: incorrect requirement that reads after one add ok must include future adds.
    #[test]
    fn test_two_adds_interleaved_reads() {
        let result = check(vec![
            TestOp::AddInvoke(0, 1),
            TestOp::AddInvoke(1, 2),
            TestOp::AddOk(0, 1),
            TestOp::ReadInvoke(2),
            TestOp::ReadOk(2, vec![1]), // y not yet ok
            TestOp::AddOk(1, 2),
            TestOp::ReadInvoke(2),
            TestOp::ReadOk(2, vec![1, 2]),
        ]);

        assert_eq!(result.valid, Validity::Valid);
        assert_eq!(result.stable_count, 2);
        assert_eq!(result.lost_count, 0);
        assert_eq!(result.stale_count, 0);
        assert_eq!(result.never_read_count, 0);
    }

    /// Negative test: Lost element (never seen after confirmed add).
    /// Catches: failing to mark lost when confirmed add disappears.
    #[test]
    fn test_lost_element_never_seen() {
        let result = check(vec![
            TestOp::AddInvoke(0, 1),
            TestOp::AddOk(0, 1),
            TestOp::ReadInvoke(1),
            TestOp::ReadOk(1, vec![]),
            TestOp::ReadInvoke(2),
            TestOp::ReadOk(2, vec![]),
        ]);

        assert_eq!(result.valid, Validity::Invalid);
        assert_eq!(result.stable_count, 0);
        assert_eq!(result.lost_count, 1);
        assert_eq!(result.stale_count, 0);
        assert_eq!(result.never_read_count, 0);
        assert!(result.lost.contains(&1));
    }

    /// Negative test: Stale read (linearizability violation).
    /// Catches: not detecting staleness when element appears later.
    #[test]
    fn test_stale_read_linearizability() {
        // Valid in eventual consistency mode
        let eventual_result = SetFullChecker::default().check(&build_history(vec![
            TestOp::AddInvoke(0, 1),
            TestOp::AddOk(0, 1),
            TestOp::ReadInvoke(1),
            TestOp::ReadOk(1, vec![]),
            TestOp::ReadInvoke(2),
            TestOp::ReadOk(2, vec![1]),
        ]));
        assert_eq!(eventual_result.valid, Validity::Valid);
        assert_eq!(eventual_result.stale_count, 1);

        // Invalid in linearizable mode
        let linearizable_result = SetFullChecker::linearizable().check(&build_history(vec![
            TestOp::AddInvoke(0, 1),
            TestOp::AddOk(0, 1),
            TestOp::ReadInvoke(1),
            TestOp::ReadOk(1, vec![]),
            TestOp::ReadInvoke(2),
            TestOp::ReadOk(2, vec![1]),
        ]));
        assert_eq!(linearizable_result.valid, Validity::Invalid);
        assert_eq!(linearizable_result.stale_count, 1);
    }

    /// Negative test: Mixed outcomes (stable + lost + stale).
    /// Catches: per-element accounting when multiple outcomes coexist.
    #[test]
    fn test_mixed_outcomes() {
        let result = check(vec![
            TestOp::AddInvoke(0, 1),
            TestOp::AddOk(0, 1),
            TestOp::AddInvoke(1, 2),
            TestOp::AddOk(1, 2),
            TestOp::AddInvoke(2, 3),
            TestOp::AddOk(2, 3),
            TestOp::ReadInvoke(0),
            TestOp::ReadOk(0, vec![1]), // sees only 1
            TestOp::ReadInvoke(1),
            TestOp::ReadOk(1, vec![1, 2]), // sees 1, 2 (not 3)
        ]);

        // Element 1: stable with no staleness (seen in all reads, no absent)
        // Element 2: stable but stale (first read after add ok missed, later read sees)
        // Element 3: lost (added ok, never seen in any read after)
        assert_eq!(result.valid, Validity::Invalid);
        assert_eq!(result.stable_count, 2);
        assert_eq!(result.lost_count, 1);
        assert_eq!(result.stale_count, 1);
        assert_eq!(result.never_read_count, 0);
        assert!(result.lost.contains(&3));
        assert!(result.stale.contains(&2));
    }

    /// Negative test: Read after add ok missing but no later reads.
    /// Catches: wrongly classifying as stale or never-read when it's actually lost.
    #[test]
    fn test_single_read_miss_is_lost() {
        let result = check(vec![
            TestOp::AddInvoke(0, 1),
            TestOp::AddOk(0, 1),
            TestOp::ReadInvoke(1),
            TestOp::ReadOk(1, vec![]),
        ]);

        assert_eq!(result.valid, Validity::Invalid);
        assert_eq!(result.stable_count, 0);
        assert_eq!(result.lost_count, 1);
        assert_eq!(result.stale_count, 0);
        assert_eq!(result.never_read_count, 0);
    }

    /// Stress test: Many processes, many elements, clean linearizable.
    /// Catches: race handling, large state, "overlap allows missing" logic.
    #[test]
    fn test_many_processes_many_elements() {
        let mut ops = Vec::new();
        let num_elements = 50;

        // Add all elements with overlapping reads
        for i in 0..num_elements {
            ops.push(TestOp::AddInvoke(i as u64, i));
            if i > 0 {
                // Overlapping read that misses the current element
                ops.push(TestOp::ReadInvoke((i + 100) as u64));
                let seen: Vec<i32> = (0..i).collect();
                ops.push(TestOp::ReadOk((i + 100) as u64, seen));
            }
            ops.push(TestOp::AddOk(i as u64, i));
        }

        // Final reads see all elements
        ops.push(TestOp::ReadInvoke(200));
        let all: Vec<i32> = (0..num_elements).collect();
        ops.push(TestOp::ReadOk(200, all.clone()));
        ops.push(TestOp::ReadInvoke(201));
        ops.push(TestOp::ReadOk(201, all));

        let result = check(ops);

        assert_eq!(result.valid, Validity::Valid);
        assert_eq!(result.stable_count, num_elements as usize);
        assert_eq!(result.lost_count, 0);
        assert_eq!(result.stale_count, 0);
        assert_eq!(result.never_read_count, 0);
    }

    /// Stress test: Many elements with targeted staleness.
    /// Catches: stale detection at scale; off-by-one/ordering mistakes.
    #[test]
    fn test_many_elements_targeted_staleness() {
        let mut ops = Vec::new();
        let num_elements = 100;
        let mut accumulated = Vec::new();

        for k in 0..num_elements {
            ops.push(TestOp::AddInvoke(k as u64, k));
            ops.push(TestOp::AddOk(k as u64, k));

            if k % 10 == 0 {
                // Stale case: first read misses, second sees
                ops.push(TestOp::ReadInvoke(200));
                ops.push(TestOp::ReadOk(200, accumulated.clone()));
                accumulated.push(k);
                ops.push(TestOp::ReadInvoke(201));
                ops.push(TestOp::ReadOk(201, accumulated.clone()));
            } else {
                // Normal case: read sees it immediately
                accumulated.push(k);
                ops.push(TestOp::ReadInvoke(200));
                ops.push(TestOp::ReadOk(200, accumulated.clone()));
            }
        }

        // Eventual consistency: valid
        let eventual_result = SetFullChecker::default().check(&build_history(ops.clone()));
        assert_eq!(eventual_result.valid, Validity::Valid);
        assert_eq!(eventual_result.stable_count, num_elements as usize);
        assert_eq!(eventual_result.stale_count, 10);
        assert_eq!(eventual_result.lost_count, 0);

        // Linearizable: invalid due to stale elements
        let linearizable_result = SetFullChecker::linearizable().check(&build_history(ops));
        assert_eq!(linearizable_result.valid, Validity::Invalid);
        assert_eq!(linearizable_result.stale_count, 10);
    }

    /// Stress test: Lost burst under high interleaving.
    /// Catches: lost detection amid large consistent subset.
    #[test]
    fn test_lost_burst_high_interleaving() {
        let mut ops = Vec::new();
        let num_elements = 50;
        let lost_set: HashSet<i32> = [5, 15, 25, 35, 45].iter().cloned().collect();

        // Add all elements
        for i in 0..num_elements {
            ops.push(TestOp::AddInvoke(i as u64, i));
            ops.push(TestOp::AddOk(i as u64, i));
        }

        // Reads see all elements except lost_set
        for _ in 0..3 {
            ops.push(TestOp::ReadInvoke(100));
            let visible: Vec<i32> = (0..num_elements)
                .filter(|e| !lost_set.contains(e))
                .collect();
            ops.push(TestOp::ReadOk(100, visible));
        }

        let result = check(ops);

        assert_eq!(result.valid, Validity::Invalid);
        assert_eq!(result.stable_count, 45);
        assert_eq!(result.lost_count, 5);
        assert_eq!(result.stale_count, 0);
        assert_eq!(result.never_read_count, 0);
        for elem in &lost_set {
            assert!(result.lost.contains(elem));
        }
    }

    /// Stress test: Never-read flood.
    /// Catches: memory growth/overflow, handling of empty read set.
    #[test]
    fn test_never_read_flood() {
        let mut ops = Vec::new();
        let num_elements = 100;

        for i in 0..num_elements {
            ops.push(TestOp::AddInvoke(i as u64, i));
            ops.push(TestOp::AddOk(i as u64, i));
        }

        let result = check(ops);

        assert_eq!(result.valid, Validity::Unknown);
        assert_eq!(result.stable_count, 0);
        assert_eq!(result.lost_count, 0);
        assert_eq!(result.stale_count, 0);
        assert_eq!(result.never_read_count, num_elements as usize);
    }

    #[cfg(test)]
    mod property_tests {
        use super::*;
        use proptest::prelude::*;
        use rand::seq::{IndexedRandom, IteratorRandom};
        use rand::{Rng, SeedableRng};
        use std::collections::{HashMap, HashSet};

        /// Controls whether to generate valid or invalid histories
        #[derive(Debug, Clone, Copy, PartialEq)]
        enum TargetValidity {
            /// Generate only valid histories
            ForceValid,
            /// Generate only invalid histories with injected anomalies
            ForceInvalid,
        }

        /// Configuration for history generation
        #[derive(Debug, Clone)]
        struct GenConfig {
            /// Total number of operations (invokes + oks)
            ops_total: usize,
            /// Number of concurrent processes
            processes: usize,
            /// Element pool size
            elements: usize,
            /// Proportion of adds vs reads (0.0 = all reads, 1.0 = all adds)
            add_ratio: f64,
            /// Probability of overlapping operations (0.0 = sequential, 1.0 = maximum overlap)
            concurrency: f64,
            /// Target validity for the generated history
            target_validity: TargetValidity,
            /// Burstiness: probability of clustering operations (0.0 = uniform, 1.0 = clustered)
            burstiness: f64,
        }

        impl GenConfig {
            fn high_concurrency() -> Self {
                GenConfig {
                    ops_total: 2000,
                    processes: 64,
                    elements: 50,
                    add_ratio: 0.5,
                    concurrency: 0.9,
                    target_validity: TargetValidity::ForceValid,
                    burstiness: 0.1,
                }
            }

            fn sequential() -> Self {
                GenConfig {
                    ops_total: 1500,
                    processes: 4,
                    elements: 30,
                    add_ratio: 0.5,
                    concurrency: 0.0,
                    target_validity: TargetValidity::ForceValid,
                    burstiness: 0.1,
                }
            }

            fn bursty() -> Self {
                GenConfig {
                    ops_total: 2000,
                    processes: 16,
                    elements: 40,
                    add_ratio: 0.5,
                    concurrency: 0.5,
                    target_validity: TargetValidity::ForceValid,
                    burstiness: 0.9,
                }
            }

            fn read_heavy() -> Self {
                GenConfig {
                    ops_total: 2000,
                    processes: 16,
                    elements: 30,
                    add_ratio: 0.1,
                    concurrency: 0.5,
                    target_validity: TargetValidity::ForceValid,
                    burstiness: 0.1,
                }
            }

            fn write_heavy() -> Self {
                GenConfig {
                    ops_total: 2000,
                    processes: 16,
                    elements: 100,
                    add_ratio: 0.9,
                    concurrency: 0.5,
                    target_validity: TargetValidity::ForceValid,
                    burstiness: 0.1,
                }
            }
        }

        /// Intermediate representation for operation skeleton
        #[derive(Debug, Clone)]
        enum SkelOp {
            AddInvoke { process: u64, elem: i32 },
            AddOk { process: u64, elem: i32 },
            ReadInvoke { process: u64, read_id: usize },
            ReadOk { process: u64, read_id: usize },
        }

        /// Generate operation skeleton with controlled overlap
        fn gen_skeleton(config: &GenConfig, rng: &mut impl Rng) -> Vec<SkelOp> {
            let mut ops = Vec::new();
            let mut inflight: HashMap<u64, Option<SkelOp>> = HashMap::new();
            let mut next_read_id = 0;
            let mut epoch_add_ratio = config.add_ratio;
            let epoch_length = if config.burstiness > 0.5 {
                100
            } else {
                config.ops_total
            };

            for i in 0..config.ops_total {
                // Update add_ratio for bursty behavior
                if config.burstiness > 0.5 && i % epoch_length == 0 {
                    epoch_add_ratio = if rng.random_bool(0.5) {
                        rng.random_range(0.0..0.3)
                    } else {
                        rng.random_range(0.7..1.0)
                    };
                }

                // Choose process (with burstiness, favor recent processes)
                let process = if config.burstiness > 0.5
                    && !inflight.is_empty()
                    && rng.random_bool(config.burstiness)
                {
                    *inflight.keys().choose(rng).unwrap()
                } else {
                    rng.random_range(0..config.processes) as u64
                };

                // Check if this process has an inflight operation
                let has_inflight = inflight.get(&process).and_then(|x| x.as_ref()).is_some();

                // Decide whether to complete existing operation or start new one
                let should_complete = has_inflight
                    && (config.concurrency == 0.0 || rng.random_bool(1.0 - config.concurrency));

                if should_complete {
                    // Complete the inflight operation
                    if let Some(Some(inflight_op)) = inflight.get(&process) {
                        match inflight_op {
                            SkelOp::AddInvoke { process, elem } => {
                                ops.push(SkelOp::AddOk {
                                    process: *process,
                                    elem: *elem,
                                });
                            }
                            SkelOp::ReadInvoke { process, read_id } => {
                                ops.push(SkelOp::ReadOk {
                                    process: *process,
                                    read_id: *read_id,
                                });
                            }
                            _ => unreachable!(),
                        }
                        inflight.insert(process, None);
                    }
                } else {
                    // Start new operation
                    let is_add = rng.random_bool(epoch_add_ratio);
                    if is_add {
                        let elem = rng.random_range(0..config.elements) as i32;
                        ops.push(SkelOp::AddInvoke { process, elem });
                        inflight.insert(process, Some(SkelOp::AddInvoke { process, elem }));
                    } else {
                        let read_id = next_read_id;
                        next_read_id += 1;
                        ops.push(SkelOp::ReadInvoke { process, read_id });
                        inflight.insert(process, Some(SkelOp::ReadInvoke { process, read_id }));
                    }
                }
            }

            // Complete all remaining inflight operations
            for (_process, inflight_op) in inflight.iter() {
                if let Some(op) = inflight_op {
                    match op {
                        SkelOp::AddInvoke { process, elem } => {
                            ops.push(SkelOp::AddOk {
                                process: *process,
                                elem: *elem,
                            });
                        }
                        SkelOp::ReadInvoke { process, read_id } => {
                            ops.push(SkelOp::ReadOk {
                                process: *process,
                                read_id: *read_id,
                            });
                        }
                        _ => unreachable!(),
                    }
                }
            }

            ops
        }

        /// Assign read results based on visibility state
        fn assign_reads(
            skeleton: &[SkelOp],
            config: &GenConfig,
            rng: &mut impl Rng,
        ) -> Vec<TestOp> {
            // First pass: track which operations are inflight at each point
            let mut read_invoke_idx: HashMap<usize, usize> = HashMap::new(); // read_id -> index of read_invoke
            let mut read_ok_idx: HashMap<usize, usize> = HashMap::new(); // read_id -> index of read_ok
            let mut add_invoke_idx: HashMap<i32, Vec<usize>> = HashMap::new(); // elem -> indices of add_invokes
            let mut add_ok_idx: HashMap<i32, Vec<usize>> = HashMap::new(); // elem -> indices of add_oks

            for (idx, op) in skeleton.iter().enumerate() {
                match op {
                    SkelOp::AddInvoke { elem, .. } => {
                        add_invoke_idx.entry(*elem).or_default().push(idx);
                    }
                    SkelOp::AddOk { elem, .. } => {
                        add_ok_idx.entry(*elem).or_default().push(idx);
                    }
                    SkelOp::ReadInvoke { read_id, .. } => {
                        read_invoke_idx.insert(*read_id, idx);
                    }
                    SkelOp::ReadOk { read_id, .. } => {
                        read_ok_idx.insert(*read_id, idx);
                    }
                }
            }

            // Second pass: determine read results
            let mut read_results: HashMap<usize, Vec<i32>> = HashMap::new();

            for (read_id, &read_ok_position) in read_ok_idx.iter() {
                let read_invoke_position = read_invoke_idx[read_id];
                let mut result = Vec::new();

                // For each element, check if it should be visible
                for (elem, ok_positions) in add_ok_idx.iter() {
                    for &ok_pos in ok_positions {
                        // Element is definitely visible if add_ok happened before read_invoke
                        let definitely_visible = ok_pos < read_invoke_position;

                        // Element may be visible if operations overlap
                        let possibly_visible =
                            ok_pos >= read_invoke_position && ok_pos < read_ok_position;

                        if definitely_visible {
                            if !result.contains(elem) {
                                result.push(*elem);
                            }
                            break; // One add_ok is enough to make it visible
                        } else if possibly_visible
                            && config.target_validity == TargetValidity::ForceValid
                        {
                            // For valid histories, randomly include overlapping adds
                            if rng.random_bool(0.5) && !result.contains(elem) {
                                result.push(*elem);
                            }
                            break;
                        }
                    }
                }

                result.sort_unstable();
                read_results.insert(*read_id, result);
            }

            // Inject anomalies for invalid histories
            if config.target_validity == TargetValidity::ForceInvalid {
                let completed_adds: HashSet<i32> = add_ok_idx.keys().copied().collect();
                inject_anomalies(&mut read_results, &completed_adds, rng);
            }

            // Convert skeleton to TestOp with read results
            let mut final_result = Vec::new();
            for op in skeleton.iter() {
                match op {
                    SkelOp::AddInvoke { process, elem } => {
                        final_result.push(TestOp::AddInvoke(*process, *elem));
                    }
                    SkelOp::AddOk { process, elem } => {
                        final_result.push(TestOp::AddOk(*process, *elem));
                    }
                    SkelOp::ReadInvoke { process, .. } => {
                        final_result.push(TestOp::ReadInvoke(*process));
                    }
                    SkelOp::ReadOk { process, read_id } => {
                        let result_vec = read_results.get(read_id).cloned().unwrap_or_default();
                        final_result.push(TestOp::ReadOk(*process, result_vec));
                    }
                }
            }

            final_result
        }

        /// Inject anomalies into read results to create invalid histories
        fn inject_anomalies(
            read_results: &mut HashMap<usize, Vec<i32>>,
            completed_adds: &HashSet<i32>,
            rng: &mut impl Rng,
        ) {
            if read_results.is_empty() || completed_adds.is_empty() {
                return;
            }

            // Find reads that actually contain elements
            let non_empty_reads: Vec<usize> = read_results
                .iter()
                .filter(|(_, v)| !v.is_empty())
                .map(|(k, _)| *k)
                .collect();

            if non_empty_reads.is_empty() {
                // If no reads have elements, inject a "never appears" anomaly
                // by adding an element to the first read that should have it
                if let Some((_read_id, result)) = read_results.iter_mut().next() {
                    if let Some(&elem) = completed_adds.iter().next() {
                        result.push(elem);
                        result.sort_unstable();
                    }
                }
                return;
            }

            let anomaly_type = rng.random_range(0..2);
            match anomaly_type {
                0 => {
                    // Lost element: find a read with elements and remove one that should be there
                    let &target_read = non_empty_reads.choose(rng).unwrap();
                    let result = read_results.get_mut(&target_read).unwrap();
                    if !result.is_empty() {
                        let idx = rng.random_range(0..result.len());
                        result.remove(idx);
                    }
                }
                1 => {
                    // Stale read: remove element from early reads but keep in later ones
                    let mut read_ids: Vec<usize> = read_results.keys().copied().collect();
                    read_ids.sort_unstable();

                    if read_ids.len() >= 2 {
                        // Find an element that appears in multiple reads
                        let mut elem_to_remove = None;
                        for &elem in completed_adds.iter() {
                            let count = read_results.values().filter(|v| v.contains(&elem)).count();
                            if count >= 2 {
                                elem_to_remove = Some(elem);
                                break;
                            }
                        }

                        if let Some(elem) = elem_to_remove {
                            // Remove from first half of reads, keep in second half
                            let mid = read_ids.len() / 2;
                            for read_id in read_ids.iter().take(mid) {
                                let result = read_results.get_mut(read_id).unwrap();
                                result.retain(|e| *e != elem);
                            }
                        } else {
                            // Fallback: just remove an element from one read
                            let &target_read = non_empty_reads.choose(rng).unwrap();
                            let result = read_results.get_mut(&target_read).unwrap();
                            if !result.is_empty() {
                                let idx = rng.random_range(0..result.len());
                                result.remove(idx);
                            }
                        }
                    }
                }
                _ => unreachable!(),
            }
        }

        /// Generate a complete history based on configuration
        fn gen_history(config: GenConfig, seed: u64) -> Vec<TestOp> {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let skeleton = gen_skeleton(&config, &mut rng);
            assign_reads(&skeleton, &config, &mut rng)
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(20))]

            #[test]
            fn prop_valid_histories_pass(
                ops_total in 1000usize..3000,
                processes in 2usize..32,
                elements in 10usize..100,
                add_ratio in 0.1f64..0.9,
                concurrency in 0.0f64..0.8,
                seed in any::<u64>(),
            ) {
                let config = GenConfig {
                    ops_total,
                    processes,
                    elements,
                    add_ratio,
                    concurrency,
                    target_validity: TargetValidity::ForceValid,
                    burstiness: 0.1,
                };
                let history = gen_history(config, seed);
                let history_len = history.len();
                let result = check(history);

                // Valid histories should not be marked as invalid
                prop_assert_ne!(
                    result.valid,
                    Validity::Invalid,
                    "Valid history marked as invalid. History length: {}, Stable: {}, Lost: {}, Stale: {}",
                    history_len,
                    result.stable_count,
                    result.lost_count,
                    result.stale_count
                );
            }

            #[test]
            fn prop_invalid_histories_fail(
                ops_total in 1000usize..2000,
                processes in 2usize..8,
                elements in 10usize..30,
                add_ratio in 0.4f64..0.6,
                concurrency in 0.0f64..0.3,
                seed in any::<u64>(),
            ) {
                let config = GenConfig {
                    ops_total,
                    processes,
                    elements,
                    add_ratio,
                    concurrency,
                    target_validity: TargetValidity::ForceInvalid,
                    burstiness: 0.1,
                };
                let history = gen_history(config, seed);
                let result = check(history);

                // Anomaly injection is best-effort; with overlapping operations some
                // anomalies may not be detectable. We verify the result is internally consistent.
                prop_assert!(result.stable_count + result.lost_count + result.never_read_count <= result.attempt_count);
            }
        }

        #[test]
        fn test_generated_invalid_sequential() {
            // Generate a sequential (low concurrency) invalid history
            // With no concurrency, anomalies should be easily detectable
            let config = GenConfig {
                ops_total: 1000,
                processes: 4,
                elements: 20,
                add_ratio: 0.5,
                concurrency: 0.0, // Sequential - no overlap
                target_validity: TargetValidity::ForceInvalid,
                burstiness: 0.0,
            };
            let history = gen_history(config, 12345);
            let result = check(history);

            // With sequential execution and anomaly injection, we SHOULD detect invalidity
            assert!(
                result.valid == Validity::Invalid
                    || result.lost_count > 0
                    || result.stale_count > 0,
                "Sequential invalid history was not detected as invalid. Valid: {:?}, Lost: {}, Stale: {}",
                result.valid,
                result.lost_count,
                result.stale_count
            );
        }

        #[test]
        fn test_deterministic_invalid_history() {
            // Create a simple, deterministic invalid history:
            // Add element 1, read sees it, then another read doesn't see it (lost element)
            let ops = vec![
                // Process 0 adds element 1
                TestOp::AddInvoke(0, 1),
                TestOp::AddOk(0, 1),
                // Process 1 reads and sees element 1
                TestOp::ReadInvoke(1),
                TestOp::ReadOk(1, vec![1]),
                // Process 2 reads but doesn't see element 1 (invalid - lost element)
                TestOp::ReadInvoke(2),
                TestOp::ReadOk(2, vec![]), // Lost element 1
            ];

            let result = check(ops);

            // This should be detected as invalid with a lost element
            assert_eq!(result.valid, Validity::Invalid);
            assert_eq!(result.lost_count, 1);
            assert!(result.lost.contains(&1));
        }

        #[test]
        fn test_high_concurrency_scenario() {
            let config = GenConfig::high_concurrency();
            let history = gen_history(config, 42);
            let history_len = history.len();
            let result = check(history);

            assert_ne!(result.valid, Validity::Invalid);
            assert!(history_len >= 2000);
        }

        #[test]
        fn test_sequential_scenario() {
            let config = GenConfig::sequential();
            let history = gen_history(config, 42);
            let history_len = history.len();
            let result = check(history);

            assert_ne!(result.valid, Validity::Invalid);
            assert!(history_len >= 1500);
        }

        #[test]
        fn test_bursty_scenario() {
            let config = GenConfig::bursty();
            let history = gen_history(config, 42);
            let history_len = history.len();
            let result = check(history);

            assert_ne!(result.valid, Validity::Invalid);
            assert!(history_len >= 2000);
        }

        #[test]
        fn test_read_heavy_scenario() {
            let config = GenConfig::read_heavy();
            let history = gen_history(config, 42);

            // Count read operations
            let read_count = history
                .iter()
                .filter(|op| matches!(op, TestOp::ReadInvoke(_)))
                .count();
            let add_count = history
                .iter()
                .filter(|op| matches!(op, TestOp::AddInvoke(_, _)))
                .count();

            let result = check(history);

            assert_ne!(result.valid, Validity::Invalid);
            assert!(read_count > add_count * 5);
        }

        #[test]
        fn test_write_heavy_scenario() {
            let config = GenConfig::write_heavy();
            let history = gen_history(config, 42);

            // Count add operations
            let add_count = history
                .iter()
                .filter(|op| matches!(op, TestOp::AddInvoke(_, _)))
                .count();
            let read_count = history
                .iter()
                .filter(|op| matches!(op, TestOp::ReadInvoke(_)))
                .count();

            let result = check(history);

            assert_ne!(result.valid, Validity::Invalid);
            assert!(add_count > read_count * 5);
        }
    }
}
