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
    pub fn stable_latency_percentile(&self, p: f64) -> Option<u64> {
        percentile(&self.stable_latencies, p)
    }

    /// Compute a percentile (0.0 to 1.0) from lost latencies. Returns None if no lost elements.
    pub fn lost_latency_percentile(&self, p: f64) -> Option<u64> {
        percentile(&self.lost_latencies, p)
    }

    /// Get all stable latencies (unsorted).
    pub fn stable_latencies(&self) -> &[u64] {
        &self.stable_latencies
    }

    /// Get all lost latencies (unsorted).
    pub fn lost_latencies(&self) -> &[u64] {
        &self.lost_latencies
    }
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
        use ahash::HashSet;

        let mut elements: HashMap<T, ElementState<T>> = HashMap::new();
        let mut duplicates: HashMap<T, usize> = HashMap::new();

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
                        // Find the invocation for this read
                        let inv = match history.invocation(op) {
                            Some(i) => i,
                            None => continue,
                        };

                        // Handle both Set and Vec read values
                        let read_set: HashSet<T> = match &op.value {
                            OpValue::Set(s) => s.clone(),
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
    use ahash::HashSet;
    use std::time::Duration;
    use crate::history::ProcessId;

    /// Operation builder for tests. Mimics Jepsen's invoke-op/ok-op pattern.
    #[derive(Clone)]
    enum TestOp {
        AddInvoke(u64, i32),       // (process, value)
        AddOk(u64, i32),           // (process, value)
        ReadInvoke(u64),           // (process)
        ReadOk(u64, Vec<i32>),     // (process, values seen)
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
                TestOp::ReadOk(p, vals) => {
                    (OpType::Ok, OpFn::Read, OpValue::Set(HashSet::from_iter(vals)), p)
                }
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
    fn a() -> TestOp { TestOp::AddInvoke(0, 0) }
    fn a_ok() -> TestOp { TestOp::AddOk(0, 0) }
    fn r() -> TestOp { TestOp::ReadInvoke(1) }
    fn r_plus() -> TestOp { TestOp::ReadOk(1, vec![0]) }
    fn r_minus() -> TestOp { TestOp::ReadOk(1, vec![]) }

    // Multi-element helpers
    fn a0() -> TestOp { TestOp::AddInvoke(0, 0) }
    fn a0_ok() -> TestOp { TestOp::AddOk(0, 0) }
    fn a1() -> TestOp { TestOp::AddInvoke(1, 1) }
    fn a1_ok() -> TestOp { TestOp::AddOk(1, 1) }
    fn r2() -> TestOp { TestOp::ReadInvoke(2) }
    fn r3() -> TestOp { TestOp::ReadInvoke(3) }
    fn r2_empty() -> TestOp { TestOp::ReadOk(2, vec![]) }
    fn r2_0() -> TestOp { TestOp::ReadOk(2, vec![0]) }
    fn r2_1() -> TestOp { TestOp::ReadOk(2, vec![1]) }
    fn r2_01() -> TestOp { TestOp::ReadOk(2, vec![0, 1]) }
    fn r3_1() -> TestOp { TestOp::ReadOk(3, vec![1]) }

    #[test]
    fn test_never_read() {
        // Add element, but never read it
        let result = check(vec![
            TestOp::AddInvoke(0, 0),
            TestOp::AddOk(0, 0),
        ]);

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
            a0(),        // 0
            a1(),        // 1
            r2(),        // 2
            r2_1(),      // 3: sees {1}
            a0_ok(),     // 4
            a1_ok(),     // 5
            r2(),        // 6
            r2_01(),     // 7: sees {0,1}
            r2(),        // 8
            r2_0(),      // 9: sees {0}
            r2(),        // 10
            r2_empty(),  // 11: sees {}
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
            a0(),        // 0: add 0 invoke
            a0_ok(),     // 1: add 0 ok
            a1(),        // 2: add 1 invoke
            r2(),        // 3: read invoke (process 2)
            r2_1(),      // 4: read ok, sees {1}
            a1_ok(),     // 5: add 1 ok
            r2(),        // 6: read invoke (process 2)
            r3(),        // 7: read invoke (process 3)
            r3_1(),      // 8: read ok (process 3), sees {1}
            r2_0(),      // 9: read ok (process 2), sees {0}
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
            a0(),        // 0
            a1(),        // 1
            r2(),        // 2
            r2_1(),      // 3: sees {1}
            a0_ok(),     // 4
            a1_ok(),     // 5
            r2(),        // 6
            r2_01(),     // 7: sees {0,1}
            r2(),        // 8
            r2_0(),      // 9: sees {0}
            r2(),        // 10
            r2_empty(),  // 11: sees {}
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
}
