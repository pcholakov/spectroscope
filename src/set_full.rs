//! Set-full checker.
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
//! These per-element outcomes measure eventual consistency. An element is **stale**
//! when some read invoked after it was known still missed it.
//!
//! When `linearizable` is enabled, the checker additionally decides whether the whole
//! history is linearizable. That is stronger than requiring every element to be
//! visible without staleness: a read returns the entire set at once, so every reader
//! must also agree on a single order in which elements appeared. A violation is
//! reported as a [`PrecedenceCycle`] of orderings that cannot all hold.
//!
//! Derived from [Jepsen's set-full
//! checker](https://jepsen-io.github.io/jepsen/jepsen.checker.html#var-set-full).

use std::hash::Hash;
use std::time::Duration;

use ahash::{HashMap, HashMapExt, HashSet, HashSetExt};

use crate::Validity;
use crate::history::{History, Op, OpType, Operation, Pid, Timestamp};

mod linearizability;

pub use linearizability::{Event, Precedence, PrecedenceCycle, Reason};

/// An operation on a grow-only set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetOp<T> {
    /// Add an element. Invocation and completion carry the same element.
    Add(T),
    /// Read the whole set. `None` until a successful completion returns the
    /// elements, in any order; an element listed twice is a duplicate.
    Read(Option<Vec<T>>),
}

impl<T: PartialEq> Operation for SetOp<T> {
    fn is_valid_invocation(&self) -> bool {
        matches!(self, Self::Add(_) | Self::Read(None))
    }

    fn is_valid_completion(&self, invocation: &Self, outcome: OpType) -> bool {
        match (invocation, self) {
            (Self::Add(requested), Self::Add(completed)) => requested == completed,
            (Self::Read(None), Self::Read(result)) => result.is_some() == (outcome == OpType::Ok),
            _ => false,
        }
    }
}

impl<T> Op<SetOp<T>> {
    /// Create an add invocation.
    pub fn add_invoke(index: usize, process: impl Into<Pid>, element: T) -> Self {
        Self::new(index, process, OpType::Invoke, SetOp::Add(element))
    }

    /// Create a successful add completion.
    pub fn add_ok(index: usize, process: impl Into<Pid>, element: T) -> Self {
        Self::new(index, process, OpType::Ok, SetOp::Add(element))
    }

    /// Create an add that definitely failed.
    pub fn add_fail(index: usize, process: impl Into<Pid>, element: T) -> Self {
        Self::new(index, process, OpType::Fail, SetOp::Add(element))
    }

    /// Create an add with indeterminate outcome (timeout, crash).
    pub fn add_info(index: usize, process: impl Into<Pid>, element: T) -> Self {
        Self::new(index, process, OpType::Info, SetOp::Add(element))
    }

    /// Create a read invocation.
    pub fn read_invoke(index: usize, process: impl Into<Pid>) -> Self {
        Self::new(index, process, OpType::Invoke, SetOp::Read(None))
    }

    /// Create a successful read completion returning `elements`.
    pub fn read_ok(
        index: usize,
        process: impl Into<Pid>,
        elements: impl IntoIterator<Item = T>,
    ) -> Self {
        let elements = elements.into_iter().collect();
        Self::new(index, process, OpType::Ok, SetOp::Read(Some(elements)))
    }

    /// Create a read that definitely failed.
    pub fn read_fail(index: usize, process: impl Into<Pid>) -> Self {
        Self::new(index, process, OpType::Fail, SetOp::Read(None))
    }

    /// Create a read with indeterminate outcome.
    pub fn read_info(index: usize, process: impl Into<Pid>) -> Self {
        Self::new(index, process, OpType::Info, SetOp::Read(None))
    }
}

/// Options for the set-full checker.
#[derive(Debug, Clone, Default)]
pub struct SetFullOptions {
    /// If true, the history must be linearizable.
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
struct ElementResult<T> {
    element: T,
    outcome: ElementOutcome,
    /// Latency in milliseconds until the element became stable (if stable).
    stable_latency_ms: Option<u64>,
    /// Latency in milliseconds until the element was lost (if lost).
    lost_latency_ms: Option<u64>,
    /// Index and time when the element was first known to exist.
    known_index: Option<usize>,
    known_time: Option<Duration>,
    /// Index and time of the last read that didn't observe the element.
    last_absent_index: Option<usize>,
    last_absent_time: Option<Duration>,
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
    /// Distinct elements that were added or observed.
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
    /// Stable elements that some read missed after they were known.
    pub stale: Vec<T>,
    /// Top 8 stale elements with highest latency, with detailed info.
    pub worst_stale: Vec<WorstStaleEntry<T>>,
    /// Elements with duplicates and their max multiplicity.
    pub duplicated: HashMap<T, usize>,
    /// Elements returned by a read although no add of them was invoked, or every
    /// add of them failed. Nothing in the history could have written them.
    pub observed_without_add: Vec<T>,
    /// In linearizable mode, a cycle proving the history's orderings contradict
    /// each other. Always `None` otherwise. A history can also be invalid without
    /// a cycle, through `lost`, `duplicated`, or `observed_without_add`.
    pub linearizability_violation: Option<PrecedenceCycle<T>>,
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
    time: Option<Timestamp>,
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
    fn on_add_ok(&mut self, op: &Op<SetOp<T>>) {
        if self.known.is_none() {
            self.known = Some(OpRef {
                index: op.index,
                time: op.time,
            });
        }
    }

    /// Process a read that observed this element.
    fn on_read_present(&mut self, inv: &Op<SetOp<T>>, completion: &Op<SetOp<T>>) {
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
    fn on_read_absent(&mut self, inv: &Op<SetOp<T>>) {
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
        // `None` orders before every index: no observation precedes any observation.
        let last_present_idx = self.last_present.as_ref().map(|p| p.index);
        let last_absent_idx = self.last_absent.as_ref().map(|a| a.index);
        let known_idx = self.known.as_ref().map(|k| k.index);

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
                la.time.map(|t| t.as_duration() + Duration::from_nanos(1))
            } else {
                Some(Duration::ZERO)
            };
            match (stable_time, known_time) {
                (Some(st), Some(kt)) => {
                    Some(st.saturating_sub(kt.as_duration()).as_millis() as u64)
                }
                _ => None, // Unknown latency when timestamps missing
            }
        } else {
            None
        };

        let lost_latency_ms = if lost {
            let lost_time = if let Some(ref lp) = self.last_present {
                lp.time.map(|t| t.as_duration() + Duration::from_nanos(1))
            } else {
                Some(Duration::ZERO)
            };
            match (lost_time, known_time) {
                (Some(lt), Some(kt)) => {
                    Some(lt.saturating_sub(kt.as_duration()).as_millis() as u64)
                }
                _ => None, // Unknown latency when timestamps missing
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
            known_time: self
                .known
                .as_ref()
                .and_then(|k| k.time.map(|t| t.as_duration())),
            last_absent_index: self.last_absent.as_ref().map(|a| a.index),
            last_absent_time: self
                .last_absent
                .as_ref()
                .and_then(|a| a.time.map(|t| t.as_duration())),
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
    pub fn check<T>(&self, history: &History<SetOp<T>>) -> SetFullResult<T>
    where
        T: Clone + Eq + Hash + Ord,
    {
        let mut elements: HashMap<T, ElementState<T>> = HashMap::new();
        let mut duplicates: HashMap<T, usize> = HashMap::new();
        // Elements with an add that did not fail.
        let mut attempted: HashSet<T> = HashSet::new();

        for (pos, op) in history.ops().iter().enumerate() {
            match (&op.value, op.op_type) {
                (SetOp::Add(element), OpType::Invoke) => {
                    elements
                        .entry(element.clone())
                        .or_insert_with(|| ElementState::new(element.clone()));
                    let outcome = history.completion(pos).map(|c| c.op_type);
                    if outcome != Some(OpType::Fail) {
                        attempted.insert(element.clone());
                    }
                }
                (SetOp::Add(element), OpType::Ok) => {
                    if let Some(state) = elements.get_mut(element) {
                        state.on_add_ok(op);
                    }
                }
                (SetOp::Read(Some(returned)), OpType::Ok) => {
                    let inv = history
                        .invocation(pos)
                        .expect("History pairs every completion with its invocation");

                    let mut freqs: HashMap<T, usize> = HashMap::new();
                    for elem in returned {
                        *freqs.entry(elem.clone()).or_insert(0) += 1;
                    }
                    // Track max multiplicity for elements returned more than once
                    for (elem, count) in &freqs {
                        if *count > 1 {
                            duplicates
                                .entry(elem.clone())
                                .and_modify(|c| *c = (*c).max(*count))
                                .or_insert(*count);
                        }
                    }
                    let read_set: HashSet<T> = freqs.into_keys().collect();

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
                _ => {}
            }
        }

        let mut observed_without_add: Vec<T> = elements
            .values()
            .filter(|s| s.last_present.is_some() && !attempted.contains(&s.element))
            .map(|s| s.element.clone())
            .collect();
        observed_without_add.sort();

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

        // A stale element was missed by a read invoked after it was known. History
        // timestamps never decrease, so this covers every non-zero stable latency.
        let mut stale_results: Vec<&ElementResult<T>> = results
            .iter()
            .filter(|r| {
                r.outcome == ElementOutcome::Stable
                    && r.last_absent_index
                        .is_some_and(|la| r.known_index.is_some_and(|k| la > k))
            })
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

        // Every stale or lost element also yields a linearizability violation.
        let linearizability_violation = if self.options.linearizable {
            linearizability::find_violation(history)
        } else {
            None
        };

        let valid = if !lost.is_empty()
            || !duplicates.is_empty()
            || !observed_without_add.is_empty()
            || linearizability_violation.is_some()
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
            observed_without_add,
            linearizability_violation,
            stable_latencies,
            lost_latencies,
        }
    }
}

/// Compute a percentile from a slice of values.
fn percentile(values: &[u64], p: f64) -> Option<u64> {
    if values.is_empty() || !p.is_finite() || !(0.0..=1.0).contains(&p) {
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
    /// Assigns indices sequentially and times of one millisecond per index.
    fn build_history(ops: Vec<TestOp>) -> History<SetOp<i32>> {
        let mut history = History::new();
        for (idx, test_op) in ops.into_iter().enumerate() {
            let op = match test_op {
                TestOp::AddInvoke(p, v) => Op::add_invoke(idx, p, v),
                TestOp::AddOk(p, v) => Op::add_ok(idx, p, v),
                TestOp::ReadInvoke(p) => Op::read_invoke(idx, p),
                TestOp::ReadOk(p, vals) => Op::read_ok(idx, p, vals),
            };
            history.push(op.at_millis(idx as u64)).unwrap();
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
        history.push(Op::add_invoke(0, 0u64, 1)).unwrap();
        history.push(Op::add_fail(1, 0u64, 1)).unwrap();
        history.push(Op::read_invoke(2, 0u64)).unwrap();
        history
            .push(Op::read_ok(3, 0u64, Vec::<i32>::new()))
            .unwrap();

        let result = SetFullChecker::default().check(&history);
        // Element was invoked but failed, so it's tracked but unknown outcome
        assert_eq!(result.attempt_count, 1);
        assert_eq!(result.never_read_count, 1);
    }

    #[test]
    fn test_info_operations_ignored() {
        // Info (indeterminate) operations should be handled gracefully
        let mut history = History::new();
        history.push(Op::add_invoke(0, 0u64, 1)).unwrap();
        history.push(Op::add_info(1, 0u64, 1)).unwrap();
        history.push(Op::read_invoke(2, 1u64)).unwrap();
        history.push(Op::read_ok(3, 1u64, vec![1])).unwrap();

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
        history
            .push(Op::add_invoke(0, 0u64, 0).at(Timestamp::from_millis(0)))
            .unwrap();
        history
            .push(Op::add_ok(1, 0u64, 0).at(Timestamp::from_millis(1)))
            .unwrap();
        history
            .push(Op::read_invoke(2, 1u64).at(Timestamp::from_millis(2)))
            .unwrap();
        // Read returns element 0 three times (duplicate!)
        history
            .push(Op::read_ok(3, 1u64, [0, 0, 0]).at(Timestamp::from_millis(3)))
            .unwrap();

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
            let ms = |n: usize| Timestamp::from_millis(n as u64);
            history
                .push(Op::add_invoke(elem * 2, elem as u64, elem as i32).at(ms(elem * 2)))
                .unwrap();
            history
                .push(Op::add_ok(elem * 2 + 1, elem as u64, elem as i32).at(ms(elem * 2 + 1)))
                .unwrap();
        }
        // Element 0 known at index 1 (1ms), Element 1 at index 3 (3ms), Element 2 at index 5 (5ms)

        // Read that misses all (creating absent records)
        history
            .push(Op::read_invoke(6, 10u64).at(Timestamp::from_millis(6)))
            .unwrap();
        history
            .push(Op::read_ok(7, 10u64, std::iter::empty::<i32>()).at(Timestamp::from_millis(7)))
            .unwrap();

        // Read that sees all (making them stable)
        history
            .push(Op::read_invoke(8, 10u64).at(Timestamp::from_millis(8)))
            .unwrap();
        history
            .push(Op::read_ok(9, 10u64, [0, 1, 2]).at(Timestamp::from_millis(9)))
            .unwrap();

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

    // ====== Bug fix regression tests ======

    #[test]
    fn test_percentile_invalid_p_returns_none() {
        let values = vec![1, 2, 3, 4, 5];

        // Valid p values should work
        assert!(percentile(&values, 0.0).is_some());
        assert!(percentile(&values, 0.5).is_some());
        assert!(percentile(&values, 1.0).is_some());

        // Invalid p values should return None
        assert!(percentile(&values, -0.1).is_none());
        assert!(percentile(&values, 1.1).is_none());
        assert!(percentile(&values, 2.0).is_none());
        assert!(percentile(&values, f64::NAN).is_none());
        assert!(percentile(&values, f64::INFINITY).is_none());
        assert!(percentile(&values, f64::NEG_INFINITY).is_none());

        // Empty values should still return None
        assert!(percentile(&[], 0.5).is_none());
    }

    #[test]
    fn test_linearizable_stale_detection_without_timestamps() {
        // Test that stale elements are detected even without timestamps,
        // using index-based ordering as fallback
        let mut history = History::new();

        // Add element without timestamps
        history.push(Op::add_invoke(0, 0u64, 1)).unwrap();
        history.push(Op::add_ok(1, 0u64, 1)).unwrap();

        // Read after add completes that misses the element (no timestamps)
        history.push(Op::read_invoke(2, 1u64)).unwrap();
        history
            .push(Op::read_ok(3, 1u64, std::iter::empty::<i32>()))
            .unwrap();

        // Later read that sees the element
        history.push(Op::read_invoke(4, 1u64)).unwrap();
        history.push(Op::read_ok(5, 1u64, [1])).unwrap();

        // In eventual consistency mode, this is valid (stale but not a violation)
        let eventual_result = SetFullChecker::default().check(&history);
        assert_eq!(eventual_result.valid, Validity::Valid);
        assert_eq!(eventual_result.stale_count, 1);

        // In linearizable mode, stale elements cause failure
        let linearizable_result = SetFullChecker::linearizable().check(&history);
        assert_eq!(linearizable_result.valid, Validity::Invalid);
        assert_eq!(linearizable_result.stale_count, 1);

        // Verify the element is detected as stale via index ordering
        // (stable_latency_ms is None because no timestamps, but last_absent_index > known_index)
        assert!(linearizable_result.stale.contains(&1));
    }

    #[test]
    fn test_linearizable_no_false_stale_without_timestamps() {
        // Ensure we don't falsely detect staleness when element appears immediately
        let mut history = History::new();

        // Add element without timestamps
        history.push(Op::add_invoke(0, 0u64, 1)).unwrap();
        history.push(Op::add_ok(1, 0u64, 1)).unwrap();

        // First read after add sees the element (no staleness)
        history.push(Op::read_invoke(2, 1u64)).unwrap();
        history.push(Op::read_ok(3, 1u64, [1])).unwrap();

        let linearizable_result = SetFullChecker::linearizable().check(&history);
        assert_eq!(linearizable_result.valid, Validity::Valid);
        assert_eq!(linearizable_result.stale_count, 0);
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

    // ====== Linearizability ======

    fn check_linearizable(ops: impl IntoIterator<Item = Op<SetOp<i32>>>) -> SetFullResult<i32> {
        SetFullChecker::linearizable().check(&History::from_ops(ops).unwrap())
    }

    fn read(invoke_index: usize, complete_index: usize) -> Event<i32> {
        Event::Read {
            invoke_index,
            complete_index,
        }
    }

    fn precedes(before: Event<i32>, after: Event<i32>, reason: Reason) -> Precedence<i32> {
        Precedence {
            before,
            after,
            reason,
        }
    }

    fn real_time(completed_index: usize, invoked_index: usize) -> Reason {
        Reason::RealTime {
            completed_index,
            invoked_index,
        }
    }

    /// Concurrent reads disagree on the order in which concurrent adds took effect.
    /// Each element is visible without staleness, so only a whole-history check sees it.
    #[test]
    fn test_divergent_reads_not_linearizable() {
        let ops = [
            Op::add_invoke(0, 0u64, 1),
            Op::add_invoke(1, 1u64, 2),
            Op::read_invoke(2, 2u64),
            Op::read_invoke(3, 3u64),
            Op::read_ok(4, 2u64, [1]),
            Op::read_ok(5, 3u64, [2]),
            Op::add_ok(6, 0u64, 1),
            Op::add_ok(7, 1u64, 2),
        ];
        let history = History::from_ops(ops.clone()).unwrap();
        assert_eq!(
            SetFullChecker::default().check(&history).valid,
            Validity::Valid
        );

        let result = check_linearizable(ops);
        assert_eq!(result.valid, Validity::Invalid);
        assert_eq!(result.stale_count, 0);
        assert_eq!(
            result.linearizability_violation.unwrap().edges,
            vec![
                precedes(Event::Add(1), read(2, 4), Reason::Observed),
                precedes(read(2, 4), Event::Add(2), Reason::Missed),
                precedes(Event::Add(2), read(3, 5), Reason::Observed),
                precedes(read(3, 5), Event::Add(1), Reason::Missed),
            ]
        );
    }

    /// A read returns an element whose add had not been invoked yet.
    #[test]
    fn test_read_from_future_not_linearizable() {
        let result = check_linearizable([
            Op::read_invoke(0, 0u64),
            Op::read_ok(1, 0u64, [1]),
            Op::add_invoke(2, 1u64, 1),
            Op::add_ok(3, 1u64, 1),
        ]);
        assert_eq!(result.valid, Validity::Invalid);
        assert_eq!(
            result.linearizability_violation.unwrap().edges,
            vec![
                precedes(Event::Add(1), read(0, 1), Reason::Observed),
                precedes(read(0, 1), Event::Add(1), real_time(1, 2)),
            ]
        );
    }

    /// A read invoked after another completed returns an older state of the set.
    #[test]
    fn test_reads_going_backwards_not_linearizable() {
        let result = check_linearizable([
            Op::add_invoke(0, 0u64, 1),
            Op::add_ok(1, 0u64, 1),
            Op::add_invoke(2, 1u64, 2),
            Op::read_invoke(3, 2u64),
            Op::read_ok(4, 2u64, [1, 2]),
            Op::read_invoke(5, 3u64),
            Op::read_ok(6, 3u64, [1]),
        ]);
        assert_eq!(result.valid, Validity::Invalid);
        assert_eq!(
            result.linearizability_violation.unwrap().edges,
            vec![
                precedes(read(5, 6), Event::Add(2), Reason::Missed),
                precedes(Event::Add(2), read(3, 4), Reason::Observed),
                precedes(read(3, 4), read(5, 6), real_time(4, 5)),
            ]
        );
    }

    /// Add 2 completes before add 1 is invoked, yet a read sees 1 without 2.
    /// The read was invoked before add 2 completed, so neither element is stale.
    #[test]
    fn test_adds_out_of_order_not_linearizable() {
        let ops = [
            Op::read_invoke(0, 0u64),
            Op::add_invoke(1, 1u64, 2),
            Op::add_ok(2, 1u64, 2),
            Op::add_invoke(3, 2u64, 1),
            Op::read_ok(4, 0u64, [1]),
            Op::add_ok(5, 2u64, 1),
            Op::read_invoke(6, 3u64),
            Op::read_ok(7, 3u64, [1, 2]),
        ];
        let history = History::from_ops(ops.clone()).unwrap();
        assert_eq!(
            SetFullChecker::default().check(&history).valid,
            Validity::Valid
        );

        let result = check_linearizable(ops);
        assert_eq!(result.valid, Validity::Invalid);
        assert_eq!(
            result.linearizability_violation.unwrap().edges,
            vec![
                precedes(Event::Add(1), read(0, 4), Reason::Observed),
                precedes(read(0, 4), Event::Add(2), Reason::Missed),
                precedes(Event::Add(2), Event::Add(1), real_time(2, 3)),
            ]
        );
    }

    /// A stale read is the simplest violation: one missed element, one real-time edge.
    #[test]
    fn test_stale_read_witness() {
        let result = SetFullChecker::linearizable().check(&build_history(vec![
            TestOp::AddInvoke(0, 1),
            TestOp::AddOk(0, 1),
            TestOp::ReadInvoke(1),
            TestOp::ReadOk(1, vec![]),
            TestOp::ReadInvoke(2),
            TestOp::ReadOk(2, vec![1]),
        ]));
        assert_eq!(
            result.linearizability_violation.unwrap().edges,
            vec![
                precedes(read(2, 3), Event::Add(1), Reason::Missed),
                precedes(Event::Add(1), read(2, 3), real_time(1, 2)),
            ]
        );
    }

    /// Indices are arbitrary increasing `usize`s, so outcomes must not depend on
    /// them fitting in a signed integer of the same width.
    #[test]
    fn test_indices_above_isize_max() {
        let b = isize::MAX as usize + 1;
        let history = History::from_ops([
            Op::read_invoke(0, 0u64),
            Op::add_invoke(1, 1u64, 1),
            Op::add_ok(b, 1u64, 1),
            Op::read_invoke(b + 1, 2u64),
            Op::read_ok(b + 2, 2u64, [1]),
            Op::read_ok(b + 3, 0u64, []),
        ])
        .unwrap();

        for checker in [SetFullChecker::default(), SetFullChecker::linearizable()] {
            let result = checker.check(&history);
            assert_eq!(result.valid, Validity::Valid);
            assert_eq!(result.stable_count, 1);
        }
    }

    /// A read returning an element that nothing in the history could have added
    /// is invalid in both modes, even if every read agrees on it.
    #[test]
    fn test_observed_without_add_invalid() {
        let cases: [(&str, Vec<Op<SetOp<i32>>>); 3] = [
            (
                "only add failed",
                vec![Op::add_invoke(0, 0u64, 1), Op::add_fail(1, 0u64, 1)],
            ),
            ("never added", vec![]),
            (
                "only a different element added",
                vec![Op::add_invoke(0, 0u64, 2), Op::add_ok(1, 0u64, 2)],
            ),
        ];
        for (name, adds) in cases {
            let reads = [
                Op::read_invoke(10, 1u64),
                Op::read_ok(11, 1u64, [1]),
                Op::read_invoke(12, 1u64),
                Op::read_ok(13, 1u64, [1]),
            ];
            let history = History::from_ops(adds.into_iter().chain(reads)).unwrap();
            for checker in [SetFullChecker::default(), SetFullChecker::linearizable()] {
                let result = checker.check(&history);
                assert_eq!(result.valid, Validity::Invalid, "{name}");
                assert_eq!(result.observed_without_add, vec![1], "{name}");
            }
        }
    }

    mod property_tests {
        use super::*;
        use proptest::prelude::*;
        use rand::seq::IndexedRandom;
        use rand::{Rng, SeedableRng};
        use std::collections::BTreeSet;

        #[derive(Debug, Clone)]
        struct GenConfig {
            /// Scheduler steps; each invokes, applies, or completes one op.
            steps: usize,
            /// Concurrent client slots. A crashed slot continues under a new pid.
            processes: usize,
            /// Probability that a new op is an add rather than a read.
            add_ratio: f64,
            /// Probability that an add re-adds an existing element.
            readd_ratio: f64,
            /// Probability that an add fails before taking effect.
            fail_ratio: f64,
            /// Probability that an op completes with `Info`.
            crash_ratio: f64,
        }

        #[derive(Debug, Clone)]
        enum Phase {
            Idle,
            AddInvoked(i32),
            AddApplied(i32),
            ReadInvoked,
            ReadApplied(Vec<i32>),
        }

        /// Generates a linearizable history by running a sequential set model.
        /// Each op takes effect at a scheduler step strictly between its invocation
        /// and completion; pending ops may or may not have taken effect.
        fn gen_linearizable(config: &GenConfig, seed: u64) -> Vec<Op<SetOp<i32>>> {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let mut set: BTreeSet<i32> = BTreeSet::new();
            let mut added: Vec<i32> = Vec::new();
            let mut slots: Vec<(u64, Phase)> = (0..config.processes as u64)
                .map(|pid| (pid, Phase::Idle))
                .collect();
            let mut next_pid = config.processes as u64;
            let mut ops = Vec::new();

            for _ in 0..config.steps {
                let slot = rng.random_range(0..slots.len());
                let (pid, phase) = slots[slot].clone();
                let index = ops.len();
                let crash = rng.random_bool(config.crash_ratio);
                let (emitted, next) = match phase {
                    Phase::Idle if rng.random_bool(config.add_ratio) => {
                        let element = match added.choose(&mut rng) {
                            Some(&e) if rng.random_bool(config.readd_ratio) => e,
                            _ => added.len() as i32,
                        };
                        added.push(element);
                        (
                            Some(Op::add_invoke(index, pid, element)),
                            Phase::AddInvoked(element),
                        )
                    }
                    Phase::Idle => (Some(Op::read_invoke(index, pid)), Phase::ReadInvoked),
                    Phase::AddInvoked(element) if rng.random_bool(config.fail_ratio) => {
                        (Some(Op::add_fail(index, pid, element)), Phase::Idle)
                    }
                    Phase::AddInvoked(element) | Phase::AddApplied(element) if crash => {
                        (Some(Op::add_info(index, pid, element)), Phase::Idle)
                    }
                    Phase::AddInvoked(element) => {
                        set.insert(element);
                        (None, Phase::AddApplied(element))
                    }
                    Phase::AddApplied(element) => {
                        (Some(Op::add_ok(index, pid, element)), Phase::Idle)
                    }
                    Phase::ReadInvoked | Phase::ReadApplied(_) if crash => {
                        (Some(Op::read_info(index, pid)), Phase::Idle)
                    }
                    Phase::ReadInvoked => (None, Phase::ReadApplied(set.iter().copied().collect())),
                    Phase::ReadApplied(state) => {
                        (Some(Op::read_ok(index, pid, state)), Phase::Idle)
                    }
                };
                if let Some(op) = emitted {
                    if op.op_type == OpType::Info {
                        next_pid += 1;
                        slots[slot].0 = next_pid;
                    }
                    ops.push(op.at_millis(index as u64));
                }
                slots[slot].1 = next;
            }
            ops
        }

        fn config_strategy() -> impl Strategy<Value = GenConfig> {
            (
                50usize..600,
                1usize..16,
                0.2f64..0.8,
                0.0f64..0.3,
                0.0f64..0.1,
                0.0f64..0.1,
            )
                .prop_map(
                    |(steps, processes, add_ratio, readd_ratio, fail_ratio, crash_ratio)| {
                        GenConfig {
                            steps,
                            processes,
                            add_ratio,
                            readd_ratio,
                            fail_ratio,
                            crash_ratio,
                        }
                    },
                )
        }

        fn read_value(op: &Op<SetOp<i32>>) -> &[i32] {
            match &op.value {
                SetOp::Read(Some(returned)) => returned,
                other => panic!("expected a read result, got {other:?}"),
            }
        }

        /// Positions of successful reads and their invocations.
        fn successful_reads(ops: &[Op<SetOp<i32>>]) -> Vec<(usize, usize)> {
            let history = History::from_ops(ops.to_vec()).unwrap();
            (0..ops.len())
                .filter(|&pos| matches!(ops[pos].value, SetOp::Read(Some(_))))
                .map(|pos| (history.invocation(pos).unwrap().index, pos))
                .collect()
        }

        /// Removes from some read an element whose add completed before the read
        /// was invoked. Returns `None` if no read has such an element.
        fn inject_stale_read(ops: &mut [Op<SetOp<i32>>], rng: &mut impl Rng) -> Option<()> {
            let candidates: Vec<(usize, i32)> = successful_reads(ops)
                .into_iter()
                .flat_map(|(invoked, pos)| {
                    ops[..invoked]
                        .iter()
                        .filter(|op| op.op_type == OpType::Ok)
                        .filter_map(move |op| match op.value {
                            SetOp::Add(element) => Some((pos, element)),
                            SetOp::Read(_) => None,
                        })
                })
                .collect();
            let &(pos, element) = candidates.choose(rng)?;
            let state = read_value(&ops[pos])
                .iter()
                .copied()
                .filter(|&e| e != element);
            ops[pos].value = SetOp::Read(Some(state.collect()));
            Some(())
        }

        /// Adds to some read an element whose every non-failed add was invoked
        /// after the read completed. Returns `None` if there is no such pair.
        fn inject_future_read(ops: &mut [Op<SetOp<i32>>], rng: &mut impl Rng) -> Option<()> {
            let history = History::from_ops(ops.to_vec()).unwrap();
            let mut first_attempt: Vec<(i32, usize)> = Vec::new();
            for (pos, op) in ops.iter().enumerate() {
                let failed = history.completion(pos).map(|c| c.op_type) == Some(OpType::Fail);
                if let (SetOp::Add(element), OpType::Invoke, false) =
                    (&op.value, op.op_type, failed)
                {
                    let element = *element;
                    if !first_attempt.iter().any(|(e, _)| *e == element) {
                        first_attempt.push((element, op.index));
                    }
                }
            }
            let candidates: Vec<(usize, i32)> = successful_reads(ops)
                .into_iter()
                .flat_map(|(_, pos)| {
                    first_attempt
                        .iter()
                        .filter(move |(_, invoked)| *invoked > pos)
                        .map(move |(element, _)| (pos, *element))
                })
                .collect();
            let &(pos, element) = candidates.choose(rng)?;
            let mut state = read_value(&ops[pos]).to_vec();
            state.push(element);
            ops[pos].value = SetOp::Read(Some(state));
            Some(())
        }

        /// Asserts that every edge of `cycle` is a fact about `history` and the
        /// edges close into a cycle, so the cycle proves non-linearizability.
        fn assert_proves_violation(history: &History<SetOp<i32>>, cycle: &PrecedenceCycle<i32>) {
            let ops = history.ops();
            // Resolves a read event to its result, asserting that the event names a
            // successful read and its actual invocation.
            let read_state = |event: &Event<i32>| match event {
                Event::Read {
                    invoke_index,
                    complete_index,
                } => {
                    let pos = ops
                        .iter()
                        .position(|op| op.index == *complete_index)
                        .unwrap_or_else(|| panic!("no op {complete_index}"));
                    let op = &ops[pos];
                    assert!(matches!(op.value, SetOp::Read(Some(_))), "{event:?}");
                    assert_eq!(history.invocation(pos).unwrap().index, *invoke_index);
                    read_value(op).to_vec()
                }
                Event::Add(_) => panic!("expected a read, got {event:?}"),
            };
            let add_bounds = |element: i32| {
                let mut start = None;
                let mut end = None;
                for (pos, op) in ops.iter().enumerate() {
                    if op.op_type != OpType::Invoke || op.value != SetOp::Add(element) {
                        continue;
                    }
                    let completion = history.completion(pos);
                    if completion.is_some_and(|c| c.op_type == OpType::Fail) {
                        continue;
                    }
                    start = start.or(Some(op.index));
                    if let Some(c) = completion.filter(|c| c.op_type == OpType::Ok) {
                        end = Some(end.map_or(c.index, |e: usize| e.min(c.index)));
                    }
                }
                (start, end)
            };

            let edges = &cycle.edges;
            assert!(!edges.is_empty());
            for (i, edge) in edges.iter().enumerate() {
                assert_eq!(edge.after, edges[(i + 1) % edges.len()].before, "{cycle:?}");
                for event in [&edge.before, &edge.after] {
                    if let Event::Read { .. } = event {
                        read_state(event);
                    }
                }
                match edge.reason {
                    Reason::Observed => {
                        let Event::Add(element) = edge.before else {
                            panic!("{edge:?}")
                        };
                        assert!(read_state(&edge.after).contains(&element), "{edge:?}");
                    }
                    Reason::Missed => {
                        let Event::Add(element) = edge.after else {
                            panic!("{edge:?}")
                        };
                        assert!(!read_state(&edge.before).contains(&element), "{edge:?}");
                        let observed = ops.iter().any(|op| {
                            matches!(&op.value, SetOp::Read(Some(returned)) if returned.contains(&element))
                        });
                        assert!(observed || add_bounds(element).1.is_some(), "{edge:?}");
                    }
                    Reason::RealTime {
                        completed_index,
                        invoked_index,
                    } => {
                        assert!(completed_index < invoked_index, "{edge:?}");
                        let completed = match edge.before {
                            Event::Read { complete_index, .. } => Some(complete_index),
                            Event::Add(element) => add_bounds(element).1,
                        };
                        let invoked = match edge.after {
                            Event::Read { invoke_index, .. } => Some(invoke_index),
                            Event::Add(element) => add_bounds(element).0,
                        };
                        assert_eq!(completed, Some(completed_index), "{edge:?}");
                        assert_eq!(invoked, Some(invoked_index), "{edge:?}");
                    }
                }
            }
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(64))]

            #[test]
            fn prop_linearizable_histories_pass(config in config_strategy(), seed in any::<u64>()) {
                let history = History::from_ops(gen_linearizable(&config, seed)).unwrap();
                let result = SetFullChecker::linearizable().check(&history);

                prop_assert_eq!(result.linearizability_violation, None);
                prop_assert_ne!(result.valid, Validity::Invalid);
                prop_assert_eq!(result.stale_count, 0);
                prop_assert_eq!(result.lost_count, 0);
            }

            #[test]
            fn prop_stale_reads_detected(config in config_strategy(), seed in any::<u64>()) {
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                let mut ops = gen_linearizable(&config, seed);
                prop_assume!(inject_stale_read(&mut ops, &mut rng).is_some());
                let history = History::from_ops(ops).unwrap();

                let eventual = SetFullChecker::default().check(&history);
                prop_assert!(eventual.stale_count + eventual.lost_count > 0);

                let result = SetFullChecker::linearizable().check(&history);
                prop_assert_eq!(result.valid, Validity::Invalid);
                assert_proves_violation(&history, result.linearizability_violation.as_ref().unwrap());
            }

            #[test]
            fn prop_future_reads_detected(config in config_strategy(), seed in any::<u64>()) {
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                let mut ops = gen_linearizable(&config, seed);
                prop_assume!(inject_future_read(&mut ops, &mut rng).is_some());
                let history = History::from_ops(ops).unwrap();

                let result = SetFullChecker::linearizable().check(&history);
                prop_assert_eq!(result.valid, Validity::Invalid);
                assert_proves_violation(&history, result.linearizability_violation.as_ref().unwrap());
            }
        }

        /// The same seed must produce the same history, so failures reproduce.
        #[test]
        fn test_generator_deterministic() {
            let config = GenConfig {
                steps: 500,
                processes: 8,
                add_ratio: 0.5,
                readd_ratio: 0.2,
                fail_ratio: 0.05,
                crash_ratio: 0.05,
            };
            let render = |ops: Vec<Op<SetOp<i32>>>| format!("{ops:?}");
            assert_eq!(
                render(gen_linearizable(&config, 7)),
                render(gen_linearizable(&config, 7))
            );
        }
    }
}
