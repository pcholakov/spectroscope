//! History representation for consistency checking.
//!
//! Operations in a history follow a request/response model:
//! - `Invoke` marks the start of an operation
//! - `Ok` marks successful completion
//! - `Fail` marks a definite failure
//! - `Info` marks an indeterminate result (crash, timeout, etc.)

use ahash::{HashMap, HashMapExt, HashSet};
use std::time::Duration;

/// Process or thread identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Pid(pub u64);

impl From<u64> for Pid {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

/// A timestamp for operation ordering, relative to an arbitrary epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Timestamp(Duration);

impl Timestamp {
    /// Create a timestamp from milliseconds.
    #[must_use]
    pub fn from_millis(ms: u64) -> Self {
        Self(Duration::from_millis(ms))
    }

    /// Get the underlying Duration.
    #[must_use]
    pub fn as_duration(&self) -> Duration {
        self.0
    }
}

impl From<Duration> for Timestamp {
    fn from(d: Duration) -> Self {
        Self(d)
    }
}

/// The type/phase of an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpType {
    /// Operation was invoked but hasn't completed yet.
    Invoke,
    /// Operation completed successfully.
    Ok,
    /// Operation definitely failed.
    Fail,
    /// Operation result is indeterminate (e.g., timeout, crash).
    Info,
}

/// The function being performed by an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpFn {
    /// Add an element to the set.
    Add,
    /// Read the current set contents.
    Read,
}

/// A single operation in a history.
#[derive(Debug, Clone)]
pub struct Op<T> {
    /// Unique index of this operation in the history.
    pub index: usize,
    /// The type/phase of this operation.
    pub op_type: OpType,
    /// The function being performed.
    pub f: OpFn,
    /// The value associated with this operation.
    /// For Add: the element being added.
    /// For Read: None on invoke, Some(set contents) on completion.
    pub value: OpValue<T>,
    /// Timestamp (optional, used for latency calculations).
    pub time: Option<Timestamp>,
    /// Process/thread that performed this operation.
    pub process: Pid,
}

impl<T> Op<T> {
    /// Create an add invocation.
    pub fn add_invoke(index: usize, process: impl Into<Pid>, value: T) -> Self {
        Self {
            index,
            op_type: OpType::Invoke,
            f: OpFn::Add,
            value: OpValue::Single(value),
            time: None,
            process: process.into(),
        }
    }

    /// Create an add completion.
    pub fn add_ok(index: usize, process: impl Into<Pid>, value: T) -> Self {
        Self {
            index,
            op_type: OpType::Ok,
            f: OpFn::Add,
            value: OpValue::Single(value),
            time: None,
            process: process.into(),
        }
    }

    /// Create a read invocation.
    pub fn read_invoke(index: usize, process: impl Into<Pid>) -> Self {
        Self {
            index,
            op_type: OpType::Invoke,
            f: OpFn::Read,
            value: OpValue::None,
            time: None,
            process: process.into(),
        }
    }

    /// Create a read completion with observed values.
    pub fn read_ok(
        index: usize,
        process: impl Into<Pid>,
        values: impl IntoIterator<Item = T>,
    ) -> Self {
        Self {
            index,
            op_type: OpType::Ok,
            f: OpFn::Read,
            value: OpValue::Vec(values.into_iter().collect()),
            time: None,
            process: process.into(),
        }
    }

    /// Create an add with indeterminate outcome (timeout, crash).
    pub fn add_info(index: usize, process: impl Into<Pid>, value: T) -> Self {
        Self {
            index,
            op_type: OpType::Info,
            f: OpFn::Add,
            value: OpValue::Single(value),
            time: None,
            process: process.into(),
        }
    }

    /// Create an add that definitely failed.
    pub fn add_fail(index: usize, process: impl Into<Pid>, value: T) -> Self {
        Self {
            index,
            op_type: OpType::Fail,
            f: OpFn::Add,
            value: OpValue::Single(value),
            time: None,
            process: process.into(),
        }
    }

    /// Create a read with indeterminate outcome.
    pub fn read_info(index: usize, process: impl Into<Pid>) -> Self {
        Self {
            index,
            op_type: OpType::Info,
            f: OpFn::Read,
            value: OpValue::None,
            time: None,
            process: process.into(),
        }
    }

    /// Create a read that definitely failed.
    pub fn read_fail(index: usize, process: impl Into<Pid>) -> Self {
        Self {
            index,
            op_type: OpType::Fail,
            f: OpFn::Read,
            value: OpValue::None,
            time: None,
            process: process.into(),
        }
    }

    /// Set the timestamp for this operation.
    #[must_use]
    pub fn at(mut self, time: Timestamp) -> Self {
        self.time = Some(time);
        self
    }

    /// Set the timestamp for this operation in milliseconds.
    #[must_use]
    pub fn at_millis(self, ms: u64) -> Self {
        self.at(Timestamp::from_millis(ms))
    }
}

/// Value associated with an operation.
#[derive(Debug, Clone)]
pub enum OpValue<T> {
    /// A single element (for Add operations).
    Single(T),
    /// A set of elements (for Read operations, no duplicates).
    Set(HashSet<T>),
    /// A list of elements (for Read operations, may have duplicates).
    Vec(Vec<T>),
    /// No value (for Read invocations).
    None,
}

impl<T> OpValue<T> {
    pub fn as_single(&self) -> Option<&T> {
        match self {
            OpValue::Single(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_set(&self) -> Option<&HashSet<T>> {
        match self {
            OpValue::Set(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_vec(&self) -> Option<&Vec<T>> {
        match self {
            OpValue::Vec(v) => Some(v),
            _ => None,
        }
    }
}

/// Why an operation cannot be appended to a [`History`].
///
/// Indices refer to [`Op::index`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HistoryError {
    /// The op's index does not exceed the preceding op's index.
    IndexNotIncreasing { index: usize, previous: usize },
    /// The op's timestamp is earlier than a preceding op's timestamp.
    TimeDecreasing { index: usize },
    /// The op's value does not have the shape its function and type require.
    InvalidValue { index: usize },
    /// The process invoked an op while its op at `outstanding` had not completed.
    ConcurrentInvoke {
        index: usize,
        process: Pid,
        outstanding: usize,
    },
    /// The process invoked an op after its op at `crashed_at` completed with
    /// [`OpType::Info`]. A crashed process must continue under a new [`Pid`].
    InvokeAfterCrash {
        index: usize,
        process: Pid,
        crashed_at: usize,
    },
    /// The process has no outstanding invocation for this completion.
    UnmatchedCompletion { index: usize, process: Pid },
    /// The completion's function or added element differs from its invocation's.
    CompletionMismatch { index: usize, invocation: usize },
}

impl std::fmt::Display for HistoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IndexNotIncreasing { index, previous } => {
                write!(f, "op {index} does not follow op {previous}")
            }
            Self::TimeDecreasing { index } => {
                write!(f, "op {index} is timestamped before a preceding op")
            }
            Self::InvalidValue { index } => {
                write!(f, "op {index} has a value of the wrong shape")
            }
            Self::ConcurrentInvoke {
                index,
                process,
                outstanding,
            } => write!(
                f,
                "op {index}: process {} invoked while op {outstanding} is outstanding",
                process.0
            ),
            Self::InvokeAfterCrash {
                index,
                process,
                crashed_at,
            } => write!(
                f,
                "op {index}: process {} invoked after crashing at op {crashed_at}",
                process.0
            ),
            Self::UnmatchedCompletion { index, process } => write!(
                f,
                "op {index}: process {} has no outstanding invocation",
                process.0
            ),
            Self::CompletionMismatch { index, invocation } => {
                write!(f, "op {index} does not complete op {invocation}")
            }
        }
    }
}

impl std::error::Error for HistoryError {}

/// A well-formed history of operations, in real-time order.
///
/// Every op is validated as it is appended, so checkers can rely on these invariants:
/// - [`Op::index`] strictly increases and timestamps never decrease.
/// - Each process has at most one outstanding invocation, and every completion
///   is paired with that invocation. Invocations may remain pending at the end.
/// - A process whose op completed with [`OpType::Info`] invokes nothing further.
/// - Adds carry [`OpValue::Single`]; successful reads carry [`OpValue::Set`] or
///   [`OpValue::Vec`]; other reads carry [`OpValue::None`].
#[derive(Debug, Clone, Default)]
pub struct History<T> {
    ops: Vec<Op<T>>,
    /// Position of each op's paired invocation or completion.
    partners: Vec<Option<usize>>,
    /// Position of each process's outstanding invocation.
    outstanding: HashMap<Pid, usize>,
    /// Index of the `Info` completion that crashed each process.
    crashed: HashMap<Pid, usize>,
    latest_time: Option<Timestamp>,
}

impl<T> History<T> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            partners: Vec::new(),
            outstanding: HashMap::new(),
            crashed: HashMap::new(),
            latest_time: None,
        }
    }

    #[must_use]
    pub fn ops(&self) -> &[Op<T>] {
        &self.ops
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// The invocation of the completion at `completion_pos`, or `None` if that
    /// position holds no completion.
    #[must_use]
    pub fn invocation(&self, completion_pos: usize) -> Option<&Op<T>> {
        self.partner(completion_pos, |op| op.op_type != OpType::Invoke)
    }

    /// The completion of the invocation at `invocation_pos`, or `None` if that
    /// position holds no invocation or the invocation is still pending.
    #[must_use]
    pub fn completion(&self, invocation_pos: usize) -> Option<&Op<T>> {
        self.partner(invocation_pos, |op| op.op_type == OpType::Invoke)
    }

    fn partner(&self, pos: usize, accept: impl Fn(&Op<T>) -> bool) -> Option<&Op<T>> {
        let op = self.ops.get(pos)?;
        if !accept(op) {
            return None;
        }
        self.partners[pos].map(|partner| &self.ops[partner])
    }
}

impl<T: PartialEq> History<T> {
    /// Build a history from ops in real-time order.
    pub fn from_ops(ops: impl IntoIterator<Item = Op<T>>) -> Result<Self, HistoryError> {
        let mut history = Self::new();
        for op in ops {
            history.push(op)?;
        }
        Ok(history)
    }

    /// Append the next op in real-time order. On error the history is unchanged.
    pub fn push(&mut self, op: Op<T>) -> Result<(), HistoryError> {
        let index = op.index;
        if let Some(previous) = self.ops.last()
            && index <= previous.index
        {
            return Err(HistoryError::IndexNotIncreasing {
                index,
                previous: previous.index,
            });
        }
        if let (Some(time), Some(latest)) = (op.time, self.latest_time)
            && time < latest
        {
            return Err(HistoryError::TimeDecreasing { index });
        }
        if !has_valid_shape(&op) {
            return Err(HistoryError::InvalidValue { index });
        }

        let pos = self.ops.len();
        let process = op.process;
        let partner = if op.op_type == OpType::Invoke {
            if let Some(&outstanding) = self.outstanding.get(&process) {
                return Err(HistoryError::ConcurrentInvoke {
                    index,
                    process,
                    outstanding: self.ops[outstanding].index,
                });
            }
            if let Some(&crashed_at) = self.crashed.get(&process) {
                return Err(HistoryError::InvokeAfterCrash {
                    index,
                    process,
                    crashed_at,
                });
            }
            self.outstanding.insert(process, pos);
            None
        } else {
            let Some(&invocation_pos) = self.outstanding.get(&process) else {
                return Err(HistoryError::UnmatchedCompletion { index, process });
            };
            let invocation = &self.ops[invocation_pos];
            if invocation.f != op.f
                || (op.f == OpFn::Add && invocation.value.as_single() != op.value.as_single())
            {
                return Err(HistoryError::CompletionMismatch {
                    index,
                    invocation: invocation.index,
                });
            }
            self.outstanding.remove(&process);
            if op.op_type == OpType::Info {
                self.crashed.insert(process, index);
            }
            self.partners[invocation_pos] = Some(pos);
            Some(invocation_pos)
        };

        self.latest_time = op.time.or(self.latest_time);
        self.ops.push(op);
        self.partners.push(partner);
        Ok(())
    }
}

fn has_valid_shape<T>(op: &Op<T>) -> bool {
    matches!(
        (op.f, op.op_type, &op.value),
        (OpFn::Add, _, OpValue::Single(_))
            | (OpFn::Read, OpType::Ok, OpValue::Set(_) | OpValue::Vec(_))
            | (
                OpFn::Read,
                OpType::Invoke | OpType::Fail | OpType::Info,
                OpValue::None
            )
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pairs_completions_with_invocations() {
        // Process 0's add is still in flight while process 1 reads.
        let history = History::from_ops([
            Op::add_invoke(0, 0u64, 1),
            Op::read_invoke(1, 1u64),
            Op::read_ok(2, 1u64, [1]),
            Op::add_ok(3, 0u64, 1),
            Op::read_invoke(4, 1u64),
        ])
        .unwrap();

        assert_eq!(history.invocation(2).map(|op| op.index), Some(1));
        assert_eq!(history.invocation(3).map(|op| op.index), Some(0));
        assert_eq!(history.completion(0).map(|op| op.index), Some(3));
        assert_eq!(history.completion(1).map(|op| op.index), Some(2));
        // Pending invocation, a completion queried as an invocation, out of range.
        assert!(history.completion(4).is_none());
        assert!(history.completion(2).is_none());
        assert!(history.invocation(0).is_none());
        assert!(history.invocation(history.len()).is_none());
    }

    #[test]
    fn test_rejects_malformed_ops() {
        let p0 = Pid(0);
        let cases: Vec<(&str, Vec<Op<i32>>, HistoryError)> = vec![
            (
                "index not increasing",
                vec![Op::add_invoke(5, p0, 1), Op::add_ok(5, p0, 1)],
                HistoryError::IndexNotIncreasing {
                    index: 5,
                    previous: 5,
                },
            ),
            (
                "time decreasing",
                vec![
                    Op::add_invoke(0, p0, 1).at_millis(10),
                    Op::add_ok(1, p0, 1),
                    Op::read_invoke(2, Pid(1)).at_millis(9),
                ],
                HistoryError::TimeDecreasing { index: 2 },
            ),
            (
                "read ok without a value",
                vec![
                    Op::read_invoke(0, p0),
                    Op {
                        value: OpValue::None,
                        ..Op::read_ok(1, p0, [])
                    },
                ],
                HistoryError::InvalidValue { index: 1 },
            ),
            (
                "invoke while outstanding",
                vec![Op::add_invoke(0, p0, 1), Op::read_invoke(1, p0)],
                HistoryError::ConcurrentInvoke {
                    index: 1,
                    process: p0,
                    outstanding: 0,
                },
            ),
            (
                "invoke after info",
                vec![
                    Op::add_invoke(0, p0, 1),
                    Op::add_info(1, p0, 1),
                    Op::read_invoke(2, p0),
                ],
                HistoryError::InvokeAfterCrash {
                    index: 2,
                    process: p0,
                    crashed_at: 1,
                },
            ),
            (
                "completion without invocation",
                vec![Op::add_ok(0, p0, 1)],
                HistoryError::UnmatchedCompletion {
                    index: 0,
                    process: p0,
                },
            ),
            (
                "completion of a different function",
                vec![Op::add_invoke(0, p0, 1), Op::read_ok(1, p0, [1])],
                HistoryError::CompletionMismatch {
                    index: 1,
                    invocation: 0,
                },
            ),
            (
                "completion of a different element",
                vec![Op::add_invoke(0, p0, 1), Op::add_ok(1, p0, 2)],
                HistoryError::CompletionMismatch {
                    index: 1,
                    invocation: 0,
                },
            ),
        ];

        for (name, ops, expected) in cases {
            let (last, prefix) = ops.split_last().unwrap();
            let mut history = History::from_ops(prefix.to_vec()).unwrap();
            assert_eq!(history.push(last.clone()), Err(expected), "{name}");
            assert_eq!(history.len(), prefix.len(), "{name}: history changed");
        }
    }
}
