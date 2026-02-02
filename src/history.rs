//! History representation for consistency checking.
//!
//! Operations in a history follow a request/response model:
//! - `Invoke` marks the start of an operation
//! - `Ok` marks successful completion
//! - `Fail` marks a definite failure
//! - `Info` marks an indeterminate result (crash, timeout, etc.)

use std::collections::HashSet;
use std::time::Duration;

/// Process or thread identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct ProcessId(pub u64);

impl From<u64> for ProcessId {
    fn from(v: u64) -> Self {
        Self(v)
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
    pub time: Option<Duration>,
    /// Process/thread that performed this operation.
    pub process: ProcessId,
}

impl<T> Op<T> {
    /// Create an add invocation.
    pub fn add_invoke(index: usize, process: impl Into<ProcessId>, value: T) -> Self {
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
    pub fn add_ok(index: usize, process: impl Into<ProcessId>, value: T) -> Self {
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
    pub fn read_invoke(index: usize, process: impl Into<ProcessId>) -> Self {
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
        process: impl Into<ProcessId>,
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
    pub fn add_info(index: usize, process: impl Into<ProcessId>, value: T) -> Self {
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
    pub fn add_fail(index: usize, process: impl Into<ProcessId>, value: T) -> Self {
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
    pub fn read_info(index: usize, process: impl Into<ProcessId>) -> Self {
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
    pub fn read_fail(index: usize, process: impl Into<ProcessId>) -> Self {
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
    pub fn at(mut self, time: Duration) -> Self {
        self.time = Some(time);
        self
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

/// A history of operations.
#[derive(Debug, Clone, Default)]
pub struct History<T> {
    ops: Vec<Op<T>>,
}

impl<T> History<T> {
    #[must_use]
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    #[must_use]
    pub fn from_ops(ops: Vec<Op<T>>) -> Self {
        Self { ops }
    }

    pub fn push(&mut self, op: Op<T>) {
        self.ops.push(op);
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

    /// Find the invocation for a given completion by searching backward.
    #[must_use]
    pub fn invocation(&self, completion_pos: usize) -> Option<&Op<T>> {
        if completion_pos == 0 || completion_pos > self.ops.len() {
            return None;
        }
        let completion = &self.ops[completion_pos];
        self.ops[..completion_pos].iter().rev().find(|op| {
            op.process == completion.process && op.op_type == OpType::Invoke && op.f == completion.f
        })
    }
}
