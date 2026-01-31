//! History representation for consistency checking.
//!
//! Operations in a history follow a request/response model:
//! - `Invoke` marks the start of an operation
//! - `Ok` marks successful completion
//! - `Fail` marks a definite failure
//! - `Info` marks an indeterminate result (crash, timeout, etc.)

use ahash::HashSet;
use std::hash::Hash;

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
    /// Timestamp in nanoseconds (optional, used for latency calculations).
    pub time: Option<u64>,
    /// Process/thread that performed this operation.
    pub process: u64,
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
#[derive(Debug, Clone)]
pub struct History<T> {
    ops: Vec<Op<T>>,
}

impl<T: Clone + Eq + Hash> History<T> {
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    pub fn from_ops(ops: Vec<Op<T>>) -> Self {
        Self { ops }
    }

    pub fn push(&mut self, op: Op<T>) {
        self.ops.push(op);
    }

    pub fn ops(&self) -> &[Op<T>] {
        &self.ops
    }

    /// Find the invocation operation for a given completion operation.
    /// Searches backward for a matching Invoke from the same process.
    pub fn invocation(&self, completion: &Op<T>) -> Option<&Op<T>> {
        self.ops[..completion.index]
            .iter()
            .rev()
            .find(|op| op.process == completion.process && op.op_type == OpType::Invoke)
    }
}

impl<T: Clone + Eq + Hash> Default for History<T> {
    fn default() -> Self {
        Self::new()
    }
}
