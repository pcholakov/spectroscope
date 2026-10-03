//! List-append transactions.
//!
//! Each key holds a list, and a transaction is a sequence of micro-operations
//! that append a value to a key's list or read a key's whole list. Because every
//! appended value is unique per key and a read returns the whole list, each read
//! reveals the order in which the writes it observed were applied. Checkers can
//! then infer dependencies between transactions directly instead of searching
//! for a version order.
//!
//! The system under test need not support appends natively: a client can
//! implement an append as a read of the list followed by a write of the extended
//! list, inside the same transaction.

use crate::history::{Op, OpType, Operation, Pid};

/// One step of a list-append transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MicroOp<K, V> {
    /// Append `value` to the list at `key`. Each value must be appended to a key
    /// at most once in the whole history, so a read can attribute it to one
    /// transaction.
    Append {
        /// The list appended to.
        key: K,
        /// The appended value.
        value: V,
    },
    /// Read the whole list at `key`.
    Read {
        /// The list read.
        key: K,
        /// The list as read, oldest value first: `None` in an invocation and in a
        /// transaction that did not complete `Ok`, and `Some` otherwise.
        list: Option<Vec<V>>,
    },
}

impl<K, V> MicroOp<K, V> {
    /// Append `value` to the list at `key`.
    pub fn append(key: K, value: V) -> Self {
        Self::Append { key, value }
    }

    /// Read the list at `key`, as written in an invocation.
    pub fn read(key: K) -> Self {
        Self::Read { key, list: None }
    }

    /// A read of `key` that returned `list`, as written in an `Ok` completion.
    pub fn read_result(key: K, list: impl IntoIterator<Item = V>) -> Self {
        Self::Read {
            key,
            list: Some(list.into_iter().collect()),
        }
    }
}

/// A transaction: micro-operations in the order the transaction executed them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Txn<K, V>(pub Vec<MicroOp<K, V>>);

impl<K: PartialEq, V: PartialEq> Operation for Txn<K, V> {
    fn is_valid_invocation(&self) -> bool {
        self.0
            .iter()
            .all(|micro_op| !matches!(micro_op, MicroOp::Read { list: Some(_), .. }))
    }

    /// A completion repeats its invocation's micro-operations in order, with
    /// every read's list filled in exactly when the transaction completed `Ok`.
    fn is_valid_completion(&self, invocation: &Self, outcome: OpType) -> bool {
        let reads_returned = outcome == OpType::Ok;
        self.0.len() == invocation.0.len()
            && self.0.iter().zip(&invocation.0).all(|pair| match pair {
                (
                    MicroOp::Append { key, value },
                    MicroOp::Append {
                        key: key_invoked,
                        value: value_invoked,
                    },
                ) => key == key_invoked && value == value_invoked,
                (
                    MicroOp::Read { key, list },
                    MicroOp::Read {
                        key: key_invoked, ..
                    },
                ) => key == key_invoked && list.is_some() == reads_returned,
                _ => false,
            })
    }
}

impl<K, V> Op<Txn<K, V>> {
    /// Create a transaction invocation.
    pub fn txn_invoke(
        index: usize,
        process: impl Into<Pid>,
        micro_ops: impl IntoIterator<Item = MicroOp<K, V>>,
    ) -> Self {
        Self::new(index, process, OpType::Invoke, txn(micro_ops))
    }

    /// Create a committed transaction's completion. Every read carries its list.
    pub fn txn_ok(
        index: usize,
        process: impl Into<Pid>,
        micro_ops: impl IntoIterator<Item = MicroOp<K, V>>,
    ) -> Self {
        Self::new(index, process, OpType::Ok, txn(micro_ops))
    }

    /// Create the completion of a transaction that definitely did not commit.
    pub fn txn_fail(
        index: usize,
        process: impl Into<Pid>,
        micro_ops: impl IntoIterator<Item = MicroOp<K, V>>,
    ) -> Self {
        Self::new(index, process, OpType::Fail, txn(micro_ops))
    }

    /// Create the completion of a transaction that may or may not have committed.
    pub fn txn_info(
        index: usize,
        process: impl Into<Pid>,
        micro_ops: impl IntoIterator<Item = MicroOp<K, V>>,
    ) -> Self {
        Self::new(index, process, OpType::Info, txn(micro_ops))
    }
}

fn txn<K, V>(micro_ops: impl IntoIterator<Item = MicroOp<K, V>>) -> Txn<K, V> {
    Txn(micro_ops.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::{History, HistoryError};

    fn append(key: &'static str, value: u32) -> MicroOp<&'static str, u32> {
        MicroOp::append(key, value)
    }

    fn read(key: &'static str) -> MicroOp<&'static str, u32> {
        MicroOp::read(key)
    }

    fn read_result(key: &'static str, list: &[u32]) -> MicroOp<&'static str, u32> {
        MicroOp::read_result(key, list.iter().copied())
    }

    #[test]
    fn test_accepts_well_formed_transactions() {
        let history = History::from_ops([
            Op::txn_invoke(0, 0u64, [append("x", 1), read("y")]),
            Op::txn_invoke(1, 1u64, [read("x")]),
            Op::txn_ok(2, 0u64, [append("x", 1), read_result("y", &[])]),
            Op::txn_info(3, 1u64, [read("x")]),
            Op::txn_invoke(4, 2u64, [append("y", 2)]),
            Op::txn_fail(5, 2u64, [append("y", 2)]),
        ]);
        assert!(history.is_ok(), "{history:?}");
    }

    /// The last op of each case is rejected; the ops before it are well-formed.
    #[test]
    fn test_rejects_malformed_transactions() {
        type Ops = Vec<Op<Txn<&'static str, u32>>>;
        let invoke = || Op::txn_invoke(0, 0u64, [append("x", 1), read("y")]);
        let mismatch = HistoryError::CompletionMismatch {
            index: 1,
            invocation: 0,
        };
        let cases: Vec<(&str, Ops, HistoryError)> = vec![
            (
                "invocation with a read result",
                vec![Op::txn_invoke(0, 0u64, [read_result("y", &[1])])],
                HistoryError::InvalidInvocation { index: 0 },
            ),
            (
                "ok without a read result",
                vec![invoke(), Op::txn_ok(1, 0u64, [append("x", 1), read("y")])],
                mismatch.clone(),
            ),
            (
                "info with a read result",
                vec![
                    invoke(),
                    Op::txn_info(1, 0u64, [append("x", 1), read_result("y", &[])]),
                ],
                mismatch.clone(),
            ),
            (
                "different appended value",
                vec![
                    invoke(),
                    Op::txn_ok(1, 0u64, [append("x", 2), read_result("y", &[])]),
                ],
                mismatch.clone(),
            ),
            (
                "different read key",
                vec![
                    invoke(),
                    Op::txn_ok(1, 0u64, [append("x", 1), read_result("z", &[])]),
                ],
                mismatch.clone(),
            ),
            (
                "micro-operations reordered",
                vec![
                    invoke(),
                    Op::txn_ok(1, 0u64, [read_result("y", &[]), append("x", 1)]),
                ],
                mismatch.clone(),
            ),
            (
                "micro-operation dropped",
                vec![invoke(), Op::txn_ok(1, 0u64, [append("x", 1)])],
                mismatch,
            ),
        ];

        for (name, ops, expected) in cases {
            let (last, prefix) = ops.split_last().unwrap();
            let mut history = History::from_ops(prefix.to_vec()).unwrap();
            assert_eq!(history.push(last.clone()), Err(expected), "{name}");
        }
    }
}
