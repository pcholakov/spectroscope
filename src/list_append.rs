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
//!
//! The workload and the checker follow Elle (Kingsbury and Alvaro, "Elle:
//! Inferring Isolation Anomalies from Experimental Observations", VLDB 2020,
//! <https://www.vldb.org/pvldb/vol14/p268-alvaro.pdf>; code at
//! <https://github.com/jepsen-io/elle>). Anomalies keep Elle's names so results
//! can be compared.
//!
//! [`ListAppendChecker`] currently reports the anomalies visible without a
//! dependency graph; see [`Anomaly`].

use crate::Validity;
use crate::history::{History, Op, OpType, Operation, Pid};

mod anomaly;
mod txns;

pub use anomaly::{Anomaly, Expected};

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

/// Identifies a transaction by the indices of its ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TxnRef {
    /// Index of the invocation.
    pub invoke_index: usize,
    /// Index of the completion; `None` while the transaction is pending.
    pub complete_index: Option<usize>,
}

/// Checks list-append histories for the anomalies in [`Anomaly`].
#[derive(Debug, Clone, Copy)]
pub struct ListAppendChecker;

/// The outcome of [`ListAppendChecker::check`].
#[derive(Debug, Clone)]
pub struct ListAppendResult<K, V> {
    /// `Unknown` if any value was appended to a key twice, since reads can then
    /// no longer be attributed to one writer. Otherwise `Invalid` if any other
    /// anomaly was found, and `Unknown` if none was: the absence of these
    /// anomalies alone does not establish any isolation level.
    pub valid: Validity,
    /// Every anomaly found, grouped by kind in the order of [`Anomaly`]'s
    /// variants. Within a kind, in history order, or key order for per-key kinds.
    pub anomalies: Vec<Anomaly<K, V>>,
}

impl ListAppendChecker {
    /// Check `history` for anomalies.
    ///
    /// Cost: O(n log n) in the total number of micro-operations and read values.
    #[must_use]
    pub fn check<K, V>(&self, history: &History<Txn<K, V>>) -> ListAppendResult<K, V>
    where
        K: Clone + Ord,
        V: Clone + Ord,
    {
        let txns = txns::Txns::new(history);
        let anomalies = anomaly::find(&txns);
        let unusable = anomalies
            .iter()
            .any(|anomaly| matches!(anomaly, Anomaly::DuplicateAppends { .. }));
        let valid = if !unusable && !anomalies.is_empty() {
            Validity::Invalid
        } else {
            Validity::Unknown
        };
        ListAppendResult { valid, anomalies }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::{History, HistoryError};

    type TestOp = MicroOp<&'static str, u32>;

    fn append(key: &'static str, value: u32) -> TestOp {
        MicroOp::append(key, value)
    }

    fn read(key: &'static str) -> TestOp {
        MicroOp::read(key)
    }

    fn read_result(key: &'static str, list: &[u32]) -> TestOp {
        MicroOp::read_result(key, list.iter().copied())
    }

    /// Builds a history of transactions, assigning op indices in push order.
    #[derive(Default)]
    struct Builder {
        ops: Vec<Op<Txn<&'static str, u32>>>,
    }

    impl Builder {
        /// An invocation of `micro_ops` with read results removed.
        fn invoke(&mut self, process: u64, micro_ops: &[TestOp]) -> usize {
            let index = self.ops.len();
            let invoked = micro_ops.iter().map(|micro_op| match micro_op {
                MicroOp::Read { key, .. } => read(key),
                MicroOp::Append { .. } => micro_op.clone(),
            });
            self.ops.push(Op::txn_invoke(index, process, invoked));
            index
        }

        fn committed(&mut self, process: u64, micro_ops: &[TestOp]) -> TxnRef {
            let invoke_index = self.invoke(process, micro_ops);
            let complete_index = self.ops.len();
            let completion = Op::txn_ok(complete_index, process, micro_ops.iter().cloned());
            self.ops.push(completion);
            TxnRef {
                invoke_index,
                complete_index: Some(complete_index),
            }
        }

        /// A transaction that completed `Fail` or `Info`, so its reads carry no lists.
        fn uncommitted(&mut self, process: u64, outcome: OpType, appends: &[TestOp]) -> TxnRef {
            let invoke_index = self.invoke(process, appends);
            let complete_index = self.ops.len();
            let value = self.ops[invoke_index].value.clone();
            self.ops
                .push(Op::new(complete_index, process, outcome, value));
            TxnRef {
                invoke_index,
                complete_index: Some(complete_index),
            }
        }

        fn check(self) -> ListAppendResult<&'static str, u32> {
            ListAppendChecker.check(&History::from_ops(self.ops).unwrap())
        }
    }

    fn assert_invalid(
        result: &ListAppendResult<&'static str, u32>,
        expected: &[Anomaly<&'static str, u32>],
    ) {
        assert_eq!(result.anomalies, expected);
        assert_eq!(result.valid, Validity::Invalid);
    }

    /// Histories free of these anomalies, including the cases that resemble them:
    /// reading one's own intermediate append, reading one's own append after
    /// others' values, reading an indeterminate append, a single
    /// read-then-append, an empty read, and compatible prefixes.
    #[test]
    fn test_clean_history_has_no_anomalies() {
        let mut b = Builder::default();
        b.committed(0, &[append("x", 1), read_result("x", &[1]), append("x", 2)]);
        b.committed(0, &[append("w", 7)]);
        b.committed(0, &[append("w", 8), read_result("w", &[7, 8])]);
        b.uncommitted(1, OpType::Info, &[append("y", 5)]);
        b.committed(
            2,
            &[
                read_result("y", &[5]),
                read_result("x", &[1, 2]),
                append("x", 3),
            ],
        );
        b.committed(3, &[read_result("z", &[])]);
        b.committed(3, &[read_result("x", &[1, 2, 3])]);
        let result = b.check();
        assert_eq!(result.anomalies, []);
        assert_eq!(result.valid, Validity::Unknown);
    }

    #[test]
    fn test_duplicate_appends_make_history_uncheckable() {
        let mut b = Builder::default();
        let first = b.committed(0, &[append("x", 1)]);
        let second = b.committed(1, &[append("x", 1)]);
        let result = b.check();
        let expected = Anomaly::DuplicateAppends {
            key: "x",
            value: 1,
            first,
            second,
        };
        assert_eq!(result.anomalies, [expected]);
        assert_eq!(result.valid, Validity::Unknown);
    }

    #[test]
    fn test_duplicate_elements() {
        let mut b = Builder::default();
        b.committed(0, &[append("x", 1)]);
        let reader = b.committed(1, &[read_result("x", &[1, 1])]);
        let expected = Anomaly::DuplicateElements {
            reader,
            key: "x",
            duplicates: vec![(1, 2)],
        };
        assert_invalid(&b.check(), &[expected]);
    }

    #[test]
    fn test_incompatible_order() {
        let mut b = Builder::default();
        b.committed(0, &[append("x", 1)]);
        b.committed(1, &[append("x", 2)]);
        let first = b.committed(2, &[read_result("x", &[1, 2])]);
        let second = b.committed(3, &[read_result("x", &[2, 1])]);
        let expected = Anomaly::IncompatibleOrder {
            key: "x",
            shorter: (vec![1, 2], first),
            longer: (vec![2, 1], second),
        };
        assert_invalid(&b.check(), &[expected]);
    }

    #[test]
    fn test_internal() {
        // A second read disagrees with the first.
        let mut b = Builder::default();
        b.committed(0, &[append("x", 1)]);
        let txn = b.committed(1, &[read_result("x", &[1]), read_result("x", &[])]);
        let expected = Anomaly::Internal {
            txn,
            key: "x",
            read: vec![],
            expected: Expected::List(vec![1]),
        };
        assert_invalid(&b.check(), &[expected]);

        // A read misses the transaction's own earlier append.
        let mut b = Builder::default();
        b.committed(0, &[append("x", 1)]);
        let txn = b.committed(1, &[append("x", 2), read_result("x", &[1])]);
        let expected = Anomaly::Internal {
            txn,
            key: "x",
            read: vec![1],
            expected: Expected::Suffix(vec![2]),
        };
        assert_invalid(&b.check(), &[expected]);
    }

    #[test]
    fn test_future_read() {
        let mut b = Builder::default();
        let txn = b.committed(0, &[read_result("x", &[1]), append("x", 1)]);
        let expected = Anomaly::FutureRead {
            txn,
            key: "x",
            value: 1,
        };
        assert_invalid(&b.check(), &[expected]);
    }

    #[test]
    fn test_g1a_aborted_read() {
        let mut b = Builder::default();
        let writer = b.uncommitted(0, OpType::Fail, &[append("x", 1)]);
        let reader = b.committed(1, &[read_result("x", &[1])]);
        let expected = Anomaly::G1a {
            reader,
            writer,
            key: "x",
            value: 1,
        };
        assert_invalid(&b.check(), &[expected]);
    }

    #[test]
    fn test_g1b_intermediate_read() {
        let mut b = Builder::default();
        let writer = b.committed(0, &[append("x", 1), append("x", 2)]);
        let reader = b.committed(1, &[read_result("x", &[1])]);
        let expected = Anomaly::G1b {
            reader,
            writer,
            key: "x",
            value: 1,
        };
        assert_invalid(&b.check(), &[expected]);
    }

    /// The committed append follows the failed one through an indeterminate
    /// append, which neither ends nor starts the failed run.
    #[test]
    fn test_dirty_update() {
        let mut b = Builder::default();
        let aborted = b.uncommitted(0, OpType::Fail, &[append("x", 1)]);
        b.uncommitted(1, OpType::Info, &[append("x", 2)]);
        let committed = b.committed(2, &[append("x", 3)]);
        let reader = b.committed(3, &[read_result("x", &[1, 2, 3])]);
        let expected = [
            Anomaly::G1a {
                reader,
                writer: aborted,
                key: "x",
                value: 1,
            },
            Anomaly::DirtyUpdate {
                key: "x",
                values: vec![1, 2, 3],
                aborted,
                committed,
            },
        ];
        assert_invalid(&b.check(), &expected);
    }

    #[test]
    fn test_lost_update() {
        let mut b = Builder::default();
        b.committed(0, &[append("x", 1)]);
        let first = b.committed(1, &[read_result("x", &[1]), append("x", 2)]);
        let second = b.committed(2, &[read_result("x", &[1]), append("x", 3)]);
        let expected = Anomaly::LostUpdate {
            key: "x",
            read: vec![1],
            txns: vec![first, second],
        };
        assert_invalid(&b.check(), &[expected]);
    }

    #[test]
    fn test_garbage_read() {
        let mut b = Builder::default();
        let reader = b.committed(0, &[read_result("x", &[9])]);
        let expected = Anomaly::GarbageRead {
            reader,
            key: "x",
            value: 9,
        };
        assert_invalid(&b.check(), &[expected]);
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
