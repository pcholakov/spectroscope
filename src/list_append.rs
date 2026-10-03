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
//! [`ListAppendChecker`] reports two kinds of evidence: anomalies visible
//! without a dependency graph (see [`Anomaly`]), and cycles in the dependency
//! graph that rule out the chosen [`ConsistencyModel`] (see [`Cycle`]).
//!
//! # Reading the results
//!
//! - One cause can appear as both kinds, as in Elle: a lost update is also a
//!   `G2-item` cycle, and a dirty update comes with the `G1a` read that showed
//!   the failed value.
//! - A key's version order is its longest committed read. When reads of the key
//!   disagree (`incompatible-order`), cycle steps on that key follow the longest
//!   read and may contradict the others.

use crate::Validity;
use crate::history::{History, Op, OpType, Operation, Pid};

mod anomaly;
mod dependency;
mod txns;

pub use anomaly::{Anomaly, Expected};
pub use dependency::{
    ConsistencyModel, Cycle, CycleAnomaly, DataCycle, Dependency, OrderEdge, Step,
};

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

/// Checks list-append histories against a [`ConsistencyModel`].
#[derive(Debug, Clone, Copy)]
pub struct ListAppendChecker {
    /// The model histories must satisfy.
    pub model: ConsistencyModel,
}

/// The outcome of [`ListAppendChecker::check`].
#[derive(Debug, Clone)]
pub struct ListAppendResult<K, V> {
    /// `Unknown` if any value was appended to a key twice, since reads can then
    /// no longer be attributed to one writer. Otherwise `Invalid` if there is
    /// any anomaly or cycle, and `Valid` if there is neither.
    pub valid: Validity,
    /// Every anomaly found, grouped by kind in the order of [`Anomaly`]'s
    /// variants. Within a kind, in history order, or key order for per-key kinds.
    pub anomalies: Vec<Anomaly<K, V>>,
    /// One shortest cycle per strongly connected component of the dependency
    /// graph under the checker's model, through the component's
    /// earliest-invoked transaction, in that transaction's order.
    pub cycles: Vec<Cycle<K, V>>,
}

impl ListAppendChecker {
    /// A checker for `model`.
    #[must_use]
    pub fn new(model: ConsistencyModel) -> Self {
        Self { model }
    }

    /// Check `history`.
    ///
    /// Cost: O(n log n) in the number of micro-operations and read values, plus
    /// one breadth-first search per strongly connected component.
    #[must_use]
    pub fn check<K, V>(&self, history: &History<Txn<K, V>>) -> ListAppendResult<K, V>
    where
        K: Clone + Ord,
        V: Clone + Ord,
    {
        let txns = txns::Txns::new(history);
        let anomalies = anomaly::find(&txns);
        let cycles = dependency::cycles(&txns, self.model);
        let unusable = anomalies
            .iter()
            .any(|anomaly| matches!(anomaly, Anomaly::DuplicateAppends { .. }));
        let valid = if unusable {
            Validity::Unknown
        } else if anomalies.is_empty() && cycles.is_empty() {
            Validity::Valid
        } else {
            Validity::Invalid
        };
        ListAppendResult {
            valid,
            anomalies,
            cycles,
        }
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

        /// A transaction that never completes.
        fn pending(&mut self, process: u64, appends: &[TestOp]) -> TxnRef {
            TxnRef {
                invoke_index: self.invoke(process, appends),
                complete_index: None,
            }
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
            self.check_against(ConsistencyModel::StrictSerializable)
        }

        fn check_against(self, model: ConsistencyModel) -> ListAppendResult<&'static str, u32> {
            ListAppendChecker::new(model).check(&History::from_ops(self.ops).unwrap())
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
        // An append that no read observed and that may not have happened, and a
        // failed append, read as absent afterwards.
        b.uncommitted(4, OpType::Info, &[append("v", 1)]);
        b.uncommitted(5, OpType::Fail, &[append("u", 1)]);
        b.committed(6, &[read_result("v", &[]), read_result("u", &[])]);
        let result = b.check();
        assert_eq!(result.anomalies, []);
        assert_eq!(result.cycles, []);
        assert_eq!(result.valid, Validity::Valid);
    }

    fn ww(
        from: TxnRef,
        to: TxnRef,
        key: &'static str,
        previous: u32,
        appended: u32,
    ) -> Step<&'static str, u32> {
        let dependency = Dependency::WriteWrite {
            key,
            previous,
            appended,
        };
        Step {
            from,
            to,
            dependency,
        }
    }

    fn rw(
        from: TxnRef,
        to: TxnRef,
        key: &'static str,
        read: Option<u32>,
        appended: u32,
    ) -> Step<&'static str, u32> {
        let dependency = Dependency::ReadWrite {
            key,
            read,
            appended,
        };
        Step {
            from,
            to,
            dependency,
        }
    }

    fn wr(from: TxnRef, to: TxnRef, key: &'static str, value: u32) -> Step<&'static str, u32> {
        let dependency = Dependency::WriteRead { key, value };
        Step {
            from,
            to,
            dependency,
        }
    }

    /// Asserts the result's only evidence is one cycle named `name` with `steps`.
    fn assert_single_cycle(
        result: &ListAppendResult<&'static str, u32>,
        name: &str,
        steps: &[Step<&'static str, u32>],
    ) {
        assert_eq!(result.anomalies, []);
        assert_eq!(result.cycles.len(), 1, "{:?}", result.cycles);
        assert_eq!(result.cycles[0].anomaly.name(), name);
        assert_eq!(result.cycles[0].steps, steps);
        assert_eq!(result.valid, Validity::Invalid);
    }

    /// Reads show T0 then T1 appended to x, but T1 then T0 appended to y.
    #[test]
    fn test_write_write_cycle() {
        let mut b = Builder::default();
        let t0 = b.committed(0, &[append("x", 1), append("y", 2)]);
        let t1 = b.committed(1, &[append("x", 3), append("y", 4)]);
        b.committed(2, &[read_result("x", &[1, 3])]);
        b.committed(3, &[read_result("y", &[4, 2])]);
        let steps = vec![ww(t0, t1, "x", 1, 3), ww(t1, t0, "y", 4, 2)];
        let result = b.check_against(ConsistencyModel::Serializable);
        assert_single_cycle(&result, "G0", &steps);
    }

    /// Write skew: each transaction read the empty list the other appended to.
    #[test]
    fn test_read_write_cycle() {
        let mut b = Builder::default();
        let t0 = b.committed(0, &[read_result("x", &[]), append("y", 1)]);
        let t1 = b.committed(1, &[read_result("y", &[]), append("x", 2)]);
        let steps = vec![rw(t0, t1, "x", None, 2), rw(t1, t0, "y", None, 1)];
        let result = b.check_against(ConsistencyModel::Serializable);
        assert_single_cycle(&result, "G2-item", &steps);
    }

    /// T1 appended after T0, and T0 read T1's append.
    #[test]
    fn test_write_read_cycle() {
        let mut b = Builder::default();
        let t0 = b.committed(0, &[append("x", 1), read_result("y", &[3])]);
        let t1 = b.committed(1, &[append("x", 2), append("y", 3)]);
        b.committed(2, &[read_result("x", &[1, 2])]);
        let steps = vec![ww(t0, t1, "x", 1, 2), wr(t1, t0, "y", 3)];
        let result = b.check_against(ConsistencyModel::Serializable);
        assert_single_cycle(&result, "G1c", &steps);
    }

    /// The anti-dependencies are adjacent only across the cycle's last and first
    /// steps, which still makes it G2.
    #[test]
    fn test_read_writes_adjacent_across_cycle_start() {
        let mut b = Builder::default();
        let t0 = b.committed(0, &[read_result("x", &[]), append("z", 2)]);
        let t1 = b.committed(1, &[append("x", 1), append("y", 5)]);
        let t2 = b.committed(2, &[read_result("y", &[5]), read_result("z", &[])]);
        let steps = vec![
            rw(t0, t1, "x", None, 1),
            wr(t1, t2, "y", 5),
            rw(t2, t0, "z", None, 2),
        ];
        let result = b.check_against(ConsistencyModel::Serializable);
        assert_single_cycle(&result, "G2-item", &steps);
    }

    /// Two anti-dependencies separated by write-read dependencies.
    #[test]
    fn test_nonadjacent_read_write_cycle() {
        let mut b = Builder::default();
        let t0 = b.committed(0, &[read_result("x", &[]), read_result("z", &[3])]);
        let t1 = b.committed(1, &[append("x", 1), append("y", 5)]);
        let t2 = b.committed(2, &[read_result("y", &[5]), read_result("w", &[])]);
        let t3 = b.committed(3, &[append("w", 4), append("z", 3)]);
        let steps = vec![
            rw(t0, t1, "x", None, 1),
            wr(t1, t2, "y", 5),
            rw(t2, t3, "w", None, 4),
            wr(t3, t0, "z", 3),
        ];
        let result = b.check_against(ConsistencyModel::Serializable);
        assert_single_cycle(&result, "G-nonadjacent-item", &steps);
    }

    /// A read invoked after an append completed misses it: serializable with the
    /// read first, but not once real-time or process order counts.
    #[test]
    fn test_stale_read_cycles_only_under_real_time_or_process_order() {
        let stale_read = |reader_process: u64| {
            let mut b = Builder::default();
            let writer = b.committed(0, &[append("x", 1)]);
            let reader = b.committed(reader_process, &[read_result("x", &[])]);
            (b, writer, reader)
        };

        let (b, _, _) = stale_read(1);
        assert_eq!(
            b.check_against(ConsistencyModel::Serializable).valid,
            Validity::Valid
        );
        let (b, _, _) = stale_read(1);
        let result = b.check_against(ConsistencyModel::StrongSessionSerializable);
        assert_eq!(result.valid, Validity::Valid);

        let (b, writer, reader) = stale_read(1);
        let real_time = Step {
            from: writer,
            to: reader,
            dependency: Dependency::RealTime {
                completed_index: 1,
                invoked_index: 2,
            },
        };
        let steps = vec![real_time, rw(reader, writer, "x", None, 1)];
        let result = b.check_against(ConsistencyModel::StrictSerializable);
        assert_single_cycle(&result, "G-single-item-realtime", &steps);

        let (b, writer, reader) = stale_read(0);
        let process = Step {
            from: writer,
            to: reader,
            dependency: Dependency::Process,
        };
        let steps = vec![process, rw(reader, writer, "x", None, 1)];
        let result = b.check_against(ConsistencyModel::StrongSessionSerializable);
        assert_single_cycle(&result, "G-single-item-process", &steps);
    }

    /// A real-time step through several time nodes counts as one step, so the
    /// two-step stale read wins over a three-step chain of reads.
    #[test]
    fn test_shortest_cycle_counts_real_time_as_one_step() {
        let mut b = Builder::default();
        let writer = b.committed(0, &[append("x", 1), append("y", 2)]);
        b.committed(1, &[read_result("y", &[2]), append("z", 3)]);
        let reader = b.committed(2, &[read_result("z", &[3]), read_result("x", &[])]);
        let real_time = Step {
            from: writer,
            to: reader,
            dependency: Dependency::RealTime {
                completed_index: 1,
                invoked_index: 4,
            },
        };
        let steps = vec![real_time, rw(reader, writer, "x", None, 1)];
        assert_single_cycle(&b.check(), "G-single-item-realtime", &steps);
    }

    /// A pending transaction may have committed, but then all of it did: a read
    /// cannot see one of its appends and miss the other.
    #[test]
    fn test_pending_transaction_is_atomic() {
        let mut b = Builder::default();
        let writer = b.pending(0, &[append("x", 1), append("y", 2)]);
        let reader = b.committed(1, &[read_result("x", &[1]), read_result("y", &[])]);
        let steps = vec![wr(writer, reader, "x", 1), rw(reader, writer, "y", None, 2)];
        let result = b.check_against(ConsistencyModel::Serializable);
        assert_single_cycle(&result, "G-single-item", &steps);
    }

    /// An append no read observed still follows every observed value, so a read
    /// that misses it after it completed is stale.
    #[test]
    fn test_stale_read_of_unobserved_append() {
        let mut b = Builder::default();
        b.committed(0, &[append("x", 1)]);
        b.committed(1, &[read_result("x", &[1])]);
        let writer = b.committed(0, &[append("x", 2)]);
        let reader = b.committed(2, &[read_result("x", &[1])]);
        let real_time = Step {
            from: writer,
            to: reader,
            dependency: Dependency::RealTime {
                completed_index: 5,
                invoked_index: 6,
            },
        };
        let steps = vec![real_time, rw(reader, writer, "x", Some(1), 2)];
        assert_single_cycle(&b.check(), "G-single-item-realtime", &steps);
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
        let result = b.check();
        assert_invalid(&result, &[expected]);
        // Neither append was observed, and each transaction read the state the
        // other appended after: the same root cause as a G2-item cycle.
        let steps = vec![
            rw(first, second, "x", Some(1), 3),
            rw(second, first, "x", Some(1), 2),
        ];
        assert_eq!(result.cycles.len(), 1);
        assert_eq!(result.cycles[0].anomaly.name(), "G2-item");
        assert_eq!(result.cycles[0].steps, steps);
    }

    /// Reads show a transaction's appends out of its own order, or a later one
    /// without the earlier one.
    #[test]
    fn test_append_order() {
        let mut b = Builder::default();
        let txn = b.committed(0, &[append("x", 1), append("x", 2)]);
        let reader = b.committed(1, &[read_result("x", &[2, 1])]);
        let expected = [
            // The out-of-order read also ends on the intermediate value.
            Anomaly::G1b {
                reader,
                writer: txn,
                key: "x",
                value: 1,
            },
            Anomaly::AppendOrder {
                txn,
                key: "x",
                appended: vec![1, 2],
                observed: vec![2, 1],
            },
        ];
        assert_invalid(&b.check(), &expected);

        let mut b = Builder::default();
        let txn = b.committed(0, &[append("x", 1), append("x", 2)]);
        b.committed(1, &[read_result("x", &[2])]);
        let expected = Anomaly::AppendOrder {
            txn,
            key: "x",
            appended: vec![1, 2],
            observed: vec![2],
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
