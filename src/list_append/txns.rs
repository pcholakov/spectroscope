//! An indexed view of a list-append history: one record per transaction, the
//! transaction that appended each value, and the distinct lists committed reads
//! returned for each key.

use std::collections::BTreeMap;

use super::{MicroOp, Txn, TxnRef};
use crate::history::{History, OpType};

/// Whether a transaction's effects took place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Outcome {
    /// Completed `Ok`: its effects took place and its reads returned lists.
    Committed,
    /// Completed `Fail`: none of its effects took place.
    Aborted,
    /// Completed `Info`, or never completed: its effects may or may not have
    /// taken place.
    Indeterminate,
}

#[derive(Debug)]
pub(super) struct TxnRecord<'a, K, V> {
    pub(super) reference: TxnRef,
    pub(super) outcome: Outcome,
    /// The completion's micro-operations when committed, so reads carry lists;
    /// otherwise the invocation's.
    pub(super) micro_ops: &'a [MicroOp<K, V>],
}

/// Who appended one value to one key.
#[derive(Debug, Clone, Copy)]
pub(super) struct Append {
    /// Position of the appending transaction in [`Txns::records`].
    pub(super) txn: usize,
    /// The same transaction later appended another value to the same key.
    pub(super) intermediate: bool,
}

pub(super) struct Txns<'a, K, V> {
    /// Every transaction, in invocation order.
    pub(super) records: Vec<TxnRecord<'a, K, V>>,
    /// The first transaction to append each value to each key.
    pub(super) appends: BTreeMap<(&'a K, &'a V), Append>,
    /// Appends of a value that an earlier append already wrote to the same key,
    /// as (key, value, first appender, repeating appender) positions.
    pub(super) repeated_appends: Vec<(&'a K, &'a V, usize, usize)>,
    /// For each key, the distinct non-empty lists committed reads returned,
    /// shortest first (ties in list order), each with its first reader.
    pub(super) observed_lists: BTreeMap<&'a K, Vec<(&'a [V], usize)>>,
}

impl<'a, K: Ord, V: Ord> Txns<'a, K, V> {
    pub(super) fn new(history: &'a History<Txn<K, V>>) -> Self {
        let records = records(history);
        let mut appends: BTreeMap<(&K, &V), Append> = BTreeMap::new();
        let mut repeated_appends = Vec::new();
        let mut observed: BTreeMap<&K, BTreeMap<&[V], usize>> = BTreeMap::new();

        for (position, record) in records.iter().enumerate() {
            let mut last_append: BTreeMap<&K, &V> = BTreeMap::new();
            for micro_op in record.micro_ops {
                match micro_op {
                    MicroOp::Append { key, value } => {
                        if let Some(previous) = last_append.insert(key, value)
                            && let Some(append) = appends.get_mut(&(key, previous))
                            && append.txn == position
                        {
                            append.intermediate = true;
                        }
                        if let Some(first) = appends.get(&(key, value)) {
                            repeated_appends.push((key, value, first.txn, position));
                        } else {
                            let append = Append {
                                txn: position,
                                intermediate: false,
                            };
                            appends.insert((key, value), append);
                        }
                    }
                    MicroOp::Read {
                        key,
                        list: Some(list),
                    } if !list.is_empty() => {
                        observed
                            .entry(key)
                            .or_default()
                            .entry(list.as_slice())
                            .or_insert(position);
                    }
                    MicroOp::Read { .. } => {}
                }
            }
        }

        let observed_lists = observed
            .into_iter()
            .map(|(key, lists)| {
                let mut lists: Vec<(&[V], usize)> = lists.into_iter().collect();
                lists.sort_by(|(a, _), (b, _)| (a.len(), a).cmp(&(b.len(), b)));
                (key, lists)
            })
            .collect();

        Self {
            records,
            appends,
            repeated_appends,
            observed_lists,
        }
    }

    /// The transaction that first appended `value` to `key`, if any did.
    pub(super) fn appender(&self, key: &K, value: &V) -> Option<(Append, &TxnRecord<'a, K, V>)> {
        let append = *self.appends.get(&(key, value))?;
        Some((append, &self.records[append.txn]))
    }
}

/// One record per invocation, in invocation order.
fn records<K, V>(history: &History<Txn<K, V>>) -> Vec<TxnRecord<'_, K, V>> {
    history
        .ops()
        .iter()
        .enumerate()
        .filter(|(_, op)| op.op_type == OpType::Invoke)
        .map(|(position, invocation)| {
            let completion = history.completion(position);
            let outcome = match completion.map(|c| c.op_type) {
                Some(OpType::Ok) => Outcome::Committed,
                Some(OpType::Fail) => Outcome::Aborted,
                Some(OpType::Info | OpType::Invoke) | None => Outcome::Indeterminate,
            };
            let micro_ops = match completion {
                Some(c) if outcome == Outcome::Committed => &c.value.0,
                _ => &invocation.value.0,
            };
            TxnRecord {
                reference: TxnRef {
                    invoke_index: invocation.index,
                    complete_index: completion.map(|c| c.index),
                },
                outcome,
                micro_ops,
            }
        })
        .collect()
}
