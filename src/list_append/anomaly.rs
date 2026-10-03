//! Anomalies visible without a dependency graph: each one is evident from a
//! single transaction, a single key's reads, or the outcomes of the
//! transactions that appended a key's values.
//!
//! Names and definitions follow Elle's list-append checker
//! (<https://github.com/jepsen-io/elle>, `elle.list-append`), so results can be
//! compared with Elle's. `garbage-read` and `append-order` are this crate's
//! additions.

use std::collections::{BTreeMap, BTreeSet};

use super::TxnRef;
use super::txns::{Outcome, TxnRecord, Txns};
use crate::list_append::MicroOp;

/// An anomaly found without a dependency graph. Each variant documents its
/// Elle name, which [`Anomaly::name`] returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Anomaly<K, V> {
    /// `duplicate-appends`: two appends wrote the same value to the same key.
    /// The workload requires unique values, so reads can no longer be
    /// attributed to one writer; the history cannot be checked.
    DuplicateAppends {
        /// The key appended to twice.
        key: K,
        /// The repeated value.
        value: V,
        /// The first append's transaction.
        first: TxnRef,
        /// The repeating append's transaction; may equal `first`.
        second: TxnRef,
    },
    /// `duplicate-elements`: a committed read returned some value more than once.
    DuplicateElements {
        /// The reading transaction.
        reader: TxnRef,
        /// The key read.
        key: K,
        /// Each repeated value with the number of times the list contains it.
        duplicates: Vec<(V, usize)>,
    },
    /// `incompatible-order`: two committed reads of a key returned lists that
    /// cannot both be prefixes of one order of appends.
    IncompatibleOrder {
        /// The key read.
        key: K,
        /// The shorter list (or the earlier one of equal length), with a reader.
        shorter: (Vec<V>, TxnRef),
        /// The other list, which does not start with `shorter`, with a reader.
        longer: (Vec<V>, TxnRef),
    },
    /// `internal`: a committed transaction read a list inconsistent with its own
    /// earlier read of, or appends to, the same key.
    Internal {
        /// The transaction.
        txn: TxnRef,
        /// The key read.
        key: K,
        /// The list read.
        read: Vec<V>,
        /// What the transaction's own earlier micro-operations imply.
        expected: Expected<V>,
    },
    /// `future-read`: a committed transaction read a value it appends later.
    FutureRead {
        /// The transaction.
        txn: TxnRef,
        /// The key read and later appended to.
        key: K,
        /// The value read before it was appended.
        value: V,
    },
    /// `G1a`, aborted read: a committed read returned a value appended by a
    /// transaction that failed.
    G1a {
        /// The reading transaction.
        reader: TxnRef,
        /// The failed transaction.
        writer: TxnRef,
        /// The key read.
        key: K,
        /// The failed transaction's value.
        value: V,
    },
    /// `G1b`, intermediate read: a committed read's last value was appended by
    /// another transaction that later appended to the same key again.
    G1b {
        /// The reading transaction.
        reader: TxnRef,
        /// The transaction whose intermediate state was read.
        writer: TxnRef,
        /// The key read.
        key: K,
        /// The intermediate value, last in the list read.
        value: V,
    },
    /// `dirty-update`: in a key's order of appends, a committed transaction's
    /// value follows a failed transaction's value.
    DirtyUpdate {
        /// The key.
        key: K,
        /// The values from the failed append through the committed one, in order.
        values: Vec<V>,
        /// The failed transaction.
        aborted: TxnRef,
        /// The committed transaction that appended on top of it.
        committed: TxnRef,
    },
    /// `lost-update`: several committed transactions read the same list for a
    /// key and then appended to it, so each one's update is based on a state
    /// the others overwrote.
    LostUpdate {
        /// The key.
        key: K,
        /// The list every transaction read before appending.
        read: Vec<V>,
        /// The transactions, in invocation order.
        txns: Vec<TxnRef>,
    },
    /// `garbage-read` (not in Elle): a committed read returned a value that no
    /// transaction in the history appended to that key.
    GarbageRead {
        /// The reading transaction.
        reader: TxnRef,
        /// The key read.
        key: K,
        /// The value nothing appended.
        value: V,
    },
    /// `append-order` (not in Elle): reads show a transaction's appends to a key
    /// out of the order it made them, or show one without the appends it made
    /// to that key before it. A transaction's appends to a key take effect in
    /// its own order, and a read containing one contains every earlier one.
    AppendOrder {
        /// The appending transaction.
        txn: TxnRef,
        /// The key.
        key: K,
        /// The transaction's values for the key, in the order it appended them.
        appended: Vec<V>,
        /// Those of its values the longest read contains, in the read's order.
        observed: Vec<V>,
    },
}

/// What a transaction's own earlier micro-operations imply a read of a key
/// must return.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expected<V> {
    /// Exactly this list: the transaction read the key before.
    List(Vec<V>),
    /// A list ending with these values: the transaction appended them without
    /// reading the key first.
    Suffix(Vec<V>),
}

impl<K, V> Anomaly<K, V> {
    /// Elle's name for the anomaly, such as `"G1a"` or `"incompatible-order"`.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::DuplicateAppends { .. } => "duplicate-appends",
            Self::DuplicateElements { .. } => "duplicate-elements",
            Self::IncompatibleOrder { .. } => "incompatible-order",
            Self::Internal { .. } => "internal",
            Self::FutureRead { .. } => "future-read",
            Self::G1a { .. } => "G1a",
            Self::G1b { .. } => "G1b",
            Self::DirtyUpdate { .. } => "dirty-update",
            Self::LostUpdate { .. } => "lost-update",
            Self::GarbageRead { .. } => "garbage-read",
            Self::AppendOrder { .. } => "append-order",
        }
    }
}

/// Every anomaly in `txns`, grouped by kind in the order of [`Anomaly`]'s
/// variants, each group in history or key order.
pub(super) fn find<K, V>(txns: &Txns<'_, K, V>) -> Vec<Anomaly<K, V>>
where
    K: Clone + Ord,
    V: Clone + Ord,
{
    let mut anomalies: Vec<Anomaly<K, V>> = txns
        .repeated_appends
        .iter()
        .map(|&(key, value, first, second)| Anomaly::DuplicateAppends {
            key: key.clone(),
            value: value.clone(),
            first: txns.records[first].reference,
            second: txns.records[second].reference,
        })
        .collect();
    let committed = || {
        txns.records
            .iter()
            .filter(|record| record.outcome == Outcome::Committed)
    };
    for record in committed() {
        anomalies.extend(duplicate_elements(record));
    }
    anomalies.extend(incompatible_orders(txns));
    for record in committed() {
        anomalies.extend(internal(record));
    }
    for record in committed() {
        anomalies.extend(future_read(record));
    }
    let mut provenance = Vec::new();
    for record in committed() {
        provenance.extend(read_provenance(txns, record));
    }
    provenance.sort_by_key(provenance_rank);
    anomalies.extend(provenance);
    anomalies.extend(dirty_updates(txns));
    anomalies.extend(lost_updates(txns));
    anomalies.extend(append_orders(txns));
    anomalies
}

fn duplicate_elements<K: Clone, V: Clone + Ord>(
    record: &TxnRecord<'_, K, V>,
) -> Vec<Anomaly<K, V>> {
    reads(record)
        .filter_map(|(key, list)| {
            let mut counts: BTreeMap<&V, usize> = BTreeMap::new();
            for value in list {
                *counts.entry(value).or_insert(0) += 1;
            }
            let duplicates: Vec<(V, usize)> = counts
                .into_iter()
                .filter(|&(_, count)| count > 1)
                .map(|(value, count)| (value.clone(), count))
                .collect();
            (!duplicates.is_empty()).then(|| Anomaly::DuplicateElements {
                reader: record.reference,
                key: key.clone(),
                duplicates,
            })
        })
        .collect()
}

/// At most one per key: the first adjacent pair, in length order, where the
/// shorter list is not a prefix of the longer.
fn incompatible_orders<K: Clone + Ord, V: Clone + Ord>(
    txns: &Txns<'_, K, V>,
) -> Vec<Anomaly<K, V>> {
    txns.observed_lists
        .iter()
        .filter_map(|(key, lists)| {
            let pair = lists
                .windows(2)
                .find(|pair| !pair[1].0.starts_with(pair[0].0))?;
            let reader = |position: usize| txns.records[position].reference;
            Some(Anomaly::IncompatibleOrder {
                key: (*key).clone(),
                shorter: (pair[0].0.to_vec(), reader(pair[0].1)),
                longer: (pair[1].0.to_vec(), reader(pair[1].1)),
            })
        })
        .collect()
}

/// The first read, if any, inconsistent with the transaction's own earlier
/// micro-operations on the same key. Elle's adaptation of the INT axiom from
/// Cerone, Bernardi and Gotsman, "A Framework for Transactional Consistency
/// Models with Atomic Visibility" (CONCUR 2015).
fn internal<K: Clone + Ord, V: Clone + PartialEq>(
    record: &TxnRecord<'_, K, V>,
) -> Option<Anomaly<K, V>> {
    let mut expected: BTreeMap<&K, Expected<V>> = BTreeMap::new();
    for micro_op in record.micro_ops {
        match micro_op {
            MicroOp::Append { key, value } => {
                match expected
                    .entry(key)
                    .or_insert_with(|| Expected::Suffix(Vec::new()))
                {
                    Expected::List(values) | Expected::Suffix(values) => values.push(value.clone()),
                }
            }
            MicroOp::Read {
                key,
                list: Some(list),
            } => {
                let consistent = match expected.get(key) {
                    None => true,
                    Some(Expected::List(values)) => list == values,
                    Some(Expected::Suffix(values)) => list.ends_with(values),
                };
                if !consistent {
                    return Some(Anomaly::Internal {
                        txn: record.reference,
                        key: key.clone(),
                        read: list.clone(),
                        expected: expected.remove(key)?,
                    });
                }
                expected.insert(key, Expected::List(list.clone()));
            }
            MicroOp::Read { list: None, .. } => {}
        }
    }
    None
}

/// The first append, if any, of a value the transaction had already read.
fn future_read<K: Clone + Ord, V: Clone + Ord>(
    record: &TxnRecord<'_, K, V>,
) -> Option<Anomaly<K, V>> {
    let mut seen: BTreeMap<&K, BTreeSet<&V>> = BTreeMap::new();
    for micro_op in record.micro_ops {
        match micro_op {
            MicroOp::Append { key, value } => {
                if seen.get(key).is_some_and(|values| values.contains(value)) {
                    return Some(Anomaly::FutureRead {
                        txn: record.reference,
                        key: key.clone(),
                        value: value.clone(),
                    });
                }
            }
            MicroOp::Read {
                key,
                list: Some(list),
            } => seen.entry(key).or_default().extend(list),
            MicroOp::Read { list: None, .. } => {}
        }
    }
    None
}

/// Values a committed read could not legitimately return: garbage reads and
/// aborted reads anywhere in the list, and an intermediate value at its end.
fn read_provenance<K: Clone + Ord, V: Clone + Ord>(
    txns: &Txns<'_, K, V>,
    record: &TxnRecord<'_, K, V>,
) -> Vec<Anomaly<K, V>> {
    let mut anomalies = Vec::new();
    for (key, list) in reads(record) {
        for value in list {
            match txns.appender(key, value) {
                None => anomalies.push(Anomaly::GarbageRead {
                    reader: record.reference,
                    key: key.clone(),
                    value: value.clone(),
                }),
                Some((_, writer)) if writer.outcome == Outcome::Aborted => {
                    anomalies.push(Anomaly::G1a {
                        reader: record.reference,
                        writer: writer.reference,
                        key: key.clone(),
                        value: value.clone(),
                    });
                }
                Some(_) => {}
            }
        }
        // A transaction may read its own intermediate state.
        if let Some(last) = list.last()
            && let Some((append, writer)) = txns.appender(key, last)
            && append.intermediate
            && writer.reference != record.reference
        {
            anomalies.push(Anomaly::G1b {
                reader: record.reference,
                writer: writer.reference,
                key: key.clone(),
                value: last.clone(),
            });
        }
    }
    anomalies
}

/// Orders read-provenance anomalies by kind (G1a, G1b, garbage-read) while
/// keeping history order within a kind.
fn provenance_rank<K, V>(anomaly: &Anomaly<K, V>) -> u8 {
    match anomaly {
        Anomaly::G1a { .. } => 0,
        Anomaly::G1b { .. } => 1,
        _ => 2,
    }
}

/// Walks each key's longest committed read, which is its order of appends as
/// far as reads revealed it, reporting each committed value that follows a
/// failed one. Indeterminate appends neither start nor end a failed run.
fn dirty_updates<K: Clone + Ord, V: Clone + Ord>(txns: &Txns<'_, K, V>) -> Vec<Anomaly<K, V>> {
    let mut anomalies = Vec::new();
    for (key, lists) in &txns.observed_lists {
        let Some(&(order, _)) = lists.last() else {
            continue;
        };
        // The latest determinate append, with the values since its own.
        let mut latest: Option<(&TxnRecord<'_, K, V>, Vec<V>)> = None;
        for value in order {
            let Some((_, writer)) = txns.appender(key, value) else {
                continue;
            };
            match (latest.as_mut(), writer.outcome) {
                (_, Outcome::Indeterminate) => {
                    if let Some((_, values)) = latest.as_mut() {
                        values.push(value.clone());
                    }
                }
                (Some((previous, values)), Outcome::Aborted)
                    if previous.outcome == Outcome::Aborted =>
                {
                    values.push(value.clone());
                }
                (Some((previous, values)), Outcome::Committed)
                    if previous.outcome == Outcome::Aborted =>
                {
                    values.push(value.clone());
                    anomalies.push(Anomaly::DirtyUpdate {
                        key: (*key).clone(),
                        values: std::mem::take(values),
                        aborted: previous.reference,
                        committed: writer.reference,
                    });
                    latest = Some((writer, vec![value.clone()]));
                }
                _ => latest = Some((writer, vec![value.clone()])),
            }
        }
    }
    anomalies
}

/// Groups committed transactions that read a key before first appending to it
/// by the list they read; every group of two or more is a lost update.
fn lost_updates<K: Clone + Ord, V: Clone + Ord>(txns: &Txns<'_, K, V>) -> Vec<Anomaly<K, V>> {
    let mut groups: BTreeMap<(&K, &[V]), Vec<TxnRef>> = BTreeMap::new();
    for record in &txns.records {
        if record.outcome != Outcome::Committed {
            continue;
        }
        let mut first_reads: BTreeMap<&K, &[V]> = BTreeMap::new();
        let mut appended: BTreeSet<&K> = BTreeSet::new();
        for micro_op in record.micro_ops {
            match micro_op {
                MicroOp::Read {
                    key,
                    list: Some(list),
                } => {
                    first_reads.entry(key).or_insert(list);
                }
                MicroOp::Append { key, .. } if appended.insert(key) => {
                    if let Some(&list) = first_reads.get(key) {
                        groups
                            .entry((key, list))
                            .or_default()
                            .push(record.reference);
                    }
                }
                MicroOp::Append { .. } | MicroOp::Read { list: None, .. } => {}
            }
        }
    }
    groups
        .into_iter()
        .filter(|(_, group)| group.len() > 1)
        .map(|((key, read), group)| Anomaly::LostUpdate {
            key: key.clone(),
            read: read.to_vec(),
            txns: group,
        })
        .collect()
}

/// Compares each transaction's appends to each key with their order in the
/// key's longest committed read: the values the read contains must be the
/// transaction's first ones, in the order it appended them.
fn append_orders<K: Clone + Ord, V: Clone + Ord>(txns: &Txns<'_, K, V>) -> Vec<Anomaly<K, V>> {
    let mut position: BTreeMap<(&K, &V), usize> = BTreeMap::new();
    for (key, lists) in &txns.observed_lists {
        if let Some(&(order, _)) = lists.last() {
            for (index, value) in order.iter().enumerate() {
                position.entry((*key, value)).or_insert(index);
            }
        }
    }

    let mut anomalies = Vec::new();
    for record in &txns.records {
        if record.outcome == Outcome::Aborted {
            continue;
        }
        let mut appended: BTreeMap<&K, Vec<&V>> = BTreeMap::new();
        for micro_op in record.micro_ops {
            if let MicroOp::Append { key, value } = micro_op {
                appended.entry(key).or_default().push(value);
            }
        }
        for (key, values) in appended {
            let positions: Vec<Option<usize>> = values
                .iter()
                .map(|value| position.get(&(key, *value)).copied())
                .collect();
            let observed_count = positions.iter().take_while(|p| p.is_some()).count();
            let prefix_observed = positions[observed_count..].iter().all(Option::is_none);
            let in_order = positions[..observed_count]
                .windows(2)
                .all(|pair| pair[0] < pair[1]);
            if prefix_observed && in_order {
                continue;
            }
            let mut observed: Vec<(usize, &V)> = values
                .iter()
                .zip(&positions)
                .filter_map(|(value, p)| p.map(|p| (p, *value)))
                .collect();
            observed.sort_unstable();
            anomalies.push(Anomaly::AppendOrder {
                txn: record.reference,
                key: key.clone(),
                appended: values.into_iter().cloned().collect(),
                observed: observed.into_iter().map(|(_, v)| v.clone()).collect(),
            });
        }
    }
    anomalies
}

/// The committed reads of a transaction, as (key, list) pairs.
fn reads<'r, K, V>(record: &'r TxnRecord<'_, K, V>) -> impl Iterator<Item = (&'r K, &'r [V])> {
    record
        .micro_ops
        .iter()
        .filter_map(|micro_op| match micro_op {
            MicroOp::Read {
                key,
                list: Some(list),
            } => Some((key, list.as_slice())),
            _ => None,
        })
}
