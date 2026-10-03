//! Checks `ListAppendChecker` against a brute-force oracle.
//!
//! The oracle decides each consistency model from its definition: some subset
//! of the indeterminate transactions, together with every committed one, runs
//! in some serial order that respects the model's ordering constraints and
//! reproduces every committed read. Comparing verdicts on many small random
//! histories tests both directions of the checker's claim: it never rejects a
//! history the model allows (soundness), and it never accepts one the model
//! forbids (completeness of the dependency inference). These are bounded
//! results: they hold for the histories generated here, not for all histories.

#![allow(
    clippy::unwrap_used,
    reason = "test helpers fail loudly on bad fixtures"
)]

use std::collections::BTreeMap;

use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use rand::{Rng, SeedableRng};
use spectroscope::{
    ConsistencyModel, History, ListAppendChecker, MicroOp, Op, OpType, Pid, Txn, Validity,
};

type Key = u16;
type Value = u32;
type TxnOp = Op<Txn<Key, Value>>;

const MODELS: [ConsistencyModel; 3] = [
    ConsistencyModel::Serializable,
    ConsistencyModel::StrongSessionSerializable,
    ConsistencyModel::StrictSerializable,
];

/// How simulated transactions see the store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Isolation {
    /// A transaction runs atomically at one point between its invocation and
    /// completion, so every history is strictly serializable.
    Serial,
    /// A transaction reads a snapshot taken at invocation and its appends land
    /// atomically, allowing write skew.
    Snapshot,
    /// One micro-operation per step: reads see the committed store plus the
    /// transaction's own appends, which land when it commits.
    ReadCommitted,
    /// One micro-operation per step, applied to the store immediately: others
    /// see partial effects, and a failed or crashed transaction leaves them.
    ReadUncommitted,
}

#[derive(Debug, Clone)]
struct Config {
    txns: usize,
    processes: usize,
    keys: Key,
    micro_ops_max: usize,
    isolation: Isolation,
    fail_probability: f64,
    info_probability: f64,
    /// Probability that a transaction whose client saw `Info` keeps running and
    /// may commit later.
    detach_probability: f64,
    /// Probability that a read returns a random prefix of what it should.
    stale_probability: f64,
    /// Probability of corrupting one committed read result after simulation.
    perturb_probability: f64,
    /// Scheduler steps to run once every transaction has started; the rest
    /// stay pending.
    steps_after_last_start: usize,
}

/// A transaction in progress.
#[derive(Debug, Clone)]
struct Running {
    plan: Vec<MicroOp<Key, Value>>,
    /// Micro-operations executed so far, with read results.
    completed: Vec<MicroOp<Key, Value>>,
    /// The store at invocation, for snapshot reads.
    snapshot: BTreeMap<Key, Vec<Value>>,
    /// Appends not yet visible to others, for read-committed.
    buffer: Vec<(Key, Value)>,
}

impl Running {
    fn done(&self) -> bool {
        self.completed.len() == self.plan.len()
    }

    /// Executes the next micro-operation, or all of them when the isolation
    /// level is atomic. Commits buffered appends once every one has run.
    fn step(&mut self, store: &mut BTreeMap<Key, Vec<Value>>, config: &Config, rng: &mut StdRng) {
        let atomic = matches!(config.isolation, Isolation::Serial | Isolation::Snapshot);
        while !self.done() {
            let micro_op = self.plan[self.completed.len()].clone();
            let executed = match micro_op {
                MicroOp::Append { key, value } => {
                    match config.isolation {
                        Isolation::ReadCommitted => self.buffer.push((key, value)),
                        _ => store.entry(key).or_default().push(value),
                    }
                    if config.isolation == Isolation::Snapshot {
                        self.snapshot.entry(key).or_default().push(value);
                    }
                    micro_op
                }
                MicroOp::Read { key, .. } => {
                    let mut list = match config.isolation {
                        Isolation::Snapshot => self.snapshot.get(&key).cloned(),
                        _ => store.get(&key).cloned(),
                    }
                    .unwrap_or_default();
                    for &(buffered_key, value) in &self.buffer {
                        if buffered_key == key {
                            list.push(value);
                        }
                    }
                    if rng.random_bool(config.stale_probability) {
                        list.truncate(rng.random_range(0..=list.len()));
                    }
                    MicroOp::read_result(key, list)
                }
            };
            self.completed.push(executed);
            if !atomic {
                break;
            }
        }
        if self.done() {
            for (key, value) in self.buffer.drain(..) {
                store.entry(key).or_default().push(value);
            }
        }
    }
}

/// Runs `config.txns` transactions on `config.processes` concurrent processes
/// against an in-memory store. Each scheduler step advances one random process,
/// and sometimes one detached transaction. Transactions still running when the
/// step budget runs out stay pending, possibly with visible effects.
fn simulate(config: &Config, rng: &mut StdRng) -> Vec<TxnOp> {
    let mut store: BTreeMap<Key, Vec<Value>> = BTreeMap::new();
    let mut slots: Vec<(u64, Option<Running>)> = (0..).zip(vec![None; config.processes]).collect();
    let mut next_pid = u64::try_from(config.processes).unwrap();
    // Transactions whose client saw `Info` but that keep running.
    let mut detached: Vec<Running> = Vec::new();
    let mut next_value: Value = 0;
    let mut started = 0;
    let mut ops: Vec<TxnOp> = Vec::new();

    let mut steps_after_last_start = config.steps_after_last_start;
    for _ in 0..config.txns * 12 {
        if started == config.txns {
            if steps_after_last_start == 0 {
                break;
            }
            steps_after_last_start -= 1;
        }
        if !detached.is_empty() && rng.random_bool(0.2) {
            let which = rng.random_range(0..detached.len());
            detached[which].step(&mut store, config, rng);
            if detached[which].done() {
                detached.swap_remove(which);
            }
        }
        let slot = rng.random_range(0..slots.len());
        let pid = slots[slot].0;
        let index = ops.len();
        let Some(mut running) = slots[slot].1.take() else {
            if started < config.txns {
                started += 1;
                let plan = random_plan(config, &mut next_value, rng);
                ops.push(Op::txn_invoke(index, pid, plan.clone()));
                slots[slot].1 = Some(Running {
                    plan,
                    completed: Vec::new(),
                    snapshot: store.clone(),
                    buffer: Vec::new(),
                });
            }
            continue;
        };
        if !running.done() && rng.random_bool(config.fail_probability) {
            // Read-committed discards its buffer; read-uncommitted leaves its
            // effects behind. Once every effect has landed it cannot fail.
            ops.push(Op::txn_fail(index, pid, running.plan));
        } else if rng.random_bool(config.info_probability) {
            ops.push(Op::txn_info(index, pid, running.plan.clone()));
            next_pid += 1;
            slots[slot].0 = next_pid;
            if !running.done() && rng.random_bool(config.detach_probability) {
                detached.push(running);
            }
        } else if running.done() {
            // Effects took place on an earlier step; now the client hears back.
            ops.push(Op::txn_ok(index, pid, running.completed));
        } else {
            running.step(&mut store, config, rng);
            slots[slot].1 = Some(running);
        }
    }
    if rng.random_bool(config.perturb_probability) {
        perturb(&mut ops, rng);
    }
    ops
}

fn random_plan(
    config: &Config,
    next_value: &mut Value,
    rng: &mut StdRng,
) -> Vec<MicroOp<Key, Value>> {
    (0..rng.random_range(1..=config.micro_ops_max))
        .map(|_| {
            let key = rng.random_range(0..config.keys);
            if rng.random_bool(0.5) {
                *next_value += 1;
                MicroOp::append(key, *next_value)
            } else {
                MicroOp::read(key)
            }
        })
        .collect()
}

/// Corrupts one committed read: drops its last value, swaps two values, adds a
/// value, or keeps only a prefix.
fn perturb(ops: &mut [TxnOp], rng: &mut StdRng) {
    let reads: Vec<(usize, usize)> = ops
        .iter()
        .enumerate()
        .filter(|(_, op)| op.op_type == OpType::Ok)
        .flat_map(|(position, op)| {
            op.value
                .0
                .iter()
                .enumerate()
                .filter_map(move |(i, micro_op)| {
                    matches!(micro_op, MicroOp::Read { .. }).then_some((position, i))
                })
        })
        .collect();
    let Some(&(position, i)) = reads.choose(rng) else {
        return;
    };
    let appended: Vec<Value> = ops
        .iter()
        .flat_map(|op| &op.value.0)
        .filter_map(|micro_op| match micro_op {
            MicroOp::Append { value, .. } => Some(*value),
            MicroOp::Read { .. } => None,
        })
        .collect();
    let MicroOp::Read {
        list: Some(list), ..
    } = &mut ops[position].value.0[i]
    else {
        return;
    };
    match rng.random_range(0..4) {
        0 => {
            list.pop();
        }
        1 if list.len() >= 2 => {
            let j = rng.random_range(1..list.len());
            list.swap(j - 1, j);
        }
        2 => {
            if let Some(&value) = appended.choose(rng) {
                list.push(value);
            }
        }
        _ => {
            let keep = rng.random_range(0..=list.len());
            list.truncate(keep);
        }
    }
}

/// One transaction as the oracle sees it.
#[derive(Debug)]
struct OracleTxn {
    process: Pid,
    invoke_index: usize,
    /// Completion index if the transaction committed.
    commit_index: Option<usize>,
    committed: bool,
    aborted: bool,
    /// Never completed.
    pending: bool,
    micro_ops: Vec<MicroOp<Key, Value>>,
}

fn oracle_txns(ops: &[TxnOp]) -> Vec<OracleTxn> {
    let mut txns: Vec<OracleTxn> = Vec::new();
    let mut open: BTreeMap<Pid, usize> = BTreeMap::new();
    for op in ops {
        if op.op_type == OpType::Invoke {
            open.insert(op.process, txns.len());
            txns.push(OracleTxn {
                process: op.process,
                invoke_index: op.index,
                commit_index: None,
                committed: false,
                aborted: false,
                pending: true,
                micro_ops: op.value.0.clone(),
            });
        } else {
            let txn = &mut txns[open.remove(&op.process).unwrap()];
            txn.pending = false;
            match op.op_type {
                OpType::Ok => {
                    txn.committed = true;
                    txn.commit_index = Some(op.index);
                    txn.micro_ops.clone_from(&op.value.0);
                }
                OpType::Fail => txn.aborted = true,
                OpType::Info | OpType::Invoke => {}
            }
        }
    }
    txns
}

/// Whether `model` allows the history: some serial order of the committed
/// transactions and some of the indeterminate ones respects the model's
/// ordering constraints and reproduces every committed read.
fn oracle(txns: &[OracleTxn], model: ConsistencyModel) -> bool {
    let must_precede = |before: &OracleTxn, after: &OracleTxn| match model {
        ConsistencyModel::Serializable => false,
        ConsistencyModel::StrongSessionSerializable => {
            before.process == after.process && before.invoke_index < after.invoke_index
        }
        ConsistencyModel::StrictSerializable => before
            .commit_index
            .is_some_and(|commit| commit < after.invoke_index),
    };
    let candidates: Vec<usize> = (0..txns.len()).filter(|&t| !txns[t].aborted).collect();
    let optional: Vec<usize> = candidates
        .iter()
        .copied()
        .filter(|&t| !txns[t].committed)
        .collect();
    (0..1u32 << optional.len()).any(|mask| {
        let included: Vec<usize> = candidates
            .iter()
            .copied()
            .filter(|t| {
                optional
                    .iter()
                    .position(|o| o == t)
                    .is_none_or(|bit| mask & (1 << bit) != 0)
            })
            .collect();
        serial_order_exists(
            txns,
            &included,
            &mut Vec::new(),
            &BTreeMap::new(),
            &must_precede,
        )
    })
}

/// Depth-first search over serial orders of `included`, extending `placed`.
fn serial_order_exists(
    txns: &[OracleTxn],
    included: &[usize],
    placed: &mut Vec<usize>,
    store: &BTreeMap<Key, Vec<Value>>,
    must_precede: &dyn Fn(&OracleTxn, &OracleTxn) -> bool,
) -> bool {
    if placed.len() == included.len() {
        return true;
    }
    for &next in included {
        if placed.contains(&next) {
            continue;
        }
        let blocked = included.iter().any(|&other| {
            other != next && !placed.contains(&other) && must_precede(&txns[other], &txns[next])
        });
        if blocked {
            continue;
        }
        let Some(after) = execute(&txns[next], store) else {
            continue;
        };
        placed.push(next);
        if serial_order_exists(txns, included, placed, &after, must_precede) {
            return true;
        }
        placed.pop();
    }
    false
}

/// The store after running `txn` on `store`, or `None` if a committed read
/// would return a different list.
fn execute(
    txn: &OracleTxn,
    store: &BTreeMap<Key, Vec<Value>>,
) -> Option<BTreeMap<Key, Vec<Value>>> {
    let mut store = store.clone();
    for micro_op in &txn.micro_ops {
        match micro_op {
            MicroOp::Append { key, value } => store.entry(*key).or_default().push(*value),
            MicroOp::Read {
                key,
                list: Some(list),
            } => {
                if store.get(key).map_or(&[][..], Vec::as_slice) != list.as_slice() {
                    return None;
                }
            }
            MicroOp::Read { list: None, .. } => {}
        }
    }
    Some(store)
}

fn random_config(rng: &mut StdRng) -> Config {
    let isolation = *[
        Isolation::Serial,
        Isolation::Snapshot,
        Isolation::ReadCommitted,
        Isolation::ReadUncommitted,
    ]
    .choose(rng)
    .unwrap();
    Config {
        txns: rng.random_range(2..=6),
        processes: rng.random_range(1..=3),
        keys: rng.random_range(1..=3),
        micro_ops_max: rng.random_range(1..=3),
        isolation,
        fail_probability: 0.1,
        info_probability: 0.1,
        detach_probability: 0.7,
        stale_probability: if rng.random_bool(0.2) { 0.3 } else { 0.0 },
        perturb_probability: 0.2,
        steps_after_last_start: if rng.random_bool(0.5) {
            rng.random_range(0..16)
        } else {
            usize::MAX
        },
    }
}

/// The checker and the oracle agree on every model for many small histories.
#[test]
fn test_checker_matches_oracle_on_small_histories() {
    let mut verdicts: BTreeMap<(ConsistencyModel, bool), usize> = BTreeMap::new();
    // Histories each model allows while the next stronger one forbids.
    let mut separations = [0usize; 2];
    // How often each anomaly and cycle class is reported, across models.
    let mut classes: BTreeMap<&str, usize> = BTreeMap::new();
    // Per model, invalid histories whose only evidence is a cycle.
    let mut cycles_only = [0usize; 3];
    // Histories where a committed read observed an append by a transaction
    // that may not have committed: completed `Info`, or never completed.
    let mut info_observed = 0;
    let mut pending_observed = 0;
    for seed in 0..40_000 {
        let mut rng = StdRng::seed_from_u64(seed);
        let config = random_config(&mut rng);
        let ops = simulate(&config, &mut rng);
        let txns = oracle_txns(&ops);
        if observes_append(&txns, |txn| !txn.committed && !txn.aborted && !txn.pending) {
            info_observed += 1;
        }
        if observes_append(&txns, |txn| txn.pending) {
            pending_observed += 1;
        }
        let history = History::from_ops(ops.clone()).unwrap();
        let mut allowed_by = [false; 3];
        for (m, model) in MODELS.into_iter().enumerate() {
            let result = ListAppendChecker::new(model).check(&history);
            let allowed = oracle(&txns, model);
            let expected = if allowed {
                Validity::Valid
            } else {
                Validity::Invalid
            };
            assert_eq!(
                result.valid, expected,
                "seed {seed}, {model:?}, {config:?}\nops: {ops:#?}\nresult: {result:#?}"
            );
            *verdicts.entry((model, allowed)).or_default() += 1;
            allowed_by[m] = allowed;
            for anomaly in &result.anomalies {
                *classes.entry(anomaly.name()).or_default() += 1;
            }
            for cycle in &result.cycles {
                *classes.entry(cycle.anomaly.name()).or_default() += 1;
            }
            if result.anomalies.is_empty() && !result.cycles.is_empty() {
                cycles_only[m] += 1;
            }
        }
        for (weaker, count) in separations.iter_mut().enumerate() {
            if allowed_by[weaker] && !allowed_by[weaker + 1] {
                *count += 1;
            }
        }
    }
    // Non-vacuity: both verdicts occur often under every model.
    for model in MODELS {
        for allowed in [false, true] {
            let count = verdicts.get(&(model, allowed)).copied().unwrap_or(0);
            assert!(
                count > 1_000,
                "{model:?} allowed={allowed}: only {count} histories"
            );
        }
    }
    // Process and real-time edges decide some histories.
    assert!(
        separations.iter().all(|&count| count > 100),
        "{separations:?}"
    );
    // The graph decides many histories on its own, under every model.
    assert!(
        cycles_only.iter().all(|&count| count > 200),
        "{cycles_only:?}"
    );
    assert!(info_observed > 100, "{info_observed}");
    assert!(pending_observed > 100, "{pending_observed}");
    // G-nonadjacent needs four transactions in a precise shape, so any of its
    // variants will do.
    assert!(
        classes
            .keys()
            .any(|class| class.starts_with("G-nonadjacent-item"))
    );
    // Every other class occurs.
    for (class, floor) in [
        ("G0", 20),
        ("G0-process", 10),
        ("G0-realtime", 10),
        ("G1c", 20),
        ("G1c-process", 20),
        ("G1c-realtime", 20),
        ("G-single-item", 20),
        ("G-single-item-process", 20),
        ("G-single-item-realtime", 20),
        ("G2-item", 20),
        ("G2-item-realtime", 1),
        ("G1a", 20),
        ("G1b", 20),
        ("append-order", 20),
        ("dirty-update", 20),
        ("duplicate-elements", 20),
        ("future-read", 20),
        ("garbage-read", 20),
        ("incompatible-order", 20),
        ("internal", 20),
        ("lost-update", 20),
    ] {
        let count = classes.get(class).copied().unwrap_or(0);
        assert!(count >= floor, "{class}: {count} < {floor}");
    }
}

/// Whether a committed read observed a value appended by a transaction that
/// satisfies `writer`.
fn observes_append(txns: &[OracleTxn], writer: impl Fn(&OracleTxn) -> bool) -> bool {
    let indeterminate: Vec<Value> = txns
        .iter()
        .filter(|txn| writer(txn))
        .flat_map(|txn| &txn.micro_ops)
        .filter_map(|micro_op| match micro_op {
            MicroOp::Append { value, .. } => Some(*value),
            MicroOp::Read { .. } => None,
        })
        .collect();
    txns.iter()
        .filter(|txn| txn.committed)
        .flat_map(|txn| &txn.micro_ops)
        .any(|micro_op| match micro_op {
            MicroOp::Read {
                list: Some(list), ..
            } => list.iter().any(|value| indeterminate.contains(value)),
            _ => false,
        })
}

/// Serial simulations are strictly serializable by construction.
#[test]
fn test_serial_simulations_are_valid() {
    for seed in 0..200 {
        let mut rng = StdRng::seed_from_u64(seed);
        let config = Config {
            txns: 500,
            processes: 8,
            keys: 10,
            micro_ops_max: 4,
            isolation: Isolation::Serial,
            fail_probability: 0.05,
            info_probability: 0.05,
            detach_probability: 0.0,
            stale_probability: 0.0,
            perturb_probability: 0.0,
            steps_after_last_start: usize::MAX,
        };
        let history = History::from_ops(simulate(&config, &mut rng)).unwrap();
        let result = ListAppendChecker::new(ConsistencyModel::StrictSerializable).check(&history);
        assert_eq!(result.valid, Validity::Valid, "seed {seed}: {result:#?}");
    }
}

/// Many readers of a key's final state followed by many appends no read
/// observes: each reader precedes each appender, which must not cost an edge
/// per pair. Quadratic edges would need hundreds of millions of edges here.
#[test]
fn test_unobserved_appends_after_many_readers_stay_linear() {
    let n: u32 = 20_000;
    let mut ops: Vec<TxnOp> = Vec::new();
    let mut push_txn = |process: u64, invoked: Vec<MicroOp<Key, Value>>, completed| {
        let index = ops.len();
        ops.push(Op::txn_invoke(index, process, invoked));
        ops.push(Op::txn_ok(index + 1, process, completed));
    };
    push_txn(0, vec![MicroOp::append(0, 0)], vec![MicroOp::append(0, 0)]);
    for reader in 0..n {
        let process = u64::from(reader % 8);
        push_txn(
            process,
            vec![MicroOp::read(0)],
            vec![MicroOp::read_result(0, [0])],
        );
    }
    for value in 1..=n {
        let process = u64::from(value % 8);
        push_txn(
            process,
            vec![MicroOp::append(0, value)],
            vec![MicroOp::append(0, value)],
        );
    }
    let history = History::from_ops(ops).unwrap();
    let result = ListAppendChecker::new(ConsistencyModel::StrictSerializable).check(&history);
    assert_eq!(result.valid, Validity::Valid);
}

/// Run with `cargo test --release -- --ignored --nocapture` to time the checker.
#[test]
#[ignore = "timing, not correctness"]
fn time_large_serial_history() {
    let mut rng = StdRng::seed_from_u64(0);
    let config = Config {
        txns: 100_000,
        processes: 32,
        keys: 1_000,
        micro_ops_max: 4,
        isolation: Isolation::Serial,
        fail_probability: 0.05,
        info_probability: 0.01,
        detach_probability: 0.0,
        stale_probability: 0.0,
        perturb_probability: 0.0,
        steps_after_last_start: usize::MAX,
    };
    let history = History::from_ops(simulate(&config, &mut rng)).unwrap();
    let start = std::time::Instant::now();
    let result = ListAppendChecker::new(ConsistencyModel::StrictSerializable).check(&history);
    println!("{} ops checked in {:?}", history.len(), start.elapsed());
    assert_eq!(result.valid, Validity::Valid);
}
