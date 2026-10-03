# Spectroscope

Consistency checkers for distributed systems testing.

Spectroscope verifies that operation histories from distributed systems conform to expected consistency models. It provides:

- a set checker that ports [Jepsen](https://github.com/jepsen-io/jepsen)'s set-full workload checker and adds an exact linearizability mode;
- a list-append transaction checker modelled on [Elle](https://github.com/jepsen-io/elle) (Kingsbury and Alvaro, ["Elle: Inferring Isolation Anomalies from Experimental Observations"](https://www.vldb.org/pvldb/vol14/p268-alvaro.pdf), VLDB 2020), working towards strict serializability checking.

## Installation

```toml
[dependencies]
spectroscope = "0.1"
```

## Usage

```rust
use spectroscope::history::{History, Op, Pid};
use spectroscope::Validity;
use spectroscope::set_full::SetFullChecker;

// Workload set value type
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Val(u32);

// Build a history of operations from your distributed system
let mut history = History::new();

// Process 0 adds element 1
history.push(Op::add_invoke(0, Pid(0), Val(1)).at_millis(0))?;
history.push(Op::add_ok(1, Pid(0), Val(1)).at_millis(5))?;
//                      ^  ^^^^^^  ^^^^^^  ^^^^^^^^^^^^
//                      |  |       |       timestamp relative to test start
//                      |  |       value being added
//                      |  process id
//                      operation index

// Process 1 reads and sees the element
history.push(Op::read_invoke(2, Pid(1)).at_millis(10))?;
history.push(Op::read_ok(3, Pid(1), [Val(1)]).at_millis(12))?;

// Check the history for consistency violations
let result = SetFullChecker::default().check(&history);

assert_eq!(result.valid, Validity::Valid);
assert_eq!(result.stable_count, 1);        // 1 element confirmed visible
assert_eq!(result.lost_count, 0);          // no elements lost
```

`History::push` returns an error for malformed histories: a completion without a
matching invocation, a second invocation while a process already has one in flight,
an invocation after the process crashed (`Info`), or non-increasing indices or
timestamps.

## What it checks

The **set-full checker** tracks elements through add and read operations:

- **Stable**: Element is visible in all reads after being added
- **Lost**: Element was confirmed added but later disappeared
- **Stale**: Element took multiple reads to become visible
- **Never-read**: Element was added but no subsequent reads occurred

A history is **valid** if no elements are lost, no read returns duplicates, and every
element a read returns has an add in the history that did not fail. The history must
be complete: an element added by a process outside it is reported in
`observed_without_add`.

## Linearizable mode

```rust
let checker = SetFullChecker::linearizable();
let result = checker.check(&history);
```

Linearizable mode decides whether the whole history is linearizable. That is stronger
than requiring each element to be visible without staleness: a read returns the
entire set at once, so all readers must also agree on a single order in which elements
appeared. For example, two concurrent reads that return `{1}` and `{2}` while both
adds are in flight have no stale elements, but no single order explains both.

When the history's orderings contradict each other, `result.linearizability_violation`
holds a cycle of orderings that cannot all hold. Duplicated elements and elements read
without any add make a history invalid without a cycle; see `duplicated` and
`observed_without_add`. An example cycle:

```text
Add(1) before Read(2..4)  -- the read observed 1
Read(2..4) before Add(2)  -- the read missed 2
Add(2) before Read(3..5)  -- the read observed 2
Read(3..5) before Add(1)  -- the read missed 1
```

## List-append transactions

Transactions append unique values to per-key lists and read whole lists, so each read
reveals the order in which the writes it saw were applied. A system without a native
append can emulate one with a read and a write of the extended list in one transaction.

`ListAppendChecker` reports the anomalies visible without a dependency graph, using
Elle's names so results can be compared: `duplicate-appends`, `duplicate-elements`,
`incompatible-order`, `internal`, `future-read`, `G1a`, `G1b`, `dirty-update` and
`lost-update`, plus `garbage-read` (a read of a value nothing appended), which Elle does
not report. Cycle detection for serializability and its real-time variants is not
implemented yet, so a history without these anomalies is reported as `Unknown`.

```rust
use spectroscope::{History, ListAppendChecker, MicroOp, Op, Validity};

let history = History::from_ops([
    Op::txn_invoke(0, 0u64, [MicroOp::append("x", 1)]),
    Op::txn_ok(1, 0u64, [MicroOp::append("x", 1)]),
    Op::txn_invoke(2, 1u64, [MicroOp::read("x")]),
    Op::txn_ok(3, 1u64, [MicroOp::read_result("x", [1, 1])]),
])?;
let result = ListAppendChecker.check(&history);
assert_eq!(result.valid, Validity::Invalid);
assert_eq!(result.anomalies[0].name(), "duplicate-elements");
```

## License

EPL-1.0
