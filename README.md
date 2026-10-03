# Spectroscope

Consistency checkers for distributed systems testing.

Spectroscope verifies that operation histories from distributed systems conform to expected consistency models. It ports [Jepsen](https://github.com/jepsen-io/jepsen)'s set-full workload checker and adds an exact linearizability mode.

## Installation

```toml
[dependencies]
spectroscope = "0.1"
```

## Usage

```rust
use spectroscope::history::{History, Op, Pid};
use spectroscope::set_full::{SetFullChecker, Validity};

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

When the history is not linearizable, `result.linearizability_violation` holds a
cycle of orderings that cannot all hold, for example:

```text
Add(1) before Read(2..4)  -- the read observed 1
Read(2..4) before Add(2)  -- the read missed 2
Add(2) before Read(3..5)  -- the read observed 2
Read(3..5) before Add(1)  -- the read missed 1
```

## License

EPL-1.0
