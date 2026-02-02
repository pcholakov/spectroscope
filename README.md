# Spectroscope

Consistency checkers for distributed systems testing.

Spectroscope verifies that operation histories from distributed systems conform to expected consistency models. This is a port of [Jepsen](https://github.com/jepsen-io/jepsen)'s set-full workload linearizability checker.

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
history.push(Op::add_invoke(0, Pid(0), Val(1)).at_millis(0));
history.push(Op::add_ok(1, Pid(0), Val(1)).at_millis(5));
//                      ^  ^^^^^^  ^^^^^^  ^^^^^^^^^^^^
//                      |  |       |       timestamp relative to test start
//                      |  |       value being added
//                      |  process id
//                      operation index

// Process 1 reads and sees the element
history.push(Op::read_invoke(2, Pid(1)).at_millis(10));
history.push(Op::read_ok(3, Pid(1), [Val(1)]).at_millis(12));

// Check the history for consistency violations
let result = SetFullChecker::default().check(&history);

assert_eq!(result.valid, Validity::Valid);
assert_eq!(result.stable_count, 1);        // 1 element confirmed visible
assert_eq!(result.lost_count, 0);          // no elements lost
```

## What it checks

The **set-full checker** tracks elements through add and read operations:

- **Stable**: Element is visible in all reads after being added
- **Lost**: Element was confirmed added but later disappeared
- **Stale**: Element took multiple reads to become visible (non-linearizable)
- **Never-read**: Element was added but no subsequent reads occurred

A history is **valid** if no elements are lost (and no stale elements when `linearizable` mode is enabled).

## Linearizable mode

For strict linearizability, elements must appear immediately after being added:

```rust
let checker = SetFullChecker::linearizable();
let result = checker.check(&history);
// Fails if any element has non-zero visibility latency
```

## License

EPL-1.0
