# Spectra

Consistency checkers for distributed systems testing.

Spectra verifies that operation histories from distributed systems conform to expected consistency models. Currently implements a set linearizability checker derived from [Jepsen](https://github.com/jepsen-io/jepsen).

## Installation

```toml
[dependencies]
spectra = "0.1"
```

## Usage

```rust
use std::time::Duration;
use spectra::{History, Op, SetFullChecker, Validity};

// Build a history of operations from your distributed system
let mut history = History::new();

// Process 0 adds element 1
history.push(Op::add_invoke(0, 0u64, 1));
history.push(Op::add_ok(1, 0u64, 1));

// Process 1 reads and sees the element
history.push(Op::read_invoke(2, 1u64));
history.push(Op::read_ok(3, 1u64, [1]));

// Check the history for consistency violations
let result = SetFullChecker::default().check(&history);

assert_eq!(result.valid, Validity::Valid);
assert_eq!(result.stable_count, 1);  // 1 element confirmed visible
assert_eq!(result.lost_count, 0);    // no elements lost
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

EPL-1.0 (derived from Jepsen)
