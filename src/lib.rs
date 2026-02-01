//! Consistency checkers for distributed systems testing.
//!
//! Spectra verifies that operation histories from distributed systems conform to
//! expected consistency models. Currently implements a set linearizability checker
//! derived from [Jepsen](https://github.com/jepsen-io/jepsen).
//!
//! # Quick Start
//!
//! ```
//! use spectra::{History, Op, SetFullChecker, Validity};
//!
//! let mut history = History::new();
//!
//! // Process 0 adds element 1
//! history.push(Op::add_invoke(0, 0u64, 1));
//! history.push(Op::add_ok(1, 0u64, 1));
//!
//! // Process 1 reads and sees it
//! history.push(Op::read_invoke(2, 1u64));
//! history.push(Op::read_ok(3, 1u64, [1]));
//!
//! let result = SetFullChecker::default().check(&history);
//! assert_eq!(result.valid, Validity::Valid);
//! ```
//!
//! # Set-Full Checker
//!
//! The [`SetFullChecker`] analyzes histories of set operations (add elements, read the set)
//! and determines whether elements were properly persisted:
//!
//! - **Stable**: Element visible in all reads after being added
//! - **Lost**: Element confirmed added but later disappeared
//! - **Stale**: Element took multiple reads to become visible
//! - **Never-read**: Element added but no subsequent reads occurred
//!
//! Use [`SetFullChecker::linearizable()`] for strict linearizability checking where
//! elements must appear immediately after being added.

pub mod history;
pub mod set_full;

pub use history::{History, Op, OpFn, OpType, OpValue, ProcessId};
pub use set_full::{
    ElementOutcome, SetFullChecker, SetFullOptions, SetFullResult, Validity, WorstStaleEntry,
};
