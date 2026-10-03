//! Consistency checkers for distributed systems testing.
//!
//! Spectroscope verifies that operation histories from distributed systems conform to
//! expected consistency models. Currently implements a set checker derived from
//! [Jepsen](https://github.com/jepsen-io/jepsen)'s set-full workload, with an exact
//! linearizability mode.
//!
//! # Quick Start
//!
//! ```
//! use spectroscope::{History, Op, Pid, SetFullChecker, Validity};
//!
//! let mut history = History::new();
//!
//! // Process 0 adds element 1
//! history.push(Op::add_invoke(0, Pid(0), 1).at_millis(0))?;
//! history.push(Op::add_ok(1, Pid(0), 1).at_millis(5))?;
//!
//! // Process 1 reads and sees it
//! history.push(Op::read_invoke(2, Pid(1)).at_millis(10))?;
//! history.push(Op::read_ok(3, Pid(1), [1]).at_millis(12))?;
//!
//! let result = SetFullChecker::default().check(&history);
//! assert_eq!(result.valid, Validity::Valid);
//! # Ok::<(), spectroscope::HistoryError>(())
//! ```
//!
//! [`History::push`] rejects malformed histories, such as a completion without an
//! invocation or a process with two operations in flight, so checkers only see
//! well-formed input.
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
//! Use [`SetFullChecker::linearizable()`] to also decide whether the whole history is
//! linearizable. When its orderings contradict each other, the result carries a
//! [`PrecedenceCycle`] proving it. Duplicates, and elements read without any add,
//! make a history invalid without a cycle.

pub mod history;
pub mod set_full;

pub use history::{History, HistoryError, Op, OpFn, OpType, OpValue, Pid, Timestamp};
pub use set_full::{
    ElementOutcome, Event, Precedence, PrecedenceCycle, Reason, SetFullChecker, SetFullOptions,
    SetFullResult, Validity, WorstStaleEntry,
};
