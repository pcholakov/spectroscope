//! Spectra: Consistency and isolation level checkers for distributed systems testing.
//!
//! This library provides checkers for verifying that operation histories conform to
//! various consistency models (linearizability, serializability, snapshot isolation, etc.)
//!
//! Derived from Jepsen's checker implementations, licensed under EPL-1.0.

pub mod history;
pub mod set_full;

pub use history::{History, Op, OpType};
pub use set_full::{SetFullChecker, SetFullOptions, SetFullResult};
