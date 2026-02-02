//! Basic usage of the set-full linearizability checker.
//!
//! Run with: cargo run --example basic

use spectroscope::history::{History, Op, Pid};
use spectroscope::set_full::{SetFullChecker, Validity};

/// A value being added to or read from the set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Val(u32);

fn main() {
    // A history records operations from concurrent processes.
    // Each operation has an invocation (start) and completion (ok/fail).
    let mut history = History::new();

    // Process 0 adds element 1
    history.push(Op::add_invoke(0, Pid(0), Val(1)).at_millis(0));
    history.push(Op::add_ok(1, Pid(0), Val(1)).at_millis(5));

    // Process 1 adds element 2
    history.push(Op::add_invoke(2, Pid(1), Val(2)).at_millis(2));
    history.push(Op::add_ok(3, Pid(1), Val(2)).at_millis(7));

    // Process 2 reads and sees both elements
    history.push(Op::read_invoke(4, Pid(2)).at_millis(10));
    history.push(Op::read_ok(5, Pid(2), [Val(1), Val(2)]).at_millis(12));

    // Check the history
    let result = SetFullChecker::default().check(&history);

    println!("Validity: {:?}", result.valid);
    println!("Elements added: {}", result.attempt_count);
    println!("Stable (visible): {}", result.stable_count);
    println!("Lost: {}", result.lost_count);

    assert_eq!(result.valid, Validity::Valid);

    // Now let's create an invalid history where an element disappears
    println!("\n--- Invalid history (element lost) ---\n");

    let mut bad_history = History::new();

    // Add element 1
    bad_history.push(Op::add_invoke(0, Pid(0), Val(1)).at_millis(0));
    bad_history.push(Op::add_ok(1, Pid(0), Val(1)).at_millis(5));

    // First read sees it
    bad_history.push(Op::read_invoke(2, Pid(1)).at_millis(10));
    bad_history.push(Op::read_ok(3, Pid(1), [Val(1)]).at_millis(12));

    // Second read doesn't see it - element was lost!
    bad_history.push(Op::read_invoke(4, Pid(1)).at_millis(20));
    bad_history.push(Op::read_ok(5, Pid(1), []).at_millis(22));

    let bad_result = SetFullChecker::default().check(&bad_history);

    println!("Validity: {:?}", bad_result.valid);
    println!("Lost elements: {:?}", bad_result.lost);

    assert_eq!(bad_result.valid, Validity::Invalid);
    assert_eq!(bad_result.lost, vec![Val(1)]);

    println!("\nDone!");
}
