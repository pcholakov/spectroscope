//! Basic usage of the set-full checker.
//!
//! Run with: cargo run --example basic

use spectroscope::Validity;
use spectroscope::history::{History, HistoryError, Op, Pid};
use spectroscope::set_full::SetFullChecker;

/// A value being added to or read from the set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Val(u32);

fn main() -> Result<(), HistoryError> {
    // A history records operations from concurrent processes, in real-time order.
    // Each operation has an invocation (start) and completion (ok/fail/info).
    let mut history = History::new();

    // Processes 0 and 1 add elements 1 and 2 concurrently
    history.push(Op::add_invoke(0, Pid(0), Val(1)).at_millis(0))?;
    history.push(Op::add_invoke(1, Pid(1), Val(2)).at_millis(2))?;
    history.push(Op::add_ok(2, Pid(0), Val(1)).at_millis(5))?;
    history.push(Op::add_ok(3, Pid(1), Val(2)).at_millis(7))?;

    // Process 2 reads and sees both elements
    history.push(Op::read_invoke(4, Pid(2)).at_millis(10))?;
    history.push(Op::read_ok(5, Pid(2), [Val(1), Val(2)]).at_millis(12))?;

    let result = SetFullChecker::default().check(&history);

    println!("Validity: {:?}", result.valid);
    println!("Elements added: {}", result.attempt_count);
    println!("Stable (visible): {}", result.stable_count);
    println!("Lost: {}", result.lost_count);

    assert_eq!(result.valid, Validity::Valid);

    println!("\n--- Invalid history (element lost) ---\n");

    let mut lost_history = History::new();

    // Add element 1
    lost_history.push(Op::add_invoke(0, Pid(0), Val(1)).at_millis(0))?;
    lost_history.push(Op::add_ok(1, Pid(0), Val(1)).at_millis(5))?;

    // First read sees it
    lost_history.push(Op::read_invoke(2, Pid(1)).at_millis(10))?;
    lost_history.push(Op::read_ok(3, Pid(1), [Val(1)]).at_millis(12))?;

    // Second read doesn't see it - element was lost!
    lost_history.push(Op::read_invoke(4, Pid(1)).at_millis(20))?;
    lost_history.push(Op::read_ok(5, Pid(1), []).at_millis(22))?;

    let lost_result = SetFullChecker::default().check(&lost_history);

    println!("Validity: {:?}", lost_result.valid);
    println!("Lost elements: {:?}", lost_result.lost);

    assert_eq!(lost_result.valid, Validity::Invalid);
    assert_eq!(lost_result.lost, vec![Val(1)]);

    println!("\n--- Non-linearizable history (readers disagree on order) ---\n");

    // Two concurrent reads, each concurrent with both adds, see different elements.
    // Every element is visible without staleness, so only linearizable mode rejects it.
    let divergent = History::from_ops([
        Op::add_invoke(0, Pid(0), Val(1)),
        Op::add_invoke(1, Pid(1), Val(2)),
        Op::read_invoke(2, Pid(2)),
        Op::read_invoke(3, Pid(3)),
        Op::read_ok(4, Pid(2), [Val(1)]),
        Op::read_ok(5, Pid(3), [Val(2)]),
        Op::add_ok(6, Pid(0), Val(1)),
        Op::add_ok(7, Pid(1), Val(2)),
    ])?;

    assert_eq!(
        SetFullChecker::default().check(&divergent).valid,
        Validity::Valid
    );
    let divergent_result = SetFullChecker::linearizable().check(&divergent);

    println!("Validity: {:?}", divergent_result.valid);
    let edges = divergent_result
        .linearizability_violation
        .iter()
        .flat_map(|cycle| &cycle.edges);
    for edge in edges {
        println!(
            "{:?} before {:?} ({:?})",
            edge.before, edge.after, edge.reason
        );
    }

    assert_eq!(divergent_result.valid, Validity::Invalid);

    println!("\nDone!");
    Ok(())
}
