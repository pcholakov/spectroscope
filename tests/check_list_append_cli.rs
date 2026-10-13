//! Runs the `check-list-append` binary on small fixtures.
#![cfg(feature = "cli")]
#![allow(clippy::expect_used)]

use std::process::Command;

fn run(fixture: &str, extra: &[&str]) -> (i32, String) {
    let path = format!("{}/tests/fixtures/{fixture}", env!("CARGO_MANIFEST_DIR"));
    let output = Command::new(env!("CARGO_BIN_EXE_check-list-append"))
        .args(extra)
        .arg(path)
        .output()
        .expect("binary runs");
    (
        output.status.code().expect("exit code"),
        String::from_utf8(output.stdout).expect("utf-8"),
    )
}

#[test]
fn valid_history_exits_zero() {
    let (code, stdout) = run("valid.jsonl", &[]);
    assert_eq!(code, 0, "{stdout}");
    assert!(stdout.contains("valid: valid"), "{stdout}");
}

#[test]
fn stale_read_is_invalid_under_strict_but_not_plain_serializability() {
    let (code, stdout) = run("invalid.jsonl", &[]);
    assert_eq!(code, 1, "{stdout}");
    assert!(stdout.contains("G-single-item-realtime"), "{stdout}");
    let (code, stdout) = run("invalid.jsonl", &["--model", "serializable"]);
    assert_eq!(code, 0, "{stdout}");
}

#[test]
fn json_output_names_the_cycle() {
    let (code, stdout) = run("invalid.jsonl", &["--json"]);
    assert_eq!(code, 1);
    let report: serde_json::Value = serde_json::from_str(&stdout).expect("json");
    assert_eq!(report["valid"], "invalid");
    assert_eq!(report["cycles"][0]["name"], "G-single-item-realtime");
}

#[test]
fn malformed_history_exits_two() {
    let (code, _) = run("malformed.jsonl", &[]);
    assert_eq!(code, 2);
}

#[test]
fn info_and_fail_ops_are_accepted() {
    let (code, stdout) = run("info_and_fail.jsonl", &[]);
    assert_eq!(code, 0, "{stdout}");
    assert!(stdout.contains("ok=2 fail=1 info=1"), "{stdout}");
}

#[test]
fn history_without_ok_ops_exits_two() {
    for fixture in ["no_ok.jsonl", "empty.jsonl"] {
        let (code, stdout) = run(fixture, &[]);
        assert_eq!(code, 2, "{fixture}: {stdout}");
    }
}
