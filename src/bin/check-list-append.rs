//! Checks a JSONL list-append history against a consistency model.
//!
//! ```text
//! check-list-append [--model serializable|strong-session-serializable|strict-serializable]
//!                   [--json] <history.jsonl | ->
//! ```
//!
//! Each input line is one op:
//! `{"index": 0, "process": 0, "type": "invoke", "time_ms": 0, "ops": [["append", "x", 1], ["r", "y", null]]}`.
//! `type` is `invoke`, `ok`, `fail` or `info`; a read is `["r", key, null]` in an
//! invocation and `["r", key, [values...]]` in an `ok` completion.
//!
//! Exit code 0: valid. 1: invalid. 2: unknown, unreadable or malformed input, or a history with no
//! `ok` operation (a checker that saw nothing proves nothing).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::Read;
use std::process::ExitCode;

use serde::Deserialize;
use serde_json::{Value, json};
use spectroscope::{
    Anomaly, ConsistencyModel, Cycle, Dependency, History, ListAppendChecker, MicroOp, Op, OpType,
    Pid, Txn, TxnRef, Validity,
};

type Key = String;
type Value64 = u64;
type Transaction = Txn<Key, Value64>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Line {
    index: usize,
    process: u64,
    #[serde(rename = "type")]
    kind: String,
    time_ms: u64,
    ops: Vec<(String, Key, Value)>,
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::from(2)
        }
    }
}

struct Args {
    model: ConsistencyModel,
    json: bool,
    path: String,
}

fn parse_args() -> Result<Args, String> {
    let mut model = ConsistencyModel::StrictSerializable;
    let mut json = false;
    let mut path = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => json = true,
            "--model" => {
                let name = args.next().ok_or("--model needs a value")?;
                model = match name.as_str() {
                    "serializable" => ConsistencyModel::Serializable,
                    "strong-session-serializable" => ConsistencyModel::StrongSessionSerializable,
                    "strict-serializable" => ConsistencyModel::StrictSerializable,
                    other => return Err(format!("unknown model {other:?}")),
                };
            }
            "-h" | "--help" => {
                return Err("usage: check-list-append [--model serializable|strong-session-serializable|strict-serializable] [--json] <history.jsonl | ->".into());
            }
            flag if flag.starts_with("--") => return Err(format!("unknown flag {flag}")),
            _ if path.is_some() => return Err("more than one history file".into()),
            _ => path = Some(arg),
        }
    }
    Ok(Args {
        model,
        json,
        path: path.ok_or("missing history file (use - for stdin)")?,
    })
}

fn run() -> Result<ExitCode, String> {
    let args = parse_args()?;
    let input = if args.path == "-" {
        let mut text = String::new();
        std::io::stdin()
            .read_to_string(&mut text)
            .map_err(|e| format!("reading stdin: {e}"))?;
        text
    } else {
        std::fs::read_to_string(&args.path).map_err(|e| format!("reading {}: {e}", args.path))?
    };
    let history = parse_history(&input)?;
    let result = ListAppendChecker::new(args.model).check(&history);
    let report = Report::new(&history, &result);
    let has_ok_ops = history.ops().iter().any(|op| op.op_type == OpType::Ok);
    if args.json {
        println!("{:#}", report.to_json(args.model));
    } else {
        report.print(args.model);
    }
    if !has_ok_ops {
        eprintln!("error: the history has no ok operation, so it proves nothing");
        return Ok(ExitCode::from(2));
    }
    Ok(match result.valid {
        Validity::Valid => ExitCode::SUCCESS,
        Validity::Invalid => ExitCode::from(1),
        Validity::Unknown => ExitCode::from(2),
    })
}

fn parse_history(input: &str) -> Result<History<Transaction>, String> {
    let mut history = History::new();
    for (number, text) in input.lines().enumerate() {
        if text.trim().is_empty() {
            continue;
        }
        let at = |message: String| format!("line {}: {message}", number + 1);
        let line: Line = serde_json::from_str(text).map_err(|e| at(e.to_string()))?;
        let op_type = match line.kind.as_str() {
            "invoke" => OpType::Invoke,
            "ok" => OpType::Ok,
            "fail" => OpType::Fail,
            "info" => OpType::Info,
            other => return Err(at(format!("unknown type {other:?}"))),
        };
        let micro_ops = line
            .ops
            .into_iter()
            .map(|(function, key, argument)| micro_op(&function, key, &argument))
            .collect::<Result<Vec<_>, _>>()
            .map_err(at)?;
        let op = Op::new(line.index, Pid(line.process), op_type, Txn(micro_ops))
            .at_millis(line.time_ms);
        history.push(op).map_err(|e| at(e.to_string()))?;
    }
    Ok(history)
}

fn micro_op(function: &str, key: Key, argument: &Value) -> Result<MicroOp<Key, Value64>, String> {
    match function {
        "append" => argument
            .as_u64()
            .map(|value| MicroOp::append(key, value))
            .ok_or_else(|| format!("append value must be a u64, got {argument}")),
        "r" => match argument {
            Value::Null => Ok(MicroOp::read(key)),
            Value::Array(items) => items
                .iter()
                .map(|item| {
                    item.as_u64()
                        .ok_or_else(|| format!("read value must be a u64, got {item}"))
                })
                .collect::<Result<Vec<_>, _>>()
                .map(|list| MicroOp::read_result(key, list)),
            other => Err(format!("read result must be null or a list, got {other}")),
        },
        other => Err(format!("unknown micro-op {other:?}")),
    }
}

struct Report<'a> {
    history: &'a History<Transaction>,
    result: &'a spectroscope::ListAppendResult<Key, Value64>,
    position_of_index: HashMap<usize, usize>,
}

impl<'a> Report<'a> {
    fn new(
        history: &'a History<Transaction>,
        result: &'a spectroscope::ListAppendResult<Key, Value64>,
    ) -> Self {
        let position_of_index = history
            .ops()
            .iter()
            .enumerate()
            .map(|(position, op)| (op.index, position))
            .collect();
        Self {
            history,
            result,
            position_of_index,
        }
    }

    fn op(&self, index: usize) -> &Op<Transaction> {
        &self.history.ops()[self.position_of_index[&index]]
    }

    /// The transaction as its completion shows it, or its invocation while pending.
    fn describe(&self, txn: TxnRef) -> String {
        let op = self.op(txn.complete_index.unwrap_or(txn.invoke_index));
        let micro_ops: Vec<String> = op.value.0.iter().map(format_micro_op).collect();
        let outcome = match txn.complete_index {
            None => "pending".to_string(),
            Some(_) => format!("{:?}", op.op_type).to_lowercase(),
        };
        format!(
            "T{} (process {}, {outcome}, op {}..{}) [{}]",
            txn.invoke_index,
            op.process.0,
            txn.invoke_index,
            txn.complete_index
                .map_or_else(|| "-".to_string(), |i| i.to_string()),
            micro_ops.join(", ")
        )
    }

    fn counts_by_type(&self) -> [(&'static str, usize); 4] {
        let count = |wanted| {
            self.history
                .ops()
                .iter()
                .filter(|op| op.op_type == wanted)
                .count()
        };
        [
            ("invoke", count(OpType::Invoke)),
            ("ok", count(OpType::Ok)),
            ("fail", count(OpType::Fail)),
            ("info", count(OpType::Info)),
        ]
    }

    fn anomaly_counts(&self) -> Vec<(&'static str, usize)> {
        let mut counts: Vec<(&'static str, usize)> = Vec::new();
        let names = self
            .result
            .anomalies
            .iter()
            .map(Anomaly::name)
            .chain(self.result.cycles.iter().map(|c| c.anomaly.name()));
        for name in names {
            match counts.iter_mut().find(|(existing, _)| *existing == name) {
                Some((_, count)) => *count += 1,
                None => counts.push((name, 1)),
            }
        }
        counts
    }

    fn print(&self, model: ConsistencyModel) {
        println!("model: {}", model_name(model));
        println!("valid: {}", validity_name(self.result.valid));
        let counts: Vec<String> = self
            .counts_by_type()
            .iter()
            .map(|(name, count)| format!("{name}={count}"))
            .collect();
        println!("ops: {}", counts.join(" "));
        let anomalies = self.anomaly_counts();
        if anomalies.is_empty() {
            println!("anomalies: none");
        } else {
            let listed: Vec<String> = anomalies
                .iter()
                .map(|(name, count)| format!("{name} x{count}"))
                .collect();
            println!("anomalies: {}", listed.join(", "));
        }
        for anomaly in &self.result.anomalies {
            println!("\n{}: {}", anomaly.name(), self.describe_anomaly(anomaly));
        }
        for (number, cycle) in self.result.cycles.iter().enumerate() {
            println!("\ncycle {} ({}):", number + 1, cycle.anomaly.name());
            for line in self.cycle_steps(cycle) {
                println!("  {line}");
            }
        }
    }

    fn describe_anomaly(&self, anomaly: &Anomaly<Key, Value64>) -> String {
        // The Debug form names every field; transactions are shown by index, so
        // list the ones involved in full.
        let mut text = format!("{anomaly:?}");
        let mut seen: Vec<usize> = Vec::new();
        for txn in anomaly_txns(anomaly) {
            if !seen.contains(&txn.invoke_index) {
                seen.push(txn.invoke_index);
                let _ = write!(text, "\n  {}", self.describe(txn));
            }
        }
        text
    }

    fn cycle_steps(&self, cycle: &Cycle<Key, Value64>) -> Vec<String> {
        cycle
            .steps
            .iter()
            .map(|step| {
                format!(
                    "{}\n    --{}--> {}",
                    self.describe(step.from),
                    explain(&step.dependency),
                    self.describe(step.to)
                )
            })
            .collect()
    }

    fn to_json(&self, model: ConsistencyModel) -> Value {
        let ops: serde_json::Map<String, Value> = self
            .counts_by_type()
            .iter()
            .map(|(name, count)| ((*name).to_string(), json!(count)))
            .collect();
        let anomalies: Vec<Value> = self
            .result
            .anomalies
            .iter()
            .map(|anomaly| json!({"name": anomaly.name(), "detail": self.describe_anomaly(anomaly)}))
            .collect();
        let cycles: Vec<Value> = self
            .result
            .cycles
            .iter()
            .map(|cycle| {
                let steps: Vec<Value> = cycle
                    .steps
                    .iter()
                    .map(|step| {
                        json!({
                            "from": self.describe(step.from),
                            "to": self.describe(step.to),
                            "dependency": explain(&step.dependency),
                        })
                    })
                    .collect();
                json!({"name": cycle.anomaly.name(), "steps": steps})
            })
            .collect();
        let counts: serde_json::Map<String, Value> = self
            .anomaly_counts()
            .into_iter()
            .map(|(name, count)| (name.to_string(), json!(count)))
            .collect();
        json!({
            "model": model_name(model),
            "valid": validity_name(self.result.valid),
            "ops": ops,
            "anomaly_counts": counts,
            "anomalies": anomalies,
            "cycles": cycles,
        })
    }
}

fn anomaly_txns(anomaly: &Anomaly<Key, Value64>) -> Vec<TxnRef> {
    match anomaly {
        Anomaly::DuplicateAppends { first, second, .. } => vec![*first, *second],
        Anomaly::DuplicateElements { reader, .. }
        | Anomaly::G1b { reader, .. }
        | Anomaly::GarbageRead { reader, .. } => vec![*reader],
        Anomaly::IncompatibleOrder {
            shorter, longer, ..
        } => vec![shorter.1, longer.1],
        Anomaly::Internal { txn, .. }
        | Anomaly::FutureRead { txn, .. }
        | Anomaly::AppendOrder { txn, .. } => vec![*txn],
        Anomaly::G1a { reader, writer, .. } => vec![*reader, *writer],
        Anomaly::DirtyUpdate {
            aborted, committed, ..
        } => vec![*aborted, *committed],
        Anomaly::LostUpdate { txns, .. } => txns.clone(),
    }
}

fn format_micro_op(micro_op: &MicroOp<Key, Value64>) -> String {
    match micro_op {
        MicroOp::Append { key, value } => format!("append {key} {value}"),
        MicroOp::Read { key, list: None } => format!("r {key}"),
        MicroOp::Read {
            key,
            list: Some(list),
        } => format!("r {key} -> {list:?}"),
    }
}

fn explain(dependency: &Dependency<Key, Value64>) -> String {
    match dependency {
        Dependency::WriteWrite {
            key,
            previous,
            appended,
        } => format!("ww: appended {appended} to {key} after {previous}"),
        Dependency::WriteRead { key, value } => {
            format!("wr: read {key} ending in the value {value} it appended")
        }
        Dependency::ReadWrite {
            key,
            read,
            appended,
        } => {
            let seen = read.map_or_else(|| "the empty list".to_string(), |v| format!("..{v}"));
            format!("rw: read {key} as {seen}, without the later append of {appended}")
        }
        Dependency::Process => "process: same process, earlier".to_string(),
        Dependency::RealTime {
            completed_index,
            invoked_index,
        } => format!(
            "realtime: completed at op {completed_index}, before the next was invoked at op {invoked_index}"
        ),
    }
}

fn model_name(model: ConsistencyModel) -> &'static str {
    match model {
        ConsistencyModel::Serializable => "serializable",
        ConsistencyModel::StrongSessionSerializable => "strong-session-serializable",
        ConsistencyModel::StrictSerializable => "strict-serializable",
    }
}

fn validity_name(validity: Validity) -> &'static str {
    match validity {
        Validity::Valid => "valid",
        Validity::Invalid => "invalid",
        Validity::Unknown => "unknown",
    }
}
