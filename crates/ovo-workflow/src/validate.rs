//! Dry-run validation of workflow scripts (meta + stub host path).
//!
//! Extract meta, then run the script against a **probe host** that never calls
//! models. Compile/runtime failures surface as [`ValidationError`].

use std::thread;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::host::{AgentResult, BudgetState, HostError, WorkflowHostRequest};
use crate::journal::Journal;
use crate::meta::{MetaError, extract_meta};
use crate::run::WorkflowOutcome;
use crate::{DEFAULT_AGENT_BUDGET, WorkflowRunParams, run_workflow};

/// Successful dry-run report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    /// Workflow name from meta.
    pub name: String,
    /// Number of declared phases.
    pub phases: usize,
    /// Whether the terminal outcome was considered successful for authoring.
    pub outcome_ok: bool,
    /// Short human summary (truncated).
    pub outcome_summary: String,
}

/// Validation failures.
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    /// Meta extraction failed.
    #[error("meta: {0}")]
    Meta(#[from] MetaError),
    /// Dry-run failed.
    #[error("dry-run: {0}")]
    Run(String),
}

/// Default `args` used when the author does not supply probe input.
#[must_use]
pub fn default_probe_args() -> serde_json::Value {
    serde_json::json!({
        "objective": "stub objective",
        "query": "stub query",
        "breadth": 2,
        "target": "stub target",
        "skeptic_count": 1,
        "max_verify_attempts": 1,
        "baseline_commit": "",
        "test_command": "cargo test",
        "diff_summary": "stub diff",
        "since_commit": "abc123",
    })
}

/// Validate a script with the default agent budget.
///
/// # Errors
///
/// Meta or dry-run failures.
pub fn validate_script(
    script: &str,
    args: Option<serde_json::Value>,
) -> Result<ValidationReport, ValidationError> {
    validate_script_with_agent_budget(script, args, DEFAULT_AGENT_BUDGET)
}

/// Validate with an explicit agent-call budget for the probe host.
///
/// # Errors
///
/// Meta or dry-run failures.
pub fn validate_script_with_agent_budget(
    script: &str,
    args: Option<serde_json::Value>,
    agent_budget: u64,
) -> Result<ValidationReport, ValidationError> {
    let meta = extract_meta(script)?;

    let (host_tx, host_rx) = mpsc::unbounded_channel();
    let host = thread::spawn(move || probe_host_loop(host_rx, agent_budget));

    let outcome = run_workflow(WorkflowRunParams {
        script: script.to_owned(),
        args: args.unwrap_or_else(default_probe_args),
        journal: Journal::new(None),
        host_tx,
        cancel: CancellationToken::new(),
        max_ops: 10_000_000,
    });
    // Dropping the sender ends the probe loop; join for hygiene.
    let _ = host.join();

    let (outcome_ok, outcome_summary) = summarize_outcome(&outcome);
    if !outcome_ok {
        return Err(ValidationError::Run(outcome_summary));
    }

    Ok(ValidationReport {
        name: meta.name,
        phases: meta.phases.len(),
        outcome_ok,
        outcome_summary,
    })
}

fn summarize_outcome(outcome: &WorkflowOutcome) -> (bool, String) {
    match outcome {
        WorkflowOutcome::Completed { result } => (
            true,
            format!("completed: {}", truncate(&result.to_string())),
        ),
        WorkflowOutcome::Paused { kind, message } => (
            true,
            format!("paused ({}): {}", kind.as_str(), truncate(message)),
        ),
        WorkflowOutcome::Failed { error } => (false, format!("failed: {error}")),
        WorkflowOutcome::BudgetExceeded { message } => {
            (false, format!("budget: {}", truncate(message)))
        }
        WorkflowOutcome::Cancelled => (false, "cancelled".into()),
    }
}

fn truncate(s: &str) -> String {
    const MAX: usize = 200;
    if s.chars().count() > MAX {
        let head: String = s.chars().take(MAX).collect();
        format!("{head}…")
    } else {
        s.to_owned()
    }
}

/// Blocking probe host: answers every host RPC without side effects or models.
fn probe_host_loop(mut rx: mpsc::UnboundedReceiver<WorkflowHostRequest>, agent_budget: u64) {
    let mut agent_calls = 0u64;
    while let Some(req) = rx.blocking_recv() {
        match req {
            WorkflowHostRequest::ReserveAgentCalls { count, reply } => {
                let requested = agent_calls.saturating_add(count);
                if requested > agent_budget {
                    let _ = reply.send(Err(HostError::AgentCallQuotaExceeded {
                        requested,
                        maximum: agent_budget,
                    }));
                } else {
                    agent_calls = requested;
                    let _ = reply.send(Ok(()));
                }
            }
            WorkflowHostRequest::ReleaseAgentCalls { count, reply } => {
                agent_calls = agent_calls.saturating_sub(count);
                let _ = reply.send(Ok(()));
            }
            WorkflowHostRequest::SpawnAgent { reply, .. } => {
                let _ = reply.send(Ok(AgentResult {
                    agent_id: "probe".into(),
                    success: true,
                    output: serde_json::json!({
                        "stub": true,
                        "achieved": true,
                        "text": "probe agent output",
                    }),
                    cancelled: false,
                    tokens_used: 1,
                    duration_ms: 1,
                }));
            }
            WorkflowHostRequest::BudgetQuery { reply } => {
                let _ = reply.send(Ok(BudgetState {
                    total: Some(agent_budget),
                    spent: agent_calls,
                    reserved: 0,
                    remaining: Some(agent_budget.saturating_sub(agent_calls)),
                }));
            }
            WorkflowHostRequest::RenderTemplate { name, reply, .. } => {
                let _ = reply.send(Ok(format!("probe-template:{name}")));
            }
            WorkflowHostRequest::WriteScratchFile { name, reply, .. } => {
                let _ = reply.send(Ok(format!("scratch/{name}")));
            }
            WorkflowHostRequest::ReadScratchFile { name, reply, .. } => {
                let _ = reply.send(Ok(format!("probe-content:{name}")));
            }
            WorkflowHostRequest::GitDiffSince { reply, .. } => {
                let _ = reply.send(Ok(String::new()));
            }
            WorkflowHostRequest::Phase { .. }
            | WorkflowHostRequest::Log { .. }
            | WorkflowHostRequest::Telemetry { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_script_passes() {
        let report = validate_script(
            r#"
            let meta = #{ name: "t", description: "d" };
            let r = agent("work");
            complete(r.output);
            "#,
            None,
        )
        .expect("validate");
        assert_eq!(report.name, "t");
        assert!(report.outcome_ok);
    }

    #[test]
    fn missing_meta_fails() {
        let err = validate_script("let x = 1;", None).expect_err("meta");
        assert!(matches!(err, ValidationError::Meta(_)));
    }

    #[test]
    fn probe_args_nonempty() {
        let args = default_probe_args();
        assert!(
            !args
                .get("objective")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .is_empty()
        );
        assert!(
            args.get("breadth")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                >= 2
        );
    }

    #[test]
    fn parallel_probe_path() {
        let report = validate_script(
            r#"
            let meta = #{ name: "p", description: "parallel probe" };
            let rs = parallel([
                #{ prompt: "a", label: "a" },
                #{ prompt: "b", label: "b" },
            ]);
            complete(#{ n: rs.len() });
            "#,
            None,
        )
        .expect("parallel validate");
        assert!(report.outcome_ok);
    }
}
