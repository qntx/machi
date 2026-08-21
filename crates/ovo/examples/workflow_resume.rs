//! Workflow resume: second run replays journaled host calls (no extra samples).
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::missing_assert_message,
    unused_crate_dependencies,
    reason = "offline demo binary uses stdout and expect for setup"
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use ovo::{
    InProcessHost, Journal, LlmSampler, MockSampler, OvoError, SampleRequest, SampleResponse,
    SessionHost, WorkflowOutcome, WorkflowRunParams, run_workflow_on_host,
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

struct CountMock {
    inner: MockSampler,
    calls: AtomicUsize,
}

#[async_trait]
impl LlmSampler for CountMock {
    async fn sample(&self, request: SampleRequest) -> Result<SampleResponse, OvoError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.sample(request).await
    }
}

#[tokio::main]
async fn main() -> Result<(), String> {
    let dir = std::env::temp_dir().join(format!("ovo-wf-resume-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let journal_path = dir.join("journal.jsonl");

    let script = r#"
        let meta = #{ name: "resume-demo", description: "two agents" };
        let a = agent("first", #{ label: "a" });
        let b = agent("second", #{ label: "b" });
        complete(#{ a: a, b: b });
    "#;

    let sampler = Arc::new(CountMock {
        inner: MockSampler::new(),
        calls: AtomicUsize::new(0),
    });
    sampler.inner.push_text("out-a");
    sampler.inner.push_text("out-b");

    let host: Arc<dyn SessionHost> = Arc::new(InProcessHost::new(sampler.clone(), Vec::new()));
    let (tx, _rx) = mpsc::unbounded_channel();
    let outcome1 = run_workflow_on_host(
        Arc::clone(&host),
        WorkflowRunParams {
            script: script.into(),
            args: serde_json::json!({}),
            journal: Journal::new(Some(journal_path.clone())),
            host_tx: tx,
            cancel: CancellationToken::new(),
            max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
        },
        Some(16),
    )
    .await
    .map_err(|e| e.to_string())?;
    let calls_after_first = sampler.calls.load(Ordering::SeqCst);
    println!("first_run_sample_calls={calls_after_first}");
    if !matches!(outcome1, WorkflowOutcome::Completed { .. }) {
        return Err(format!("first run failed: {outcome1:?}"));
    }
    if calls_after_first != 2 {
        return Err(format!(
            "first run should sample twice, got {calls_after_first}"
        ));
    }

    let journal = Journal::load(journal_path).map_err(|e| e.to_string())?;
    println!("journal_entries={}", journal.len());
    let (tx2, _rx2) = mpsc::unbounded_channel();
    let outcome2 = run_workflow_on_host(
        host,
        WorkflowRunParams {
            script: script.into(),
            args: serde_json::json!({}),
            journal,
            host_tx: tx2,
            cancel: CancellationToken::new(),
            max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
        },
        Some(16),
    )
    .await
    .map_err(|e| e.to_string())?;
    let calls_after_resume = sampler.calls.load(Ordering::SeqCst);
    println!("resume_run_sample_calls_total={calls_after_resume}");
    println!(
        "resume_new_samples={}",
        calls_after_resume.saturating_sub(calls_after_first)
    );
    match outcome2 {
        WorkflowOutcome::Completed { result } => {
            println!("resume_outcome=completed");
            println!(
                "a_output={}",
                result
                    .get("a")
                    .and_then(|v| v.get("output"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
            );
            println!(
                "b_output={}",
                result
                    .get("b")
                    .and_then(|v| v.get("output"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
            );
        }
        other => return Err(format!("resume failed: {other:?}")),
    }
    if calls_after_resume != calls_after_first {
        return Err("resume must not re-sample".into());
    }
    println!("resume_ok=true");
    Ok(())
}
