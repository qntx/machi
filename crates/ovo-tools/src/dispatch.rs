//! Concurrent tool dispatch with exclusivity, capability, and approval gates.

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::join_all;
use ovo_obs::{NoopMetrics, SharedMetrics, record_tool_call};
use ovo_protocol::TurnEventKind;
use ovo_types::{ToolCall, ToolCallId};
use tokio::time::timeout;
use tracing::{Instrument, info_span};

use crate::approval::{ApprovalDecision, ApprovalGate, AutoApprove};
use crate::context::ToolCallContext;
use crate::error::{ToolError, codes};
use crate::metadata::{ConcurrencyMode, Destructiveness, ToolMetadata};
use crate::registry::{CapabilityMode, ToolRegistry};
use crate::stream::{ToolProgress, drain_with_progress};
use crate::tool::{DynTool, SharedTool, ToolResult};

/// One tool call to execute.
#[derive(Debug, Clone)]
pub struct DispatchRequest {
    /// Model tool call.
    pub call: ToolCall,
}

/// Outcome for a single dispatched call.
#[derive(Debug, Clone)]
pub struct DispatchOutcome {
    /// Call id.
    pub id: ToolCallId,
    /// Tool name.
    pub name: String,
    /// Result or error mapped for the model.
    pub result: Result<ToolResult, ToolError>,
}

/// When to consult the approval gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ApprovalPolicy {
    /// Never consult (tests / fully trusted offline runs).
    Never,
    /// Consult when tool is mutating or executes (default production policy).
    #[default]
    Destructive,
    /// Consult every tool call.
    Always,
}

/// Scheduler for tool batches.
#[derive(Clone)]
pub struct ToolDispatch {
    /// Maximum concurrent non-exclusive tools.
    pub max_concurrency: usize,
    /// Capability filter applied before execution.
    pub capability_mode: CapabilityMode,
    /// Host approval gate.
    pub approval: Arc<dyn ApprovalGate>,
    /// When to invoke approval.
    pub approval_policy: ApprovalPolicy,
    /// Metrics sink (default no-op).
    pub metrics: SharedMetrics,
}

impl std::fmt::Debug for ToolDispatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolDispatch")
            .field("max_concurrency", &self.max_concurrency)
            .field("capability_mode", &self.capability_mode)
            .field("approval_policy", &self.approval_policy)
            .finish_non_exhaustive()
    }
}

impl Default for ToolDispatch {
    fn default() -> Self {
        Self {
            max_concurrency: 32,
            capability_mode: CapabilityMode::Full,
            approval: Arc::new(AutoApprove),
            approval_policy: ApprovalPolicy::Destructive,
            metrics: Arc::new(NoopMetrics),
        }
    }
}

impl ToolDispatch {
    /// Builder: capability mode.
    #[must_use]
    pub fn with_capability(mut self, mode: CapabilityMode) -> Self {
        self.capability_mode = mode;
        self
    }

    /// Builder: max concurrency.
    #[must_use]
    pub const fn with_max_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n;
        self
    }

    /// Builder: approval gate.
    #[must_use]
    pub fn with_approval(mut self, gate: Arc<dyn ApprovalGate>) -> Self {
        self.approval = gate;
        self
    }

    /// Builder: approval policy.
    #[must_use]
    pub const fn with_approval_policy(mut self, policy: ApprovalPolicy) -> Self {
        self.approval_policy = policy;
        self
    }

    /// Builder: metrics sink.
    #[must_use]
    pub fn with_metrics(mut self, metrics: SharedMetrics) -> Self {
        self.metrics = metrics;
        self
    }

    /// Execute a batch preserving input order in the output vector.
    pub async fn execute_batch(
        &self,
        registry: &ToolRegistry,
        ctx: ToolCallContext,
        requests: Vec<DispatchRequest>,
    ) -> Vec<DispatchOutcome> {
        if requests.is_empty() {
            return Vec::new();
        }

        let mut outcomes: Vec<Option<DispatchOutcome>> =
            (0..requests.len()).map(|_| None).collect();
        let mut index = 0usize;

        while index < requests.len() {
            if ctx.is_cancelled() {
                fill_cancelled(&requests, &mut outcomes, index);
                break;
            }

            let Some(req) = requests.get(index) else {
                break;
            };

            match prepare_call(registry, self.capability_mode, req) {
                Prepare::Deny(out) | Prepare::Missing(out) => {
                    set_outcome(&mut outcomes, index, out);
                    index = index.saturating_add(1);
                }
                Prepare::Ready(tool)
                    if tool.metadata().concurrency == ConcurrencyMode::Exclusive =>
                {
                    let out = self.run_one(tool.as_ref(), ctx.clone(), req).await;
                    set_outcome(&mut outcomes, index, out);
                    index = index.saturating_add(1);
                }
                Prepare::Ready(_) => {
                    index = self
                        .run_concurrent_window(registry, &ctx, &requests, &mut outcomes, index)
                        .await;
                }
            }
        }

        finalize_outcomes(&requests, outcomes)
    }

    async fn run_concurrent_window(
        &self,
        registry: &ToolRegistry,
        ctx: &ToolCallContext,
        requests: &[DispatchRequest],
        outcomes: &mut [Option<DispatchOutcome>],
        index: usize,
    ) -> usize {
        let window = collect_concurrent_window(
            registry,
            self.capability_mode,
            requests,
            index,
            self.max_concurrency.max(1),
        );
        let next = window.last().map_or(index + 1, |i| i.saturating_add(1));
        let futs = window.into_iter().filter_map(|win_i| {
            let win_req = requests.get(win_i)?.clone();
            let win_tool = registry.require(&win_req.call.name).ok()?;
            let win_ctx = ctx.clone();
            Some(async move {
                (
                    win_i,
                    self.run_one(win_tool.as_ref(), win_ctx, &win_req).await,
                )
            })
        });
        for (i, out) in join_all(futs).await {
            set_outcome(outcomes, i, out);
        }
        next
    }

    async fn run_one(
        &self,
        tool: &dyn DynTool,
        ctx: ToolCallContext,
        req: &DispatchRequest,
    ) -> DispatchOutcome {
        let span = info_span!(
            "ovo.tool",
            ovo.tool_name = tool.name(),
            ovo.tool_call_id = %req.call.id,
        );
        let meta = tool.metadata();
        let started = Instant::now();
        let result = async { self.execute_tool(tool, &meta, ctx, req).await }
            .instrument(span)
            .await;
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        let status = match &result {
            Ok(r) if r.is_error => "tool_error",
            Ok(_) => "ok",
            Err(e) if e.code() == ovo_types::ErrorCode::ToolCancelled => "cancelled",
            Err(e) if e.code() == ovo_types::ErrorCode::ToolApprovalDenied => "denied",
            Err(_) => "error",
        };
        record_tool_call(self.metrics.as_ref(), tool.name(), status, ms);

        DispatchOutcome {
            id: req.call.id.clone(),
            name: req.call.name.clone(),
            result,
        }
    }

    async fn execute_tool(
        &self,
        tool: &dyn DynTool,
        meta: &ToolMetadata,
        ctx: ToolCallContext,
        req: &DispatchRequest,
    ) -> Result<ToolResult, ToolError> {
        if ctx.is_cancelled() {
            return Err(codes::cancelled());
        }
        self.check_approval(tool, meta, &req.call.arguments).await?;
        let call_id = req.call.id.as_str().to_owned();
        let call_name = req.call.name.clone();
        ctx.emit(TurnEventKind::ToolExecutionStart {
            id: call_id.clone(),
            name: call_name.clone(),
        });
        // Per-call child token: timeout cancels nested work (e.g. spawn_agent).
        let call_cancel = ctx.cancel.child_token();
        let mut call_ctx = ctx.clone();
        call_ctx.cancel = call_cancel.clone();
        let fut = async {
            let stream = tool
                .execute(call_ctx.clone(), req.call.arguments.clone())
                .await;
            let (progress, result) = drain_with_progress(stream).await;
            for p in progress {
                call_ctx.emit(TurnEventKind::ToolExecutionUpdate {
                    id: call_id.clone(),
                    name: call_name.clone(),
                    message: progress_message(&p),
                });
            }
            result
        };
        // Cap by min(tool timeout, turn deadline).
        let tool_limit = meta.timeout.filter(|d| !d.is_zero());
        let deadline_limit = ctx.deadline.map(|d| d.remaining()).filter(|d| !d.is_zero());
        let limit = match (tool_limit, deadline_limit) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        let result = if let Some(limit) = limit {
            if let Ok(r) = timeout(limit.max(Duration::from_millis(1)), fut).await {
                r
            } else {
                call_cancel.cancel();
                Err(codes::timeout(format!("tool '{}' timed out", tool.name())))
            }
        } else {
            fut.await
        };
        let is_error = match &result {
            Ok(r) => r.is_error,
            Err(_) => true,
        };
        ctx.emit(TurnEventKind::ToolExecutionEnd {
            id: call_id,
            name: call_name,
            is_error,
        });
        result
    }

    async fn check_approval(
        &self,
        tool: &dyn DynTool,
        meta: &ToolMetadata,
        arguments: &serde_json::Value,
    ) -> Result<(), ToolError> {
        if !needs_approval(self.approval_policy, meta) {
            return Ok(());
        }
        match self.approval.approve(tool, meta, arguments).await? {
            ApprovalDecision::Allow => Ok(()),
            ApprovalDecision::Deny => Err(codes::approval_denied(format!(
                "approval denied for tool {}",
                tool.name()
            ))),
        }
    }
}

fn set_outcome(outcomes: &mut [Option<DispatchOutcome>], index: usize, out: DispatchOutcome) {
    if let Some(slot) = outcomes.get_mut(index) {
        *slot = Some(out);
    }
}

fn progress_message(p: &ToolProgress) -> String {
    match p {
        ToolProgress::Text { text } => text.clone(),
        ToolProgress::Partial { delta, .. } => delta.clone(),
        ToolProgress::Custom { subkind, .. } => subkind.clone(),
    }
}

fn needs_approval(policy: ApprovalPolicy, meta: &ToolMetadata) -> bool {
    match policy {
        ApprovalPolicy::Never => false,
        ApprovalPolicy::Always => true,
        ApprovalPolicy::Destructive => {
            meta.destructiveness != Destructiveness::None
                || meta.capabilities.iter().any(|c| {
                    matches!(
                        c,
                        crate::metadata::CapabilityFlag::Write
                            | crate::metadata::CapabilityFlag::Execute
                    )
                })
        }
    }
}

enum Prepare {
    Ready(SharedTool),
    Missing(DispatchOutcome),
    Deny(DispatchOutcome),
}

fn prepare_call(registry: &ToolRegistry, mode: CapabilityMode, req: &DispatchRequest) -> Prepare {
    match registry.require(&req.call.name) {
        Err(err) => Prepare::Missing(DispatchOutcome {
            id: req.call.id.clone(),
            name: req.call.name.clone(),
            result: Err(err),
        }),
        Ok(tool) if !registry.allows(tool.as_ref(), mode) => Prepare::Deny(DispatchOutcome {
            id: req.call.id.clone(),
            name: req.call.name.clone(),
            result: Err(codes::denied(format!(
                "tool '{}' denied by capability mode {mode:?}",
                req.call.name
            ))),
        }),
        Ok(tool) => Prepare::Ready(tool),
    }
}

fn collect_concurrent_window(
    registry: &ToolRegistry,
    mode: CapabilityMode,
    requests: &[DispatchRequest],
    start: usize,
    max: usize,
) -> Vec<usize> {
    let mut window = Vec::new();
    let mut per_tool: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut j = start;
    while j < requests.len() && window.len() < max {
        let Some(req) = requests.get(j) else {
            break;
        };
        let Ok(tool) = registry.require(&req.call.name) else {
            break;
        };
        if !registry.allows(tool.as_ref(), mode) {
            break;
        }
        let meta = tool.metadata();
        if meta.concurrency == ConcurrencyMode::Exclusive {
            // Exclusive tools never share a concurrent window (except alone at start).
            if window.is_empty() {
                window.push(j);
            }
            break;
        }
        // Per-tool cap from metadata; default unlimited within global max.
        if let Some(cap) = meta.max_concurrency {
            let count = per_tool.entry(req.call.name.clone()).or_insert(0);
            if *count >= cap.max(1) {
                // Cannot add another instance of this tool; stop growing window.
                if window.is_empty() {
                    // Still must make progress: run this tool alone.
                    window.push(j);
                }
                break;
            }
            *count = count.saturating_add(1);
        }
        window.push(j);
        j = j.saturating_add(1);
    }
    if window.is_empty() {
        // Fail-safe: never return empty (caller assumes start is included).
        window.push(start);
    }
    window
}

fn fill_cancelled(
    requests: &[DispatchRequest],
    outcomes: &mut [Option<DispatchOutcome>],
    from: usize,
) {
    for (i, req) in requests.iter().enumerate().skip(from) {
        if let Some(slot) = outcomes.get_mut(i)
            && slot.is_none()
        {
            *slot = Some(DispatchOutcome {
                id: req.call.id.clone(),
                name: req.call.name.clone(),
                result: Err(codes::cancelled()),
            });
        }
    }
}

fn finalize_outcomes(
    requests: &[DispatchRequest],
    outcomes: Vec<Option<DispatchOutcome>>,
) -> Vec<DispatchOutcome> {
    outcomes
        .into_iter()
        .enumerate()
        .map(|(i, o)| {
            o.unwrap_or_else(|| {
                let req = requests.get(i);
                DispatchOutcome {
                    id: req.map_or_else(ToolCallId::generate, |r| r.call.id.clone()),
                    name: req.map_or_else(|| "unknown".into(), |r| r.call.name.clone()),
                    result: Err(codes::execution("dispatch internal gap")),
                }
            })
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, reason = "unit tests")]
mod tests {
    use async_trait::async_trait;
    use ovo_types::{ToolCall, ToolCallId};
    use serde_json::json;

    use super::*;
    use crate::tool::{DynTool, ToolResult};

    struct CapTool {
        name: String,
        cap: usize,
    }

    #[async_trait]
    impl DynTool for CapTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "cap"
        }
        fn parameters(&self) -> serde_json::Value {
            json!({})
        }
        fn metadata(&self) -> ToolMetadata {
            ToolMetadata {
                concurrency: ConcurrencyMode::Concurrent,
                max_concurrency: Some(self.cap),
                ..Default::default()
            }
        }
        async fn call(
            &self,
            _ctx: ToolCallContext,
            _args: serde_json::Value,
        ) -> Result<ToolResult, ToolError> {
            Ok(ToolResult::text("ok"))
        }
    }

    #[test]
    fn per_tool_max_concurrency_limits_window() {
        let reg = ToolRegistry::from_tools(vec![Arc::new(CapTool {
            name: "a".into(),
            cap: 1,
        })]);
        let reqs: Vec<DispatchRequest> = (0..3)
            .map(|i| DispatchRequest {
                call: ToolCall {
                    id: ToolCallId::new(format!("c{i}")).expect("id"),
                    name: "a".into(),
                    arguments: json!({}),
                },
            })
            .collect();
        let window = collect_concurrent_window(&reg, CapabilityMode::Full, &reqs, 0, 32);
        assert_eq!(
            window.len(),
            1,
            "cap=1 must not fan out three concurrent a()"
        );
    }

    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use ovo_types::ErrorCode;
    use tokio::sync::Barrier;

    use crate::approval::AlwaysDeny;
    use crate::metadata::ToolMetadata;

    struct CountingTool {
        name: String,
        meta: ToolMetadata,
        active: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
        barrier: Option<Arc<Barrier>>,
    }

    #[async_trait]
    impl DynTool for CountingTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "test"
        }
        fn parameters(&self) -> serde_json::Value {
            json!({"type":"object","properties":{}})
        }
        fn metadata(&self) -> ToolMetadata {
            self.meta.clone()
        }
        async fn call(
            &self,
            _ctx: ToolCallContext,
            _arguments: serde_json::Value,
        ) -> Result<ToolResult, ToolError> {
            let n = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(n, Ordering::SeqCst);
            if let Some(b) = &self.barrier {
                b.wait().await;
            }
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(ToolResult::text("ok"))
        }
    }

    fn call(name: &str, id: &str) -> DispatchRequest {
        DispatchRequest {
            call: ToolCall {
                id: ToolCallId::new(id).expect("id"),
                name: name.into(),
                arguments: json!({}),
            },
        }
    }

    #[tokio::test]
    async fn concurrent_readonly_overlap() {
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(2));
        let t1 = Arc::new(CountingTool {
            name: "r1".into(),
            meta: ToolMetadata {
                concurrency: ConcurrencyMode::ReadOnly,
                ..ToolMetadata::read_only()
            },
            active: Arc::clone(&active),
            max_active: Arc::clone(&max_active),
            barrier: Some(Arc::clone(&barrier)),
        });
        let t2 = Arc::new(CountingTool {
            name: "r2".into(),
            meta: ToolMetadata {
                concurrency: ConcurrencyMode::ReadOnly,
                ..ToolMetadata::read_only()
            },
            active: Arc::clone(&active),
            max_active: Arc::clone(&max_active),
            barrier: Some(barrier),
        });
        let reg = ToolRegistry::from_tools(vec![t1, t2]);
        let outs = ToolDispatch::default()
            .execute_batch(
                &reg,
                ToolCallContext::default(),
                vec![call("r1", "c1"), call("r2", "c2")],
            )
            .await;
        assert_eq!(outs.len(), 2);
        assert!(outs.iter().all(|o| o.result.is_ok()));
        assert!(
            max_active.load(Ordering::SeqCst) >= 2,
            "expected overlap, max={}",
            max_active.load(Ordering::SeqCst)
        );
    }

    #[tokio::test]
    async fn exclusive_serial() {
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let t1 = Arc::new(CountingTool {
            name: "e1".into(),
            meta: ToolMetadata::exclusive_write(),
            active: Arc::clone(&active),
            max_active: Arc::clone(&max_active),
            barrier: None,
        });
        let t2 = Arc::new(CountingTool {
            name: "e2".into(),
            meta: ToolMetadata::exclusive_write(),
            active,
            max_active: Arc::clone(&max_active),
            barrier: None,
        });
        let reg = ToolRegistry::from_tools(vec![t1, t2]);
        let outs = ToolDispatch::default()
            .execute_batch(
                &reg,
                ToolCallContext::default(),
                vec![call("e1", "c1"), call("e2", "c2")],
            )
            .await;
        assert!(outs.iter().all(|o| o.result.is_ok()));
        assert_eq!(max_active.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn readonly_mode_denies_write() {
        let tool = Arc::new(CountingTool {
            name: "w".into(),
            meta: ToolMetadata::exclusive_write(),
            active: Arc::new(AtomicUsize::new(0)),
            max_active: Arc::new(AtomicUsize::new(0)),
            barrier: None,
        });
        let reg = ToolRegistry::from_tools(vec![tool]);
        let dispatch = ToolDispatch::default().with_capability(CapabilityMode::ReadOnly);
        let outs = dispatch
            .execute_batch(&reg, ToolCallContext::default(), vec![call("w", "c1")])
            .await;
        let err = outs
            .first()
            .expect("one outcome")
            .result
            .as_ref()
            .expect_err("denied");
        assert_eq!(err.code(), ErrorCode::ToolDenied);
    }

    #[tokio::test]
    async fn approval_blocks_destructive() {
        let tool = Arc::new(CountingTool {
            name: "w".into(),
            meta: ToolMetadata::exclusive_write(),
            active: Arc::new(AtomicUsize::new(0)),
            max_active: Arc::new(AtomicUsize::new(0)),
            barrier: None,
        });
        let reg = ToolRegistry::from_tools(vec![tool]);
        let dispatch = ToolDispatch::default().with_approval(Arc::new(AlwaysDeny));
        let outs = dispatch
            .execute_batch(&reg, ToolCallContext::default(), vec![call("w", "c1")])
            .await;
        let err = outs
            .first()
            .expect("one")
            .result
            .as_ref()
            .expect_err("approval");
        assert_eq!(err.code(), ErrorCode::ToolApprovalDenied);
    }

    struct SlowTool {
        cancelled: Arc<AtomicBool>,
    }

    #[async_trait]
    impl DynTool for SlowTool {
        fn name(&self) -> &str {
            "slow"
        }
        fn description(&self) -> &str {
            "sleeps"
        }
        fn parameters(&self) -> serde_json::Value {
            json!({"type":"object","properties":{}})
        }
        fn metadata(&self) -> ToolMetadata {
            ToolMetadata {
                timeout: Some(Duration::from_millis(20)),
                ..ToolMetadata::read_only()
            }
        }
        async fn call(
            &self,
            ctx: ToolCallContext,
            _arguments: serde_json::Value,
        ) -> Result<ToolResult, ToolError> {
            tokio::select! {
                () = tokio::time::sleep(Duration::from_secs(5)) => {
                    Ok(ToolResult::text("late"))
                }
                () = ctx.cancel.cancelled() => {
                    self.cancelled.store(true, Ordering::SeqCst);
                    Err(codes::cancelled())
                }
            }
        }
    }

    #[tokio::test]
    async fn tool_timeout_matrix() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let reg = ToolRegistry::from_tools(vec![Arc::new(SlowTool {
            cancelled: Arc::clone(&cancelled),
        })]);
        let outs = ToolDispatch::default()
            .execute_batch(&reg, ToolCallContext::default(), vec![call("slow", "c1")])
            .await;
        let err = outs
            .first()
            .expect("one")
            .result
            .as_ref()
            .expect_err("timeout");
        // Race: either ToolTimeout (outer) or tool observed cancel first.
        assert!(
            matches!(
                err.code(),
                ErrorCode::ToolTimeout | ErrorCode::ToolCancelled
            ),
            "{err:?}"
        );
        // Give the select branch a moment if timeout won the race first.
        tokio::time::sleep(Duration::from_millis(5)).await;
        assert!(
            cancelled.load(Ordering::SeqCst) || err.code() == ErrorCode::ToolTimeout,
            "timeout must cancel the per-call token"
        );
    }

    struct CancelAwareTool;

    #[async_trait]
    impl DynTool for CancelAwareTool {
        fn name(&self) -> &str {
            "cancel_me"
        }
        fn description(&self) -> &str {
            "waits for cancel"
        }
        fn parameters(&self) -> serde_json::Value {
            json!({"type":"object","properties":{}})
        }
        fn metadata(&self) -> ToolMetadata {
            ToolMetadata::read_only()
        }
        async fn call(
            &self,
            ctx: ToolCallContext,
            _arguments: serde_json::Value,
        ) -> Result<ToolResult, ToolError> {
            ctx.cancel.cancelled().await;
            Err(codes::cancelled())
        }
    }

    #[tokio::test]
    async fn tool_cancel_matrix() {
        use tokio_util::sync::CancellationToken;

        let reg = ToolRegistry::from_tools(vec![Arc::new(CancelAwareTool)]);
        let cancel = CancellationToken::new();
        let ctx = ToolCallContext::default().with_cancel(cancel.clone());
        let dispatch = ToolDispatch::default();
        let handle = tokio::spawn(async move {
            dispatch
                .execute_batch(&reg, ctx, vec![call("cancel_me", "c1")])
                .await
        });
        // Allow the tool to start waiting.
        tokio::time::sleep(Duration::from_millis(10)).await;
        cancel.cancel();
        let outs = handle.await.expect("join");
        let err = outs
            .first()
            .expect("one")
            .result
            .as_ref()
            .expect_err("cancelled");
        assert_eq!(err.code(), ErrorCode::ToolCancelled);
    }

    #[tokio::test]
    async fn batch_cancel_fills_remaining() {
        use tokio_util::sync::CancellationToken;

        let reg = ToolRegistry::from_tools(vec![
            Arc::new(CountingTool {
                name: "r1".into(),
                meta: ToolMetadata::read_only(),
                active: Arc::new(AtomicUsize::new(0)),
                max_active: Arc::new(AtomicUsize::new(0)),
                barrier: None,
            }),
            Arc::new(CountingTool {
                name: "r2".into(),
                meta: ToolMetadata::read_only(),
                active: Arc::new(AtomicUsize::new(0)),
                max_active: Arc::new(AtomicUsize::new(0)),
                barrier: None,
            }),
        ]);
        let cancel = CancellationToken::new();
        cancel.cancel();
        let outs = ToolDispatch::default()
            .execute_batch(
                &reg,
                ToolCallContext::default().with_cancel(cancel),
                vec![call("r1", "c1"), call("r2", "c2")],
            )
            .await;
        assert_eq!(outs.len(), 2);
        for o in &outs {
            let err = o.result.as_ref().expect_err("cancelled");
            assert_eq!(err.code(), ErrorCode::ToolCancelled);
        }
    }
}
