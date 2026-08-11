//! Session host for nested agent runs (dynamic multi-agent delegation).
//!
//! # Limits (fail-closed)
//!
//! - **`agent_budget`** — absolute number of admitted spawns for this host
//! - **`max_spawn_depth`** — max nesting index (`depth` on [`SpawnOpts`]); `depth >= max` rejects
//! - **`max_concurrent_children`** — simultaneous in-flight spawns (try-acquire; no queue)

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use async_trait::async_trait;
use futures::future::try_join_all;
use machi_agent::{
    Agent, AgentBuilder, AgentDefinition, AgentRegistry, IdentityAssembler, PromptAssembler,
};
use machi_llm::LlmSampler;
use machi_obs::{NoopMetrics, SharedMetrics, record_spawn};
use machi_state::ChatStateHandle;
use machi_tools::SharedTool;
use machi_tools::registry::CapabilityMode;
use machi_types::{AgentId, ErrorCode, MachiError, Message, Usage};
use machi_workflow::{WorkflowRunStatus, WorkflowRunStore};
use serde_json::Value;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;
use tracing::{Instrument, Span, info_span};

use crate::isolation::{InProcessIsolation, IsolationBackend};
use crate::state::VecConversationState;
use crate::turn::{TurnInput, TurnOptions, TurnRuntime};

/// Default max nesting depth for nested agents (`0..DEFAULT_MAX_SPAWN_DEPTH`).
pub const DEFAULT_MAX_SPAWN_DEPTH: u32 = 16;
/// Default max concurrent in-flight nested agents.
pub const DEFAULT_MAX_CONCURRENT_CHILDREN: usize = 64;

/// Options for spawning a nested agent.
///
/// Field parity with workflow [`machi_workflow::AgentOpts`] for Mode A/B isomorphism.
#[derive(Debug, Clone)]
pub struct SpawnOpts {
    /// User prompt for the child.
    pub prompt: String,
    /// Optional label for logs / result correlation.
    pub label: Option<String>,
    /// Override model.
    pub model: Option<String>,
    /// Capability mode for child tools.
    pub capability_mode: CapabilityMode,
    /// Max steps for child turn.
    pub max_steps: Option<usize>,
    /// Cancel token for this child.
    pub cancel: CancellationToken,
    /// Definition name resolved via host agent catalogue.
    pub agent_type: Option<String>,
    /// Optional JSON schema for structured child output.
    pub output_schema: Option<Value>,
    /// When true, seed the child conversation from [`Self::fork_messages`].
    pub fork_context: bool,
    /// Parent messages injected when [`Self::fork_context`] is true.
    pub fork_messages: Option<Vec<Message>>,
    /// Resume a prior nested run id (unsupported on default host → error).
    pub resume_from: Option<String>,
    /// Max output tokens hint for the child sample.
    pub max_output_tokens: Option<u64>,
    /// Nesting depth of this spawn (`0` = first level under the host).
    pub depth: u32,
}

impl SpawnOpts {
    /// Prompt-only spawn with a fresh cancel token at depth 0.
    #[must_use]
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            label: None,
            model: None,
            capability_mode: CapabilityMode::Full,
            max_steps: None,
            cancel: CancellationToken::new(),
            agent_type: None,
            output_schema: None,
            fork_context: false,
            fork_messages: None,
            resume_from: None,
            max_output_tokens: None,
            depth: 0,
        }
    }

    /// Set label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set capability mode.
    #[must_use]
    pub const fn with_capability(mut self, mode: CapabilityMode) -> Self {
        self.capability_mode = mode;
        self
    }

    /// Set cancel token (often a child of a parent token).
    #[must_use]
    pub fn with_cancel(mut self, cancel: CancellationToken) -> Self {
        self.cancel = cancel;
        self
    }

    /// Set max steps.
    #[must_use]
    pub const fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    /// Set nesting depth.
    #[must_use]
    pub const fn with_depth(mut self, depth: u32) -> Self {
        self.depth = depth;
        self
    }

    /// Set agent type / definition name.
    #[must_use]
    pub fn with_agent_type(mut self, agent_type: impl Into<String>) -> Self {
        self.agent_type = Some(agent_type.into());
        self
    }

    /// Set structured output schema for the child.
    #[must_use]
    pub fn with_output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Set max output tokens hint.
    #[must_use]
    pub const fn with_max_output_tokens(mut self, n: u64) -> Self {
        self.max_output_tokens = Some(n);
        self
    }

    /// Request parent conversation fork (requires [`Self::with_fork_messages`]).
    #[must_use]
    pub const fn with_fork_context(mut self, fork: bool) -> Self {
        self.fork_context = fork;
        self
    }

    /// Seed child state with parent messages and enable fork mode.
    #[must_use]
    pub fn with_fork_messages(mut self, messages: Vec<Message>) -> Self {
        self.fork_context = true;
        self.fork_messages = Some(messages);
        self
    }

    /// Request resume of a prior nested run (unsupported on default host → error).
    #[must_use]
    pub fn with_resume_from(mut self, id: impl Into<String>) -> Self {
        self.resume_from = Some(id.into());
        self
    }
}

/// Result of a nested agent run.
#[derive(Debug, Clone)]
pub struct AgentRunResult {
    /// Child agent id.
    pub agent_id: AgentId,
    /// Optional label echoed from [`SpawnOpts`].
    pub label: Option<String>,
    /// Whether the run completed without runtime error.
    pub success: bool,
    /// Model text or structured payload.
    pub output: Value,
    /// Cancelled flag.
    pub cancelled: bool,
    /// Usage.
    pub usage: Usage,
    /// Wall duration ms.
    pub duration_ms: u64,
    /// Steps.
    pub steps: usize,
}

/// Host capable of spawning nested agents.
#[async_trait]
pub trait SessionHost: Send + Sync {
    /// Spawn and run a nested agent to completion.
    async fn spawn_agent(&self, opts: SpawnOpts) -> Result<AgentRunResult, MachiError>;

    /// Spawn many nested agents concurrently (order of results matches input).
    async fn spawn_agents(&self, opts: Vec<SpawnOpts>) -> Result<Vec<AgentRunResult>, MachiError> {
        try_join_all(opts.into_iter().map(|o| self.spawn_agent(o))).await
    }
}

/// In-process host: nested [`TurnRuntime`] with shared sampler, tool pool, and limits.
pub struct InProcessHost {
    sampler: Arc<dyn LlmSampler>,
    tools: Vec<SharedTool>,
    base_instructions: String,
    runtime: TurnRuntime,
    /// Absolute cap on nested agent spawns (`None` = unlimited).
    agent_budget: Option<u64>,
    spent: AtomicU64,
    /// Max spawn depth (`None` = unlimited). Depth must satisfy `depth < max`.
    max_spawn_depth: Option<u32>,
    /// Concurrent in-flight children (`None` = unlimited).
    concurrency: Option<Arc<Semaphore>>,
    max_concurrent_children: Option<usize>,
    /// Named agent definitions for `agent_type` resolution.
    agent_registry: AgentRegistry,
    /// System prompt assembler (project AGENTS.md, etc.).
    prompt_assembler: Arc<dyn PromptAssembler>,
    /// Isolation backend for child environments (default in-process).
    isolation: Arc<dyn IsolationBackend>,
    metrics: SharedMetrics,
    /// Parent session handle: seeds `fork_context` when `fork_messages` is unset.
    parent_handle: Option<ChatStateHandle>,
    /// Workflow run store for `resume_from` lookups.
    run_store: Option<Arc<dyn WorkflowRunStore>>,
}

impl std::fmt::Debug for InProcessHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InProcessHost")
            .field("tools", &self.tools.len())
            .field("base_instructions_len", &self.base_instructions.len())
            .field("runtime", &self.runtime)
            .field("agent_budget", &self.agent_budget)
            .field("spent", &self.spent.load(Ordering::Relaxed))
            .field("max_spawn_depth", &self.max_spawn_depth)
            .field("max_concurrent_children", &self.max_concurrent_children)
            .field("agent_registry", &self.agent_registry.len())
            .field("isolation", &self.isolation.name())
            .finish_non_exhaustive()
    }
}

impl InProcessHost {
    /// Create a host with default depth/concurrency caps and unlimited agent budget.
    #[must_use]
    pub fn new(sampler: Arc<dyn LlmSampler>, tools: Vec<SharedTool>) -> Self {
        Self {
            sampler,
            tools,
            base_instructions: "You are a focused sub-agent. Complete the task.".into(),
            runtime: TurnRuntime::new(),
            agent_budget: None,
            spent: AtomicU64::new(0),
            max_spawn_depth: Some(DEFAULT_MAX_SPAWN_DEPTH),
            concurrency: Some(Arc::new(Semaphore::new(DEFAULT_MAX_CONCURRENT_CHILDREN))),
            max_concurrent_children: Some(DEFAULT_MAX_CONCURRENT_CHILDREN),
            agent_registry: AgentRegistry::with_builtins(),
            prompt_assembler: Arc::new(IdentityAssembler),
            isolation: Arc::new(InProcessIsolation),
            metrics: Arc::new(NoopMetrics),
            parent_handle: None,
            run_store: None,
        }
    }

    /// Absolute agent-call budget for this host (every successful admission counts 1).
    #[must_use]
    pub const fn with_agent_budget(mut self, budget: u64) -> Self {
        self.agent_budget = Some(budget);
        self
    }

    /// Cap nesting depth (`depth` must be `< max`). `None` disables the limit.
    #[must_use]
    pub const fn with_max_spawn_depth(mut self, max: Option<u32>) -> Self {
        self.max_spawn_depth = max;
        self
    }

    /// Cap concurrent in-flight children. `None` disables the limit.
    #[must_use]
    pub fn with_max_concurrent_children(mut self, max: Option<usize>) -> Self {
        self.max_concurrent_children = max;
        self.concurrency = max.map(|n| Arc::new(Semaphore::new(n.max(1))));
        self
    }

    /// Install a full agent registry for `agent_type` resolution.
    #[must_use]
    pub fn with_agent_registry(mut self, registry: AgentRegistry) -> Self {
        self.agent_registry = registry;
        self
    }

    /// Register agent definitions (merged into the host registry).
    #[must_use]
    pub fn with_agent_definitions(
        mut self,
        defs: impl IntoIterator<Item = AgentDefinition>,
    ) -> Self {
        self.agent_registry = self
            .agent_registry
            .merge(&AgentRegistry::from_definitions(defs));
        self
    }

    /// Install a prompt assembler applied when resolving `agent_type` definitions.
    #[must_use]
    pub fn with_prompt_assembler(mut self, assembler: Arc<dyn PromptAssembler>) -> Self {
        self.prompt_assembler = assembler;
        self
    }

    /// Install an isolation backend (default [`InProcessIsolation`]).
    #[must_use]
    pub fn with_isolation(mut self, isolation: Arc<dyn IsolationBackend>) -> Self {
        self.isolation = isolation;
        self
    }

    /// Metrics sink for spawn/turn accounting.
    #[must_use]
    pub fn with_metrics(mut self, metrics: SharedMetrics) -> Self {
        self.metrics = metrics;
        self
    }

    /// Parent conversation handle used when `fork_context` is set without messages.
    #[must_use]
    pub fn with_parent_handle(mut self, handle: ChatStateHandle) -> Self {
        self.parent_handle = Some(handle);
        self
    }

    /// Workflow run store enabling `resume_from` on spawn opts.
    #[must_use]
    pub fn with_run_store(mut self, store: Arc<dyn WorkflowRunStore>) -> Self {
        self.run_store = Some(store);
        self
    }

    /// Override child system instructions (used when `agent_type` is unset).
    #[must_use]
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.base_instructions = instructions.into();
        self
    }

    /// Shared agent registry (clone is cheap).
    #[must_use]
    pub fn agent_registry(&self) -> &AgentRegistry {
        &self.agent_registry
    }

    /// Agents admitted so far (including in-flight after reservation).
    #[must_use]
    pub fn agents_spent(&self) -> u64 {
        self.spent.load(Ordering::Relaxed)
    }

    /// Remaining budget when capped.
    #[must_use]
    pub fn agents_remaining(&self) -> Option<u64> {
        self.agent_budget
            .map(|b| b.saturating_sub(self.spent.load(Ordering::Relaxed)))
    }

    /// Configured max spawn depth.
    #[must_use]
    pub const fn max_spawn_depth(&self) -> Option<u32> {
        self.max_spawn_depth
    }

    /// Configured max concurrent children.
    #[must_use]
    pub const fn max_concurrent_children(&self) -> Option<usize> {
        self.max_concurrent_children
    }

    fn check_depth(&self, depth: u32) -> Result<(), MachiError> {
        if let Some(max) = self.max_spawn_depth
            && depth >= max
        {
            return Err(MachiError::new(
                ErrorCode::HostDepth,
                format!("spawn depth {depth} exceeds max_spawn_depth {max}"),
            ));
        }
        Ok(())
    }

    fn check_fork_opts(opts: &SpawnOpts, has_parent_handle: bool) -> Result<(), MachiError> {
        if opts.fork_context && opts.fork_messages.is_none() && !has_parent_handle {
            return Err(MachiError::new(
                ErrorCode::HostUnsupported,
                "fork_context requires fork_messages or host parent_handle",
            ));
        }
        Ok(())
    }

    async fn child_state(&self, opts: &SpawnOpts) -> Result<VecConversationState, MachiError> {
        if !opts.fork_context {
            return Ok(VecConversationState::new());
        }
        if let Some(msgs) = &opts.fork_messages {
            return Ok(VecConversationState::from_messages(msgs.clone()));
        }
        if let Some(handle) = &self.parent_handle {
            let msgs = handle.messages().await;
            return Ok(VecConversationState::from_messages(msgs));
        }
        Err(MachiError::new(
            ErrorCode::HostUnsupported,
            "fork_context requires fork_messages or host parent_handle",
        ))
    }

    /// Resolve `resume_from` via [`WorkflowRunStore`] when configured.
    fn try_resume(&self, opts: &SpawnOpts) -> Result<Option<AgentRunResult>, MachiError> {
        let Some(id) = opts.resume_from.as_deref() else {
            return Ok(None);
        };
        let Some(store) = &self.run_store else {
            return Err(MachiError::new(
                ErrorCode::HostUnsupported,
                "resume_from requires host run_store (WorkflowRunStore)",
            ));
        };
        let rec = store.get(id).map_err(|e| {
            MachiError::new(ErrorCode::HostSpawn, format!("workflow run store: {e}"))
        })?;
        let Some(rec) = rec else {
            return Err(MachiError::new(
                ErrorCode::HostUnsupported,
                format!("resume_from run_id '{id}' not found in WorkflowRunStore"),
            ));
        };
        match rec.status {
            WorkflowRunStatus::Completed => {
                let output = rec
                    .message
                    .clone()
                    .map_or_else(|| Value::String(rec.name.clone()), Value::String);
                Ok(Some(AgentRunResult {
                    agent_id: AgentId::generate(),
                    label: opts.label.clone().or_else(|| Some(rec.name.clone())),
                    success: true,
                    output,
                    cancelled: false,
                    usage: Usage::zero(),
                    duration_ms: 0,
                    steps: 0,
                }))
            }
            WorkflowRunStatus::Paused | WorkflowRunStatus::BudgetExceeded => Err(MachiError::new(
                ErrorCode::HostUnsupported,
                format!(
                    "resume_from '{id}' is {:?}; resume via workflow engine (journal {})",
                    rec.status,
                    rec.journal_path.display()
                ),
            )),
            other => Err(MachiError::new(
                ErrorCode::HostUnsupported,
                format!("resume_from '{id}' has non-resumable status {other:?}"),
            )),
        }
    }

    fn try_acquire_concurrency(&self) -> Result<Option<OwnedSemaphorePermit>, MachiError> {
        let Some(sem) = &self.concurrency else {
            return Ok(None);
        };
        match Arc::clone(sem).try_acquire_owned() {
            Ok(permit) => Ok(Some(permit)),
            Err(_) => Err(MachiError::new(
                ErrorCode::HostConcurrency,
                format!(
                    "max concurrent children reached ({})",
                    self.max_concurrent_children.unwrap_or(0)
                ),
            )),
        }
    }

    fn reserve_slot(&self) -> Result<(), MachiError> {
        let Some(budget) = self.agent_budget else {
            self.spent.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        };
        self.reserve_against_budget(budget)
    }

    fn reserve_against_budget(&self, budget: u64) -> Result<(), MachiError> {
        loop {
            let spent = self.spent.load(Ordering::Acquire);
            if spent >= budget {
                return Err(MachiError::new(
                    ErrorCode::HostBudget,
                    format!("agent budget exhausted: spent {spent}, maximum {budget}"),
                ));
            }
            if self
                .spent
                .compare_exchange(spent, spent + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    fn effective_capability(&self, opts: &SpawnOpts) -> CapabilityMode {
        let mut mode = opts.capability_mode;
        if let Some(name) = opts.agent_type.as_deref()
            && let Some(def) = self.agent_registry.get(name)
            && let Some(def_cap) = def.capability
        {
            mode = mode.intersect(def_cap);
        }
        mode
    }

    fn build_child(&self, opts: &SpawnOpts) -> Result<Agent, MachiError> {
        let mut builder = if let Some(name) = opts.agent_type.as_deref() {
            let def = self.agent_registry.require(name)?.clone();
            let system = self.prompt_assembler.assemble(&def)?;
            AgentBuilder::from_definition(def)
                .instructions(system)
                .tools(self.tools.clone())
        } else {
            let name = opts.label.clone().unwrap_or_else(|| "subagent".to_owned());
            AgentBuilder::named(name)
                .instructions(self.base_instructions.clone())
                .tools(self.tools.clone())
        };

        if let Some(model) = &opts.model {
            builder = builder.model(model.clone());
        }
        if let Some(max_steps) = opts.max_steps {
            builder = builder.max_steps(max_steps);
        }
        if let Some(schema) = opts.output_schema.clone() {
            builder = builder.output_schema(schema);
        }
        builder.build()
    }

    async fn spawn_one(&self, opts: SpawnOpts) -> Result<AgentRunResult, MachiError> {
        let agent_id = AgentId::generate();
        let label = opts.label.clone();
        let parent = Span::current();
        let span = info_span!(
            parent: parent,
            "machi.spawn",
            machi.agent_id = %agent_id,
            machi.agent_label = label.as_deref().unwrap_or(""),
            machi.agent_type = opts.agent_type.as_deref().unwrap_or(""),
            machi.capability = ?opts.capability_mode,
            machi.spawn_depth = opts.depth,
        );

        async move {
            if opts.cancel.is_cancelled() {
                return Err(MachiError::new(
                    ErrorCode::HostCancelled,
                    "spawn cancelled before start",
                ));
            }
            if let Some(resumed) = self.try_resume(&opts)? {
                return Ok(resumed);
            }
            Self::check_fork_opts(&opts, self.parent_handle.is_some())?;
            self.check_depth(opts.depth)?;
            let _permit = self.try_acquire_concurrency()?;
            self.reserve_slot()?;

            let isolation_env = self.isolation.prepare(&opts).await?;
            let started = Instant::now();
            let agent = self.build_child(&opts)?;
            let mut state = self.child_state(&opts).await?;
            let capability_mode = self.effective_capability(&opts);
            let max_steps = opts.max_steps.or_else(|| {
                opts.agent_type
                    .as_deref()
                    .and_then(|n| self.agent_registry.get(n).map(|d| d.max_steps))
            });
            let max_output_tokens = opts.max_output_tokens.and_then(|n| u32::try_from(n).ok());
            let turn_opts = TurnOptions {
                max_steps,
                capability_mode,
                cancel: opts.cancel.clone(),
                agent_id: Some(agent_id.clone()),
                metrics: Arc::clone(&self.metrics),
                spawn_depth: Some(opts.depth),
                max_output_tokens,
                cwd: isolation_env.cwd.clone(),
                ..TurnOptions::default()
            };
            let outcome = match self
                .runtime
                .run(
                    &agent,
                    self.sampler.as_ref(),
                    &mut state,
                    TurnInput::Text(opts.prompt),
                    turn_opts,
                )
                .await
            {
                Ok(o) => o,
                Err(e) => {
                    let _ = self.isolation.cleanup(&isolation_env).await;
                    record_spawn(self.metrics.as_ref(), "error");
                    return Err(map_turn_error(e));
                }
            };
            if let Err(e) = self.isolation.cleanup(&isolation_env).await {
                record_spawn(self.metrics.as_ref(), "error");
                return Err(e);
            }
            let status = if outcome.cancelled { "cancelled" } else { "ok" };
            record_spawn(self.metrics.as_ref(), status);

            let output = outcome
                .output_json
                .unwrap_or_else(|| Value::String(outcome.output_text.clone()));
            Ok(AgentRunResult {
                agent_id,
                label,
                success: !outcome.cancelled,
                output,
                cancelled: outcome.cancelled,
                usage: outcome.usage,
                duration_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
                steps: outcome.steps,
            })
        }
        .instrument(span)
        .await
    }
}

fn map_turn_error(e: MachiError) -> MachiError {
    if matches!(
        e.code(),
        ErrorCode::RuntimeCancelled | ErrorCode::LlmCancelled
    ) {
        MachiError::new(ErrorCode::HostCancelled, e.message().to_owned()).with_source(e)
    } else {
        MachiError::new(ErrorCode::HostSpawn, e.message().to_owned()).with_source(e)
    }
}

#[async_trait]
impl SessionHost for InProcessHost {
    async fn spawn_agent(&self, opts: SpawnOpts) -> Result<AgentRunResult, MachiError> {
        self.spawn_one(opts).await
    }

    async fn spawn_agents(&self, opts: Vec<SpawnOpts>) -> Result<Vec<AgentRunResult>, MachiError> {
        try_join_all(opts.into_iter().map(|o| self.spawn_one(o))).await
    }
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::excessive_nesting,
    reason = "unit tests use expect and nested mock structs"
)]
mod tests {
    use std::sync::Arc;

    use machi_llm::MockSampler;
    use machi_types::ErrorCode;
    use serde_json::json;

    use super::*;

    #[tokio::test]
    async fn concurrent_two_workers() {
        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("task a", "worker-a-result");
        sampler.map_user_text("task b", "worker-b-result");
        let host = InProcessHost::new(sampler, vec![]);
        let results = host
            .spawn_agents(vec![
                SpawnOpts::new("task a").with_label("alpha"),
                SpawnOpts::new("task b").with_label("beta"),
            ])
            .await
            .expect("spawn");
        assert_eq!(results.len(), 2);
        assert_eq!(
            results.first().and_then(|r| r.label.as_deref()),
            Some("alpha")
        );
        assert_eq!(
            results.get(1).and_then(|r| r.label.as_deref()),
            Some("beta")
        );
        assert_eq!(
            results.first().map(|r| &r.output),
            Some(&Value::String("worker-a-result".into()))
        );
        assert_eq!(
            results.get(1).map(|r| &r.output),
            Some(&Value::String("worker-b-result".into()))
        );
        assert_eq!(host.agents_spent(), 2);
    }

    #[tokio::test]
    async fn budget_exhausted() {
        let sampler = Arc::new(MockSampler::new());
        sampler.push_text("only-one");
        let host = InProcessHost::new(sampler, vec![]).with_agent_budget(1);
        host.spawn_agent(SpawnOpts::new("first"))
            .await
            .expect("first");
        let err = host
            .spawn_agent(SpawnOpts::new("second"))
            .await
            .expect_err("budget");
        assert_eq!(err.code(), ErrorCode::HostBudget);
    }

    #[tokio::test]
    async fn cancel_before_start() {
        let sampler = Arc::new(MockSampler::new());
        let host = InProcessHost::new(sampler, vec![]);
        let cancel = CancellationToken::new();
        cancel.cancel();
        let err = host
            .spawn_agent(SpawnOpts::new("x").with_cancel(cancel))
            .await
            .expect_err("cancel");
        assert_eq!(err.code(), ErrorCode::HostCancelled);
    }

    #[tokio::test]
    async fn depth_fail_closed() {
        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("ok", "done");
        let host = InProcessHost::new(sampler, vec![]).with_max_spawn_depth(Some(1));
        host.spawn_agent(SpawnOpts::new("ok").with_depth(0))
            .await
            .expect("depth 0");
        let err = host
            .spawn_agent(SpawnOpts::new("ok").with_depth(1))
            .await
            .expect_err("depth");
        assert_eq!(err.code(), ErrorCode::HostDepth);
    }

    #[tokio::test]
    async fn concurrency_fail_closed() {
        struct HoldingSampler {
            inner: MockSampler,
            release: tokio::sync::Notify,
            entered: tokio::sync::Notify,
        }

        #[async_trait]
        impl LlmSampler for HoldingSampler {
            async fn sample(
                &self,
                request: machi_llm::SampleRequest,
            ) -> Result<machi_llm::SampleResponse, MachiError> {
                self.entered.notify_one();
                self.release.notified().await;
                self.inner.sample(request).await
            }
        }

        let holder = Arc::new(HoldingSampler {
            inner: MockSampler::new(),
            release: tokio::sync::Notify::new(),
            entered: tokio::sync::Notify::new(),
        });
        holder.inner.map_user_text("slow", "done");
        holder.inner.map_user_text("fast", "nope");

        let sampler: Arc<dyn LlmSampler> = holder.clone();
        let host =
            Arc::new(InProcessHost::new(sampler, vec![]).with_max_concurrent_children(Some(1)));

        let h1 = Arc::clone(&host);
        let t1 = tokio::spawn(async move { h1.spawn_agent(SpawnOpts::new("slow")).await });
        holder.entered.notified().await;

        let err = host
            .spawn_agent(SpawnOpts::new("fast"))
            .await
            .expect_err("concurrency");
        assert_eq!(err.code(), ErrorCode::HostConcurrency);

        holder.release.notify_one();
        t1.await.expect("join").expect("first ok");
    }

    #[tokio::test]
    async fn fork_context_requires_messages() {
        let sampler = Arc::new(MockSampler::new());
        let host = InProcessHost::new(sampler, vec![]);
        let err = host
            .spawn_agent(SpawnOpts::new("x").with_fork_context(true))
            .await
            .expect_err("fork");
        assert_eq!(err.code(), ErrorCode::HostUnsupported);
    }

    #[tokio::test]
    async fn fork_context_from_parent_handle() {
        use machi_state::ChatStateHandle;
        use machi_types::Message;

        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("continue", "handle-fork");
        let handle = ChatStateHandle::spawn(vec![Message::user("seed"), Message::assistant("a")]);
        let host = InProcessHost::new(sampler, vec![]).with_parent_handle(handle);
        let run = host
            .spawn_agent(SpawnOpts::new("continue").with_fork_context(true))
            .await
            .expect("fork");
        assert_eq!(run.output, Value::String("handle-fork".into()));
    }

    #[tokio::test]
    async fn resume_from_completed_run_store() {
        use std::path::PathBuf;
        use std::sync::Arc;

        use machi_workflow::{MemoryWorkflowRunStore, WorkflowOutcome, WorkflowRunRecord};

        let sampler = Arc::new(MockSampler::new());
        let store = Arc::new(MemoryWorkflowRunStore::new());
        let mut rec = WorkflowRunRecord::new_running("r1", "wf", PathBuf::from("/tmp/j.jsonl"));
        rec.apply_outcome(&WorkflowOutcome::Completed { result: json!({}) });
        rec.message = Some("cached-output".into());
        store.put(rec).expect("put");
        let host = InProcessHost::new(sampler, vec![]).with_run_store(store);
        let run = host
            .spawn_agent(SpawnOpts::new("unused").with_resume_from("r1"))
            .await
            .expect("resume");
        assert!(run.success);
        assert_eq!(run.output, Value::String("cached-output".into()));
    }

    #[tokio::test]
    async fn builtin_explore_spawnable() {
        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("look", "found");
        let host = InProcessHost::new(sampler, vec![]);
        let run = host
            .spawn_agent(SpawnOpts::new("look").with_agent_type("explore"))
            .await
            .expect("explore");
        assert_eq!(run.output, Value::String("found".into()));
    }

    #[tokio::test]
    async fn fork_context_seeds_parent_messages() {
        use machi_types::Message;

        let sampler = Arc::new(MockSampler::new());
        // Child user prompt is "continue"; parent context is already in state.
        sampler.map_user_text("continue", "forked-ok");
        let host = InProcessHost::new(sampler, vec![]);
        let parent = vec![
            Message::system("parent-sys"),
            Message::user("earlier"),
            Message::assistant("prior answer"),
        ];
        let run = host
            .spawn_agent(SpawnOpts::new("continue").with_fork_messages(parent))
            .await
            .expect("fork spawn");
        assert_eq!(run.output, Value::String("forked-ok".into()));
    }

    #[tokio::test]
    async fn agent_type_not_found() {
        let sampler = Arc::new(MockSampler::new());
        let host = InProcessHost::new(sampler, vec![]);
        let err = host
            .spawn_agent(SpawnOpts::new("x").with_agent_type("missing"))
            .await
            .expect_err("type");
        assert_eq!(err.code(), ErrorCode::AgentNotFound);
    }

    #[tokio::test]
    async fn agent_type_resolves_definition() {
        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("do work", "from-def");
        let mut def = AgentDefinition::new("worker");
        def.description = "w".into();
        def.instructions = machi_agent::Instructions::Static("Be brief.".into());
        def.model = "mock".into();
        def.max_steps = 4;
        let reg = AgentRegistry::from_definitions([def]);
        let host = InProcessHost::new(sampler, vec![]).with_agent_registry(reg);
        let run = host
            .spawn_agent(SpawnOpts::new("do work").with_agent_type("worker"))
            .await
            .expect("spawn");
        assert_eq!(run.output, Value::String("from-def".into()));
    }

    #[tokio::test]
    async fn prompt_assembler_applied_for_agent_type() {
        use machi_agent::ProjectPromptAssembler;

        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("task", "done");
        let mut def = AgentDefinition::new("worker");
        def.description = "w".into();
        def.instructions = machi_agent::Instructions::Static("Body.".into());
        def.model = "mock".into();
        def.max_steps = 4;
        let asm = Arc::new(ProjectPromptAssembler::with_preamble("PREAMBLE_MARK"));
        let host = InProcessHost::new(sampler, vec![])
            .with_agent_definitions([def])
            .with_prompt_assembler(asm);
        let run = host
            .spawn_agent(SpawnOpts::new("task").with_agent_type("worker"))
            .await
            .expect("spawn");
        assert!(run.success);
        // Assembler applied at build time; spawn still succeeds with mock.
        assert_eq!(run.output, Value::String("done".into()));
    }

    #[tokio::test]
    async fn output_schema_field_accepted() {
        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("schema", r#"{"ok":true}"#);
        let host = InProcessHost::new(sampler, vec![]);
        let schema = json!({
            "type": "object",
            "properties": { "ok": { "type": "boolean" } },
            "required": ["ok"]
        });
        let run = host
            .spawn_agent(SpawnOpts::new("schema").with_output_schema(schema))
            .await
            .expect("spawn");
        assert!(run.success);
    }

    #[tokio::test]
    async fn isolation_prepare_cleanup_on_spawn() {
        use std::sync::atomic::{AtomicUsize, Ordering as AtOrd};

        use crate::isolation::{IsolationBackend, IsolationEnv};

        struct CountingIsolation {
            prepares: Arc<AtomicUsize>,
            cleanups: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl IsolationBackend for CountingIsolation {
            fn name(&self) -> &'static str {
                "counting"
            }

            async fn prepare(&self, opts: &SpawnOpts) -> Result<IsolationEnv, MachiError> {
                self.prepares.fetch_add(1, AtOrd::SeqCst);
                Ok(IsolationEnv {
                    cwd: None,
                    label: opts.label.clone(),
                })
            }

            async fn cleanup(&self, _env: &IsolationEnv) -> Result<(), MachiError> {
                self.cleanups.fetch_add(1, AtOrd::SeqCst);
                Ok(())
            }
        }

        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("iso", "ok");
        let prepares = Arc::new(AtomicUsize::new(0));
        let cleanups = Arc::new(AtomicUsize::new(0));
        let host =
            InProcessHost::new(sampler, vec![]).with_isolation(Arc::new(CountingIsolation {
                prepares: Arc::clone(&prepares),
                cleanups: Arc::clone(&cleanups),
            }));
        host.spawn_agent(SpawnOpts::new("iso").with_label("child"))
            .await
            .expect("spawn");
        assert_eq!(prepares.load(AtOrd::SeqCst), 1, "prepare once");
        assert_eq!(cleanups.load(AtOrd::SeqCst), 1, "cleanup once");
    }

    #[tokio::test]
    async fn isolation_prepare_fail_closed() {
        use crate::isolation::{IsolationBackend, IsolationEnv, isolation_error};

        struct FailPrepare;

        #[async_trait]
        impl IsolationBackend for FailPrepare {
            fn name(&self) -> &'static str {
                "fail"
            }

            async fn prepare(&self, _opts: &SpawnOpts) -> Result<IsolationEnv, MachiError> {
                Err(isolation_error(self.name(), "no sandbox available"))
            }

            async fn cleanup(&self, _env: &IsolationEnv) -> Result<(), MachiError> {
                Ok(())
            }
        }

        let sampler = Arc::new(MockSampler::new());
        let host = InProcessHost::new(sampler, vec![]).with_isolation(Arc::new(FailPrepare));
        let err = host
            .spawn_agent(SpawnOpts::new("x"))
            .await
            .expect_err("iso fail");
        assert_eq!(err.code(), ErrorCode::HostIsolation);
        assert_eq!(host.agents_spent(), 1, "slot reserved before prepare");
    }
}
