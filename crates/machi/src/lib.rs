//! Machi — embeddable multi-agent runtime kernel (v1 clean break).
//!
//! # Layers (implemented)
//!
//! | Crate | Role |
//! |-------|------|
//! | `machi-types` | ids, messages, usage, errors |
//! | `machi-protocol` | tool id, content blocks, span catalogue |
//! | `machi-obs` | metrics sink, redact, recording / prometheus text |
//! | `machi-tools` | `DynTool`, stream, dispatch, approval |
//! | `machi-toolkit` | cwd-jailed fs/shell tools (feature) |
//! | `machi-llm` | sampler + mock / openai / ollama |
//! | `machi-agent` | definition, builder, discovery |
//! | `machi-state` | conversation handle, ledger, persistence |
//! | `machi-compaction` | compaction strategies |
//! | `machi-runtime` | turn, session, host, workflow adapter |
//! | `machi-workflow` | Rhai engine, journal, validate (no LLM) |
//!
//! # Vertical slice (canonical product path)
//!
//! Session / handle → `TurnRuntime` → tools(+toolkit) → approval / stop gates →
//! metrics → `SessionHost` spawn and/or journaled workflow (+ scratch/template).
//!
//! **Not implemented yet (do not assume):** hooks crate, long-term memory crate,
//! proc-macro derive, full OTEL SDK export, MCP.
//!
//! Optional host capabilities (e.g. `git_diff_since`) require explicit setup.

#![forbid(unsafe_code)]
// Feature-gated transitive deps are unused when compiling the facade lib alone.
#![allow(
    unused_crate_dependencies,
    reason = "facade re-exports optional workspace crates"
)]

#[cfg(feature = "runtime")]
pub use machi_agent as agent;
#[cfg(feature = "runtime")]
pub use machi_agent::{
    Agent, AgentBuilder, AgentDefinition, AgentRegistry, AgentSource, CompletionRequirement,
    EXPLORE, GENERAL_PURPOSE, IdentityAssembler, Instructions, ORCHESTRATOR_DELEGATION_PROMPT,
    PLAN, PROJECT_AGENTS_DIR, PROJECT_AGENTS_MD, ProjectPromptAssembler, PromptAssembler,
    ToolPolicy, USER_AGENTS_DIR, agents_md_path, builtin_definitions, builtin_names, by_name,
    by_name_in_dir, by_name_resolved, discover_in_dir, discover_project, discover_user, load_file,
    parse_definition_markdown, project_agent_dirs, resolve_agents, user_agents_dir,
};
#[cfg(feature = "compaction")]
pub use machi_compaction as compaction;
#[cfg(feature = "runtime")]
pub use machi_llm as llm;
#[cfg(feature = "openai")]
pub use machi_llm::OpenAiCompatSampler;
#[cfg(feature = "runtime")]
pub use machi_llm::{
    Admission, BreakerConfig, BreakerOutcome, BreakerSampler, BreakerState, CircuitBreaker,
    DEFAULT_IDLE_TIMEOUT, DEFAULT_MAX_ATTEMPTS, HttpRetryClass, LlmSampler, MAX_RETRY_AFTER,
    MAX_RETRY_BACKOFF, MockSampler, OpenAiCompatConfig, RATE_LIMIT_RETRY_THRESHOLD, RetryContext,
    RetryDecision, RetryPolicy, RetryingSampler, SampleEvent, SampleRequest, SampleResponse,
    SampleStream, ToolChoice, backoff_for_attempt, build_chat_completions_body,
    classify_http_status, decide_retry, error_code_for_http, is_empty_response,
    parse_chat_completions_response, response_to_stream,
};
#[cfg(feature = "ollama")]
pub use machi_llm::{OllamaConfig, OllamaSampler};
#[cfg(feature = "obs")]
pub use machi_obs as obs;
#[cfg(feature = "obs")]
pub use machi_obs::{
    PrometheusRecorder, REDACTED, RecordingMetrics, emit_catalogue_smoke, looks_like_secret_key,
    metric_catalogue_snapshot, redact_key_value, redact_map, required_metric_names,
    required_span_names,
};
pub use machi_protocol as protocol;
pub use machi_protocol::{
    ContentBlock, IMAGE_TOKEN_COST, ImageBlock, MESSAGE_FRAME_TOKENS, PreflightOverflow,
    SPAN_COMPACT, SPAN_SAMPLE, SPAN_SESSION, SPAN_SPAWN, SPAN_TOOL, SPAN_TOOL_BATCH, SPAN_TURN,
    SPAN_WORKFLOW, SPAN_WORKFLOW_HOST, ToolId, check_context_overflow, estimate_image_tokens,
    estimate_text_tokens, span_catalogue_snapshot,
};
#[cfg(feature = "runtime")]
pub use machi_runtime as runtime;
#[cfg(feature = "runtime")]
pub use machi_runtime::{
    AgentRunResult, CompactionOutcome, CompactionStrategy, CompletionToolGate, ConversationState,
    DEFAULT_MAX_CONCURRENT_CHILDREN, DEFAULT_MAX_SPAWN_DEPTH, GateChain, GateDecision,
    HARD_STOP_THRESHOLD, InProcessHost, InProcessIsolation, IsolationBackend, IsolationEnv,
    LifecycleFanout, MaxMessages, MetricsSink, NUDGE_THRESHOLD, NoopLifecycle, NoopMetrics,
    Session, SessionHost, SharedMetrics, SpawnAgentTool, SpawnOpts, StationarityAction,
    StationarityTracker, StopGate, TokenThreshold, TurnAbortReason, TurnInput,
    TurnLifecycleContributor, TurnOptions, TurnOutcome, TurnRuntime, VecConversationState,
    estimate_conversation_tokens, evaluate_stop_gates, fingerprint_batch, isolation_error,
    nudge_message,
};
#[cfg(all(feature = "runtime", feature = "workflow"))]
pub use machi_runtime::{
    WorkflowSideEffects, run_workflow_configured, run_workflow_on_host,
    run_workflow_on_host_with_metrics,
};
#[cfg(feature = "state")]
pub use machi_state as state;
#[cfg(feature = "state")]
pub use machi_state::{
    ChatPersistence, ChatStateHandle, ChatStateSnapshot, CompactionRecord, DEFAULT_SESSIONS_DIR,
    DEFAULT_SNAPSHOT_EVERY, EVENTS_HEADER, FilePersistence, InMemoryMemory, JsonlPersistence,
    MemoryItem, MemoryPersistence, MemoryPort, NullMemory, NullPersistence, UsageLedger,
    check_tool_pairing, default_session_path, messages_only, session_jsonl_dir,
};
#[cfg(feature = "toolkit")]
pub use machi_toolkit as toolkit;
#[cfg(feature = "toolkit")]
pub use machi_toolkit::{
    GlobTool, GrepTool, ReadFileTool, ShellTool, WriteFileTool, default_toolkit, glob_match,
    resolve_jailed,
};
#[cfg(feature = "runtime")]
pub use machi_tools as tools;
#[cfg(feature = "runtime")]
pub use machi_tools::{
    AlwaysDeny, ApprovalDecision, ApprovalGate, ApprovalPolicy, AutoApprove, CalcTool,
    CapabilityFlag, CapabilityMode, ConcurrencyMode, Destructiveness, DispatchOutcome,
    DispatchRequest, DynTool, EXTRA_SPAWN_DEPTH, InterruptBehavior, MAX_DELTA_BYTES,
    MAX_FRAME_BYTES, SharedTool, StaticToolSource, ToolCallContext, ToolDefinition, ToolDispatch,
    ToolError, ToolMetadata, ToolProgress, ToolRegistry, ToolResult, ToolSource, ToolStream,
    ToolStreamItem, drain_terminal, drain_with_progress, merge_arc_sources, merge_tool_sources,
    partial_progress_frames, terminal_only, with_progress,
};
pub use machi_types as types;
pub use machi_types::{
    AgentId, CompletionTokensDetails, ContentPart, Deadline, ErrorCode, ImageMime, MachiError,
    Message, PromptTokensDetails, Result, RetryClass, Role, RunId, SessionId, ToolCall, ToolCallId,
    Usage, WorkflowRunId,
};
#[cfg(feature = "workflow")]
pub use machi_workflow as workflow;
#[cfg(feature = "workflow")]
pub use machi_workflow::{
    AgentOpts, AgentResult as WorkflowAgentResult, BudgetState, DEFAULT_AGENT_BUDGET,
    FileWorkflowRunStore, HostError, Journal, JournalEntry, JournalError, MAX_AGENT_BUDGET,
    MemoryWorkflowRunStore, PauseKind, StoreError, ValidationError, ValidationReport,
    WorkflowHostRequest, WorkflowMeta, WorkflowOutcome, WorkflowRunParams, WorkflowRunRecord,
    WorkflowRunStatus, WorkflowRunStore, default_probe_args, extract_meta, request_hash,
    run_workflow, validate_script, validate_script_with_agent_budget,
};
