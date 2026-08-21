//! Deterministic workflow orchestration engine.
//!
//! Scripts never call LLMs. They issue [`WorkflowHostRequest`] values over a
//! channel; a product host executes nested agents and replies. Results are
//! recorded in a [`Journal`] for resume.
//!
//! **Dependency firewall:** this crate must not depend on `ovo-llm` or
//! HTTP provider stacks.

#![forbid(unsafe_code)]

// Keep the pure-engine surface free of LLM crates by construction.

pub mod engine;
pub mod host;
pub mod journal;
pub mod meta;
pub mod run;
pub mod store;
pub mod validate;

pub use engine::{WorkflowRunParams, run_workflow};
pub use host::{AgentOpts, AgentResult, BudgetState, HostError, WorkflowHostRequest};
pub use journal::{
    HOST_ERROR_KEY, JOURNAL_VERSION_HEADER, Journal, JournalEntry, JournalError, MAX_JOURNAL_BYTES,
    MAX_JOURNAL_ENTRIES, canonical_json, host_error_message, host_error_sentinel,
    is_host_error_sentinel, request_hash,
};
pub use meta::{MetaError, WorkflowMeta, extract_meta};
pub use run::{PauseKind, WorkflowOutcome};
pub use store::{
    FileWorkflowRunStore, MemoryWorkflowRunStore, StoreError, WorkflowRunRecord, WorkflowRunStatus,
    WorkflowRunStore,
};
pub use validate::{
    ValidationError, ValidationReport, default_probe_args, validate_script,
    validate_script_with_agent_budget,
};

/// Default cumulative agent-call budget.
pub const DEFAULT_AGENT_BUDGET: u64 = 128;
/// Hard ceiling for agent budget.
pub const MAX_AGENT_BUDGET: u64 = 1_024;
/// Max items in one `parallel()` panel.
pub const MAX_PARALLEL: usize = 1_024;
/// Max result-bearing host calls per run.
pub const MAX_HOST_CALLS: u64 = 10_000;
