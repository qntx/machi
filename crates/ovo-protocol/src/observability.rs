//! Stable tracing span names and field keys for OpenTelemetry-friendly hosts.

/// Session lifetime span.
pub const SPAN_SESSION: &str = "ovo.session";
/// Single turn span.
pub const SPAN_TURN: &str = "ovo.turn";
/// One LLM sample call.
pub const SPAN_SAMPLE: &str = "ovo.sample";
/// One tool execution.
pub const SPAN_TOOL: &str = "ovo.tool";
/// Dispatched tool batch.
pub const SPAN_TOOL_BATCH: &str = "ovo.tool.batch";
/// Nested agent spawn.
pub const SPAN_SPAWN: &str = "ovo.spawn";
/// Workflow run.
pub const SPAN_WORKFLOW: &str = "ovo.workflow";
/// One workflow host request.
pub const SPAN_WORKFLOW_HOST: &str = "ovo.workflow.host";
/// Compaction pass.
pub const SPAN_COMPACT: &str = "ovo.compact";

/// Canonical field names (string constants for hosts and tests).
pub mod field {
    /// Session id.
    pub const SESSION_ID: &str = "ovo.session_id";
    /// Agent id.
    pub const AGENT_ID: &str = "ovo.agent_id";
    /// Agent name.
    pub const AGENT_NAME: &str = "ovo.agent_name";
    /// Run / turn id.
    pub const RUN_ID: &str = "ovo.run_id";
    /// Step index within a turn.
    pub const STEP: &str = "ovo.step";
    /// Tool name.
    pub const TOOL_NAME: &str = "ovo.tool_name";
    /// Model id.
    pub const MODEL: &str = "ovo.model";
    /// Input tokens.
    pub const USAGE_INPUT: &str = "ovo.usage.input_tokens";
    /// Output tokens.
    pub const USAGE_OUTPUT: &str = "ovo.usage.output_tokens";
    /// Workflow run id.
    pub const WORKFLOW_RUN_ID: &str = "ovo.workflow.run_id";
    /// Workflow journal sequence.
    pub const WORKFLOW_SEQ: &str = "ovo.workflow.seq";
}

/// All required span names (contract test surface).
///
/// **Rename = break:** CI asserts [`span_catalogue_snapshot`].
#[must_use]
pub fn required_span_names() -> &'static [&'static str] {
    &[
        SPAN_SESSION,
        SPAN_TURN,
        SPAN_SAMPLE,
        SPAN_TOOL,
        SPAN_TOOL_BATCH,
        SPAN_SPAWN,
        SPAN_WORKFLOW,
        SPAN_WORKFLOW_HOST,
        SPAN_COMPACT,
    ]
}

/// Exact newline-joined span catalogue for CI golden comparison.
#[must_use]
pub fn span_catalogue_snapshot() -> String {
    required_span_names().join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPAN_CATALOGUE_GOLDEN: &str = "\
ovo.session
ovo.turn
ovo.sample
ovo.tool
ovo.tool.batch
ovo.spawn
ovo.workflow
ovo.workflow.host
ovo.compact";

    #[test]
    fn span_names_are_ovo_prefixed_and_unique() {
        let names = required_span_names();
        assert_eq!(names.len(), 9, "expected nine spans");
        let mut seen = std::collections::BTreeSet::new();
        for name in names {
            assert!(name.starts_with("ovo."), "span {name} must start with ovo.");
            assert!(seen.insert(*name), "duplicate span {name}");
        }
    }

    #[test]
    fn span_catalogue_snapshot_matches_golden() {
        assert_eq!(
            span_catalogue_snapshot(),
            SPAN_CATALOGUE_GOLDEN,
            "span catalogue changed — update golden only with deliberate contract change"
        );
    }

    #[test]
    fn field_keys_are_stable() {
        assert_eq!(field::SESSION_ID, "ovo.session_id");
        assert_eq!(field::TOOL_NAME, "ovo.tool_name");
        assert_eq!(field::WORKFLOW_SEQ, "ovo.workflow.seq");
    }
}
