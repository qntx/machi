//! Builtin agent types (W5.2).

use machi_tools::CapabilityMode;

use crate::definition::{AgentDefinition, AgentSource, Instructions, ToolPolicy};

/// Builtin agent type names.
pub const GENERAL_PURPOSE: &str = "general-purpose";
/// Read-only exploration agent.
pub const EXPLORE: &str = "explore";
/// Plan-mode agent (no execute/spawn tools).
pub const PLAN: &str = "plan";

/// Orchestrator delegation guidance (host / parent system prompt fragment).
pub const ORCHESTRATOR_DELEGATION_PROMPT: &str = r"# Delegation
When a subtask is well-scoped, spawn a nested agent:
- `explore` — read-only investigation (search, read files); do not mutate.
- `plan` — design and outline without executing shell/spawn tools.
- `general-purpose` — multi-step work that may use the full tool set.
Pass a clear prompt, optional `label`, and `agent_type`. Prefer the narrowest type that can finish the job.";

const GENERAL_PURPOSE_BODY: &str = r"You are a general-purpose sub-agent.
Complete the assigned task thoroughly. Use tools when needed. Prefer concise, actionable results.";

const EXPLORE_BODY: &str = r"You are a read-only explore agent.
Investigate the codebase or environment using only non-mutating tools.
Do not write files, run destructive shell, or spawn further agents.
Return findings, paths, and concise conclusions.";

const PLAN_BODY: &str = r"You are a plan agent.
Produce a clear plan, design, or review. Do not execute shell commands or spawn nested agents.
Focus on structure, risks, and next steps.";

/// Built-in catalogue (always enabled).
#[must_use]
pub fn builtin_definitions() -> Vec<AgentDefinition> {
    vec![
        AgentDefinition {
            name: GENERAL_PURPOSE.into(),
            description: "General multi-step agent with full tool access".into(),
            instructions: Instructions::Static(GENERAL_PURPOSE_BODY.into()),
            model: "default".into(),
            tools: ToolPolicy::InheritAll,
            output_schema: None,
            completion: None,
            max_steps: 32,
            enabled: true,
            capability: Some(CapabilityMode::Full),
            source: Some(AgentSource::Builtin),
        },
        AgentDefinition {
            name: EXPLORE.into(),
            description: "Read-only exploration agent".into(),
            instructions: Instructions::Static(EXPLORE_BODY.into()),
            model: "default".into(),
            tools: ToolPolicy::InheritAll,
            output_schema: None,
            completion: None,
            max_steps: 24,
            enabled: true,
            capability: Some(CapabilityMode::ReadOnly),
            source: Some(AgentSource::Builtin),
        },
        AgentDefinition {
            name: PLAN.into(),
            description: "Plan/design agent without execute or spawn tools".into(),
            instructions: Instructions::Static(PLAN_BODY.into()),
            model: "default".into(),
            tools: ToolPolicy::InheritAll,
            output_schema: None,
            completion: None,
            max_steps: 16,
            enabled: true,
            capability: Some(CapabilityMode::Plan),
            source: Some(AgentSource::Builtin),
        },
    ]
}

/// Builtin name set (for user-level shadowing skip).
#[must_use]
pub fn builtin_names() -> Vec<&'static str> {
    vec![GENERAL_PURPOSE, EXPLORE, PLAN]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_builtins() {
        let defs = builtin_definitions();
        assert_eq!(defs.len(), 3);
        assert!(
            defs.iter()
                .all(|d| d.enabled && d.source == Some(AgentSource::Builtin))
        );
        assert_eq!(
            defs.iter()
                .find(|d| d.name == EXPLORE)
                .and_then(|d| d.capability),
            Some(CapabilityMode::ReadOnly)
        );
    }
}
