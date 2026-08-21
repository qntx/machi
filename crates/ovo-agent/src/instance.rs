//! Built agent instance.

use std::sync::Arc;

use ovo_tools::ToolRegistry;

use crate::definition::AgentDefinition;

/// Session-bound agent: definition + resolved prompt + tools.
#[derive(Clone)]
pub struct Agent {
    definition: AgentDefinition,
    system_prompt: String,
    tools: Arc<ToolRegistry>,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("name", &self.definition.name)
            .field("model", &self.definition.model)
            .field("system_prompt_len", &self.system_prompt.len())
            .field("tools", &self.tools.len())
            .finish()
    }
}

impl Agent {
    /// Construct directly (prefer [`crate::AgentBuilder`]).
    #[must_use]
    pub fn new(
        definition: AgentDefinition,
        system_prompt: String,
        tools: Arc<ToolRegistry>,
    ) -> Self {
        Self {
            definition,
            system_prompt,
            tools,
        }
    }

    /// Agent name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.definition.name
    }

    /// Model id.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.definition.model
    }

    /// Resolved system prompt.
    #[must_use]
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// Definition.
    #[must_use]
    pub const fn definition(&self) -> &AgentDefinition {
        &self.definition
    }

    /// Tool registry.
    #[must_use]
    pub fn tools(&self) -> &Arc<ToolRegistry> {
        &self.tools
    }

    /// Max steps default.
    #[must_use]
    pub const fn max_steps(&self) -> usize {
        self.definition.max_steps
    }
}
