//! Agent builder.

use std::sync::Arc;

use machi_tools::{SharedTool, ToolRegistry};
use machi_types::{ErrorCode, MachiError};

use crate::definition::{AgentDefinition, CompletionRequirement, Instructions};
use crate::instance::Agent;

/// Builds a validated [`Agent`].
#[derive(Default)]
pub struct AgentBuilder {
    definition: Option<AgentDefinition>,
    tools: Vec<SharedTool>,
}

impl std::fmt::Debug for AgentBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentBuilder")
            .field("definition", &self.definition)
            .field("tools", &self.tools.len())
            .finish()
    }
}

impl AgentBuilder {
    /// Empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Start from a definition.
    #[must_use]
    pub fn from_definition(definition: AgentDefinition) -> Self {
        Self {
            definition: Some(definition),
            tools: Vec::new(),
        }
    }

    /// Programmatic minimal definition.
    #[must_use]
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            definition: Some(AgentDefinition::new(name)),
            tools: Vec::new(),
        }
    }

    /// Set instructions.
    #[must_use]
    pub fn instructions(mut self, instructions: impl Into<Instructions>) -> Self {
        if let Some(def) = &mut self.definition {
            def.instructions = instructions.into();
        }
        self
    }

    /// Set model.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        if let Some(def) = &mut self.definition {
            def.model = model.into();
        }
        self
    }

    /// Set description.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        if let Some(def) = &mut self.definition {
            def.description = description.into();
        }
        self
    }

    /// Set max steps.
    #[must_use]
    pub fn max_steps(mut self, max_steps: usize) -> Self {
        if let Some(def) = &mut self.definition {
            def.max_steps = max_steps;
        }
        self
    }

    /// Attach tools available to the agent (filtered by definition policy).
    #[must_use]
    pub fn tools(mut self, tools: Vec<SharedTool>) -> Self {
        self.tools = tools;
        self
    }

    /// Require a named tool call before the turn may complete.
    #[must_use]
    pub fn completion(mut self, requirement: CompletionRequirement) -> Self {
        if let Some(def) = &mut self.definition {
            def.completion = Some(requirement);
        }
        self
    }

    /// Require structured JSON output matching a JSON Schema object.
    #[must_use]
    pub fn output_schema(mut self, schema: serde_json::Value) -> Self {
        if let Some(def) = &mut self.definition {
            def.output_schema = Some(schema);
        }
        self
    }

    /// Build the agent instance.
    ///
    /// # Errors
    ///
    /// Returns validation or build errors.
    pub fn build(self) -> Result<Agent, MachiError> {
        let definition = self.definition.ok_or_else(|| {
            MachiError::new(ErrorCode::AgentBuild, "agent definition is required")
        })?;
        definition.validate()?;

        // Definition-level allowed_tools / denylist applied at resolution (W5.4).
        let filtered: Vec<SharedTool> = self
            .tools
            .into_iter()
            .filter(|t| definition.tools.admits(t.name()))
            .collect();

        let system_prompt = definition.instructions.resolve();
        let tools = Arc::new(ToolRegistry::from_tools(filtered));
        Ok(Agent::new(definition, system_prompt, tools))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_name() {
        let err = AgentBuilder::named("  ").build().expect_err("empty");
        assert_eq!(err.code(), ErrorCode::AgentInvalidDefinition);
    }

    #[test]
    fn builds_minimal() {
        let agent = AgentBuilder::named("assistant")
            .instructions("You are helpful.")
            .model("mock")
            .build()
            .expect("build");
        assert_eq!(agent.name(), "assistant");
        assert_eq!(agent.system_prompt(), "You are helpful.");
    }

    #[test]
    fn allowed_tools_filtered_at_build() {
        use std::sync::Arc;

        use machi_tools::CalcTool;

        let mut def = AgentDefinition::new("x");
        def.tools = crate::definition::ToolPolicy::Allowlist(vec!["calc".into()]);
        def.model = "m".into();
        // Two calc instances under different names aren't available; policy drops non-calc.
        // Empty allowlist of "other" leaves no tools.
        let agent_empty = AgentBuilder::from_definition({
            let mut d = def.clone();
            d.tools = crate::definition::ToolPolicy::Allowlist(vec!["other".into()]);
            d
        })
        .tools(vec![Arc::new(CalcTool)])
        .build()
        .expect("build");
        assert!(agent_empty.tools().is_empty());

        let agent = AgentBuilder::from_definition(def)
            .tools(vec![Arc::new(CalcTool)])
            .build()
            .expect("build");
        assert_eq!(agent.tools().names(), vec!["calc".to_owned()]);
    }
}
