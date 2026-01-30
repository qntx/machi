//! Implementation of the Tool trait for Agent.
//!
//! This module allows agents to be used as tools within other agents,
//! enabling hierarchical agent workflows and sub-agent patterns.

use crate::{
    completion::{CompletionModel, Prompt, PromptError, ToolDefinition},
    tool::Tool,
};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};

use super::Agent;

/// Arguments for calling an agent as a tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AgentToolArgs {
    /// The prompt to send to the sub-agent.
    prompt: String,
}

impl<M: CompletionModel> Tool for Agent<M> {
    const NAME: &'static str = "agent_tool";

    type Error = PromptError;
    type Args = AgentToolArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        let description = format!(
            "Prompt a sub-agent to do a task for you.\n\n\
             Agent name: {name}\n\
             Agent description: {description}\n\
             Agent system prompt: {sysprompt}",
            name = self.name(),
            description = self.description.clone().unwrap_or_default(),
            sysprompt = self.preamble.clone().unwrap_or_default()
        );

        ToolDefinition {
            name: <Self as Tool>::name(self),
            description,
            parameters: serde_json::to_value(schema_for!(AgentToolArgs))
                .expect("converting JSON schema to JSON value should never fail"),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.prompt(args.prompt).await
    }

    fn name(&self) -> String {
        self.name.clone().unwrap_or_else(|| Self::NAME.to_string())
    }
}
