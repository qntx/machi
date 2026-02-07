//! Hook dispatch bridge for the agent execution engine.
//!
//! [`HookPair`] combines run-level [`RunHooks`] and per-agent [`AgentHooks`]
//! into a single dispatcher, firing both layers concurrently via
//! [`tokio::join!`] — mirroring the OpenAI Agents SDK's `asyncio.gather`
//! pattern for parallel hook execution.

use serde_json::Value;

use crate::callback::{AgentHooks, RunContext, RunHooks};
use crate::chat::ChatResponse;
use crate::error::Error;
use crate::message::Message;

/// Dispatches lifecycle events to both run-level and agent-level hooks.
///
/// At each event point in the agent loop, two hooks fire concurrently:
/// a global [`RunHooks`] and an optional per-agent [`AgentHooks`].
/// Both layers are invoked in parallel via `tokio::join!`.
pub(super) struct HookPair<'a> {
    run: &'a dyn RunHooks,
    agent: Option<&'a dyn AgentHooks>,
    name: &'a str,
}

impl<'a> HookPair<'a> {
    /// Create a new hook pair from run-level hooks, optional agent hooks,
    /// and the agent name used for run-hook dispatch.
    pub fn new(run: &'a dyn RunHooks, agent: Option<&'a dyn AgentHooks>, name: &'a str) -> Self {
        Self { run, agent, name }
    }

    pub async fn agent_start(&self, ctx: &RunContext) {
        if let Some(ah) = self.agent {
            tokio::join!(self.run.on_agent_start(ctx, self.name), ah.on_start(ctx));
        } else {
            self.run.on_agent_start(ctx, self.name).await;
        }
    }

    pub async fn agent_end(&self, ctx: &RunContext, output: &Value) {
        if let Some(ah) = self.agent {
            tokio::join!(
                self.run.on_agent_end(ctx, self.name, output),
                ah.on_end(ctx, output)
            );
        } else {
            self.run.on_agent_end(ctx, self.name, output).await;
        }
    }

    pub async fn llm_start(&self, ctx: &RunContext, system: Option<&str>, msgs: &[Message]) {
        if let Some(ah) = self.agent {
            tokio::join!(
                self.run.on_llm_start(ctx, self.name, system, msgs),
                ah.on_llm_start(ctx, system, msgs)
            );
        } else {
            self.run.on_llm_start(ctx, self.name, system, msgs).await;
        }
    }

    pub async fn llm_end(&self, ctx: &RunContext, response: &ChatResponse) {
        if let Some(ah) = self.agent {
            tokio::join!(
                self.run.on_llm_end(ctx, self.name, response),
                ah.on_llm_end(ctx, response)
            );
        } else {
            self.run.on_llm_end(ctx, self.name, response).await;
        }
    }

    pub async fn tool_start(&self, ctx: &RunContext, tool_name: &str) {
        if let Some(ah) = self.agent {
            tokio::join!(
                self.run.on_tool_start(ctx, self.name, tool_name),
                ah.on_tool_start(ctx, tool_name)
            );
        } else {
            self.run.on_tool_start(ctx, self.name, tool_name).await;
        }
    }

    pub async fn tool_end(&self, ctx: &RunContext, tool_name: &str, result: &str) {
        if let Some(ah) = self.agent {
            tokio::join!(
                self.run.on_tool_end(ctx, self.name, tool_name, result),
                ah.on_tool_end(ctx, tool_name, result)
            );
        } else {
            self.run
                .on_tool_end(ctx, self.name, tool_name, result)
                .await;
        }
    }

    pub async fn error(&self, ctx: &RunContext, err: &Error) {
        if let Some(ah) = self.agent {
            tokio::join!(
                self.run.on_error(ctx, self.name, err),
                ah.on_error(ctx, err)
            );
        } else {
            self.run.on_error(ctx, self.name, err).await;
        }
    }
}
