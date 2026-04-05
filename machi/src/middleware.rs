//! Middleware pipeline for intercepting and modifying agent execution.
//!
//! Unlike [`Hooks`](crate::hooks::Hooks) which are purely observational,
//! middleware can **intercept** and **modify** the execution flow — rejecting
//! tool calls, transforming LLM requests, caching responses, enforcing rate
//! limits, etc.
//!
//! # Design
//!
//! Middleware methods use [`ControlFlow`] to signal whether execution should
//! continue or short-circuit:
//!
//! - `ControlFlow::Continue(action)` — proceed (optionally with a modified action).
//! - `ControlFlow::Break(error)` — abort the current operation with an error.
//!
//! Multiple middleware are executed in pipeline order (first-added runs first
//! for requests, last-added runs first for responses).
//!
//! # Examples
//!
//! ```rust
//! use std::ops::ControlFlow;
//! use async_trait::async_trait;
//! use machi::middleware::{Middleware, MiddlewareContext};
//! use machi::agent::ToolCallRequest;
//! use machi::middleware::ToolCallAction;
//! use machi::tool::ToolError;
//!
//! /// Middleware that blocks any tool with "dangerous" in its name.
//! struct BlockDangerousTools;
//!
//! #[async_trait]
//! impl Middleware for BlockDangerousTools {
//!     async fn on_tool_call(
//!         &self,
//!         _ctx: &MiddlewareContext,
//!         call: &ToolCallRequest,
//!     ) -> ControlFlow<ToolError, ToolCallAction> {
//!         if call.name.contains("dangerous") {
//!             ControlFlow::Break(ToolError::Forbidden(call.name.clone()))
//!         } else {
//!             ControlFlow::Continue(ToolCallAction::Execute)
//!         }
//!     }
//! }
//! ```

use std::ops::ControlFlow;

use async_trait::async_trait;
use serde_json::Value;

use crate::agent::result::ToolCallRequest;
use crate::error::Error;
use crate::hooks::RunContext;
use crate::message::Message;
use crate::tool::ToolError;

/// A shared, thread-safe [`Middleware`] trait object.
pub type SharedMiddleware = std::sync::Arc<dyn Middleware>;

/// Context passed to middleware methods.
///
/// Provides read access to the current run state. Middleware can inspect
/// the agent name, step number, and accumulated usage.
#[derive(Debug, Clone)]
pub struct MiddlewareContext {
    /// The underlying run context.
    pub run_context: RunContext,
    /// Name of the agent currently executing.
    pub agent_name: String,
}

impl MiddlewareContext {
    /// Create a new middleware context.
    #[must_use]
    pub const fn new(run_context: RunContext, agent_name: String) -> Self {
        Self {
            run_context,
            agent_name,
        }
    }
}

/// Action to take after `on_tool_call` middleware processing.
#[derive(Debug, Clone)]
pub enum ToolCallAction {
    /// Execute the tool call normally.
    Execute,
    /// Skip this tool call entirely (return a synthetic result).
    Skip {
        /// The synthetic result to return instead of executing the tool.
        result: String,
    },
    /// Replace the tool call with a pre-computed result.
    Replace {
        /// The replacement result value.
        result: Value,
    },
}

/// Middleware trait for intercepting and modifying agent execution.
///
/// All methods have default no-op implementations — override only the
/// events you need to intercept.
///
/// # Execution Order
///
/// For a middleware stack `[A, B, C]`:
/// - **Pre-execution** hooks (`on_tool_call`, `on_llm_request`) run in order: A → B → C
/// - **Post-execution** hooks (`on_tool_result`, `on_llm_response`) run in reverse: C → B → A
///
/// This follows the "onion" pattern common in HTTP middleware.
#[async_trait]
pub trait Middleware: Send + Sync {
    /// Called before a tool is executed.
    ///
    /// Return `ControlFlow::Continue(ToolCallAction::Execute)` to proceed,
    /// `ControlFlow::Continue(ToolCallAction::Skip { .. })` to skip with a
    /// synthetic result, or `ControlFlow::Break(error)` to abort.
    async fn on_tool_call(
        &self,
        _ctx: &MiddlewareContext,
        _call: &ToolCallRequest,
    ) -> ControlFlow<ToolError, ToolCallAction> {
        ControlFlow::Continue(ToolCallAction::Execute)
    }

    /// Called after a tool produces a result.
    ///
    /// The `result` string can be inspected or modified. Return
    /// `ControlFlow::Break(error)` to replace the result with an error.
    async fn on_tool_result(
        &self,
        _ctx: &MiddlewareContext,
        _tool_name: &str,
        _result: &str,
        _success: bool,
    ) -> ControlFlow<ToolError> {
        ControlFlow::Continue(())
    }

    /// Called before the message list is sent to the LLM.
    ///
    /// Middleware can inspect and log the messages. Return
    /// `ControlFlow::Break(error)` to abort the LLM call.
    async fn on_llm_request(
        &self,
        _ctx: &MiddlewareContext,
        _messages: &[Message],
    ) -> ControlFlow<Error> {
        ControlFlow::Continue(())
    }

    /// Called after the LLM returns a response.
    ///
    /// Middleware can inspect and log the response text. Return
    /// `ControlFlow::Break(error)` to discard the response.
    async fn on_llm_response(
        &self,
        _ctx: &MiddlewareContext,
        _response_text: Option<&str>,
    ) -> ControlFlow<Error> {
        ControlFlow::Continue(())
    }

    /// Called when the agent run starts.
    async fn on_agent_start(&self, _ctx: &MiddlewareContext) -> ControlFlow<Error> {
        ControlFlow::Continue(())
    }

    /// Called when the agent run ends (successfully).
    async fn on_agent_end(&self, _ctx: &MiddlewareContext, _output: &Value) -> ControlFlow<Error> {
        ControlFlow::Continue(())
    }
}

/// Run a tool call through a middleware pipeline.
///
/// Returns the action to take, or an error if any middleware short-circuits.
pub(crate) async fn run_tool_call_middleware(
    middleware: &[SharedMiddleware],
    ctx: &MiddlewareContext,
    call: &ToolCallRequest,
) -> Result<ToolCallAction, ToolError> {
    for mw in middleware {
        match mw.on_tool_call(ctx, call).await {
            ControlFlow::Continue(action) => {
                if !matches!(action, ToolCallAction::Execute) {
                    return Ok(action);
                }
            }
            ControlFlow::Break(err) => return Err(err),
        }
    }
    Ok(ToolCallAction::Execute)
}

/// Run post-tool-result middleware in reverse order.
pub(crate) async fn run_tool_result_middleware(
    middleware: &[SharedMiddleware],
    ctx: &MiddlewareContext,
    tool_name: &str,
    result: &str,
    success: bool,
) -> Result<(), ToolError> {
    for mw in middleware.iter().rev() {
        if let ControlFlow::Break(err) = mw.on_tool_result(ctx, tool_name, result, success).await {
            return Err(err);
        }
    }
    Ok(())
}

/// Run pre-LLM-request middleware in order.
pub(crate) async fn run_llm_request_middleware(
    middleware: &[SharedMiddleware],
    ctx: &MiddlewareContext,
    messages: &[Message],
) -> Result<(), Error> {
    for mw in middleware {
        if let ControlFlow::Break(err) = mw.on_llm_request(ctx, messages).await {
            return Err(err);
        }
    }
    Ok(())
}
