//! Hook traits for observing agent execution events.
//!
//! This module provides traits for implementing hooks that can observe and
//! react to various events during agent execution, such as completion calls,
//! tool invocations, and streaming events.

use std::future::Future;

use crate::{
    completion::{CompletionModel, CompletionResponse, Message},
    core::wasm_compat::{WasmCompatSend, WasmCompatSync},
};

use super::CancelSignal;

/// Control flow action for tool call hooks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCallHookAction {
    /// Continue tool execution as normal.
    Continue,
    /// Skip tool execution and return the provided reason as the tool result.
    Skip { reason: String },
}

/// Trait for per-request hooks to observe tool call events in non-streaming mode.
///
/// Implement this trait to receive callbacks during agent execution.
/// All methods have default no-op implementations, so you only need to
/// implement the methods you're interested in.
pub trait PromptHook<M>: Clone + WasmCompatSend + WasmCompatSync
where
    M: CompletionModel,
{
    /// Called before the prompt is sent to the model.
    #[allow(unused_variables)]
    fn on_completion_call(
        &self,
        prompt: &Message,
        history: &[Message],
        cancel_sig: CancelSignal,
    ) -> impl Future<Output = ()> + WasmCompatSend {
        async {}
    }

    /// Called after the prompt is sent to the model and a response is received.
    #[allow(unused_variables)]
    fn on_completion_response(
        &self,
        prompt: &Message,
        response: &CompletionResponse<M::Response>,
        cancel_sig: CancelSignal,
    ) -> impl Future<Output = ()> + WasmCompatSend {
        async {}
    }

    /// Called before a tool is invoked.
    ///
    /// # Returns
    /// - `ToolCallHookAction::Continue` - Allow tool execution to proceed
    /// - `ToolCallHookAction::Skip { reason }` - Reject tool execution;
    ///   `reason` will be returned to the LLM as the tool result
    #[allow(unused_variables)]
    fn on_tool_call(
        &self,
        tool_name: &str,
        tool_call_id: Option<String>,
        args: &str,
        cancel_sig: CancelSignal,
    ) -> impl Future<Output = ToolCallHookAction> + WasmCompatSend {
        async { ToolCallHookAction::Continue }
    }

    /// Called after a tool is invoked (and a result has been returned).
    #[allow(unused_variables)]
    fn on_tool_result(
        &self,
        tool_name: &str,
        tool_call_id: Option<String>,
        args: &str,
        result: &str,
        cancel_sig: CancelSignal,
    ) -> impl Future<Output = ()> + WasmCompatSend {
        async {}
    }
}

/// Default implementation for unit type, allowing no-hook usage.
impl<M> PromptHook<M> for () where M: CompletionModel {}

/// Trait for per-request hooks to observe tool call events in streaming mode.
///
/// This trait extends the non-streaming hook with additional callbacks
/// for streaming-specific events like text deltas and tool call deltas.
pub trait StreamingPromptHook<M>: Clone + Send + Sync
where
    M: CompletionModel,
{
    /// Called before the prompt is sent to the model.
    #[allow(unused_variables)]
    fn on_completion_call(
        &self,
        prompt: &Message,
        history: &[Message],
        cancel_sig: CancelSignal,
    ) -> impl Future<Output = ()> + Send {
        async {}
    }

    /// Called when receiving a text delta during streaming.
    #[allow(unused_variables)]
    fn on_text_delta(
        &self,
        text_delta: &str,
        aggregated_text: &str,
        cancel_sig: CancelSignal,
    ) -> impl Future<Output = ()> + Send {
        async {}
    }

    /// Called when receiving a tool call delta during streaming.
    ///
    /// `tool_name` is `Some` on the first delta for a tool call, `None` on subsequent deltas.
    #[allow(unused_variables)]
    fn on_tool_call_delta(
        &self,
        tool_call_id: &str,
        tool_name: Option<&str>,
        tool_call_delta: &str,
        cancel_sig: CancelSignal,
    ) -> impl Future<Output = ()> + Send {
        async {}
    }

    /// Called after the model provider has finished streaming a text response.
    #[allow(unused_variables)]
    fn on_stream_completion_response_finish(
        &self,
        prompt: &Message,
        response: &<M as CompletionModel>::StreamingResponse,
        cancel_sig: CancelSignal,
    ) -> impl Future<Output = ()> + Send {
        async {}
    }

    /// Called before a tool is invoked.
    ///
    /// # Returns
    /// - [`ToolCallHookAction::Continue`] - Allow tool execution to proceed
    /// - [`ToolCallHookAction::Skip`] - Reject tool execution;
    ///   `reason` will be returned to the LLM as the tool result
    #[allow(unused_variables)]
    fn on_tool_call(
        &self,
        tool_name: &str,
        tool_call_id: Option<String>,
        args: &str,
        cancel_sig: CancelSignal,
    ) -> impl Future<Output = ToolCallHookAction> + Send {
        async { ToolCallHookAction::Continue }
    }

    /// Called after a tool is invoked (and a result has been returned).
    #[allow(unused_variables)]
    fn on_tool_result(
        &self,
        tool_name: &str,
        tool_call_id: Option<String>,
        args: &str,
        result: &str,
        cancel_sig: CancelSignal,
    ) -> impl Future<Output = ()> + Send {
        async {}
    }
}

/// Default implementation for unit type, allowing no-hook usage.
impl<M> StreamingPromptHook<M> for () where M: CompletionModel {}
