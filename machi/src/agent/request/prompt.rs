//! Non-streaming prompt request handling.
//!
//! This module provides the [`PromptRequest`] builder for creating and executing
//! non-streaming agent prompts with support for multi-turn conversations,
//! chat history, and execution hooks.

use std::{
    future::IntoFuture,
    marker::PhantomData,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use futures::{StreamExt, stream};
use tracing::{Instrument, info_span, span::Id};

use crate::{
    completion::{
        Completion, CompletionModel, Message, PromptError, Usage,
        message::{AssistantContent, UserContent},
    },
    core::wasm_compat::WasmBoxedFuture,
    core::{OneOrMany, json_utils},
    tool::ToolSetError,
};

use super::super::{Agent, PromptHook, ToolCallHookAction};

/// Marker trait for prompt request types.
pub trait PromptType {}

/// Standard prompt request returning a string.
pub struct Standard;

/// Extended prompt request returning detailed response with usage.
pub struct Extended;

impl PromptType for Standard {}
impl PromptType for Extended {}

/// A builder for creating prompt requests with customizable options.
///
/// Uses generics to track which options have been set during the build process.
/// If you expect to continuously call tools, use `.multi_turn()` to set the
/// maximum depth, as the default is 0 (single tool round-trip).
///
/// # Type Parameters
/// - `'a`: Lifetime of the agent reference
/// - `S`: State type (Standard or Extended)
/// - `M`: Completion model type
/// - `P`: Hook type implementing PromptHook
pub struct PromptRequest<'a, S, M, P>
where
    S: PromptType,
    M: CompletionModel,
    P: PromptHook<M>,
{
    /// The prompt message to send.
    prompt: Message,
    /// Optional chat history.
    chat_history: Option<&'a mut Vec<Message>>,
    /// Maximum depth for multi-turn conversations.
    max_depth: usize,
    /// The agent to use for execution.
    agent: &'a Agent<M>,
    /// Phantom data for state tracking.
    state: PhantomData<S>,
    /// Optional execution hook.
    hook: Option<P>,
    /// Tool execution concurrency level.
    concurrency: usize,
}

impl<'a, M> PromptRequest<'a, Standard, M, ()>
where
    M: CompletionModel,
{
    /// Creates a new prompt request.
    pub fn new(agent: &'a Agent<M>, prompt: impl Into<Message>) -> Self {
        Self {
            prompt: prompt.into(),
            chat_history: None,
            max_depth: agent.default_max_depth.unwrap_or_default(),
            agent,
            state: PhantomData,
            hook: None,
            concurrency: 1,
        }
    }
}

impl<'a, S, M, P> PromptRequest<'a, S, M, P>
where
    S: PromptType,
    M: CompletionModel,
    P: PromptHook<M>,
{
    /// Enables extended response details including token usage.
    ///
    /// Changes the return type from `String` to `PromptResponse`.
    pub fn extended_details(self) -> PromptRequest<'a, Extended, M, P> {
        PromptRequest {
            prompt: self.prompt,
            chat_history: self.chat_history,
            max_depth: self.max_depth,
            agent: self.agent,
            state: PhantomData,
            hook: self.hook,
            concurrency: self.concurrency,
        }
    }

    /// Sets the maximum depth for multi-turn conversations.
    ///
    /// If exceeded, returns [`PromptError::MaxDepthError`].
    pub fn multi_turn(self, depth: usize) -> PromptRequest<'a, S, M, P> {
        PromptRequest {
            prompt: self.prompt,
            chat_history: self.chat_history,
            max_depth: depth,
            agent: self.agent,
            state: PhantomData,
            hook: self.hook,
            concurrency: self.concurrency,
        }
    }

    /// Sets the tool execution concurrency level.
    pub fn with_tool_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Adds chat history to the request.
    pub fn with_history(self, history: &'a mut Vec<Message>) -> PromptRequest<'a, S, M, P> {
        PromptRequest {
            prompt: self.prompt,
            chat_history: Some(history),
            max_depth: self.max_depth,
            agent: self.agent,
            state: PhantomData,
            hook: self.hook,
            concurrency: self.concurrency,
        }
    }

    /// Attaches an execution hook.
    pub fn with_hook<P2>(self, hook: P2) -> PromptRequest<'a, S, M, P2>
    where
        P2: PromptHook<M>,
    {
        PromptRequest {
            prompt: self.prompt,
            chat_history: self.chat_history,
            max_depth: self.max_depth,
            agent: self.agent,
            state: PhantomData,
            hook: Some(hook),
            concurrency: self.concurrency,
        }
    }
}

/// Signal for cancelling agent execution from hooks.
///
/// Use `cancel()` to terminate the agent loop early, optionally with a reason
/// via `cancel_with_reason()`.
pub struct CancelSignal {
    sig: Arc<AtomicBool>,
    reason: Arc<OnceLock<String>>,
}

impl CancelSignal {
    pub(crate) fn new() -> Self {
        Self {
            sig: Arc::new(AtomicBool::new(false)),
            reason: Arc::new(OnceLock::new()),
        }
    }

    /// Cancels the agent execution.
    pub fn cancel(&self) {
        self.sig.store(true, Ordering::SeqCst);
    }

    /// Cancels with a reason message.
    pub fn cancel_with_reason(&self, reason: &str) {
        let _ = self.reason.set(reason.to_string());
        self.cancel();
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.sig.load(Ordering::SeqCst)
    }

    pub(crate) fn cancel_reason(&self) -> Option<&str> {
        self.reason.get().map(String::as_str)
    }
}

impl Clone for CancelSignal {
    fn clone(&self) -> Self {
        Self {
            sig: self.sig.clone(),
            reason: self.reason.clone(),
        }
    }
}

impl<'a, M, P> IntoFuture for PromptRequest<'a, Standard, M, P>
where
    M: CompletionModel,
    P: PromptHook<M> + 'static,
{
    type Output = Result<String, PromptError>;
    type IntoFuture = WasmBoxedFuture<'a, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}

impl<'a, M, P> IntoFuture for PromptRequest<'a, Extended, M, P>
where
    M: CompletionModel,
    P: PromptHook<M> + 'static,
{
    type Output = Result<PromptResponse, PromptError>;
    type IntoFuture = WasmBoxedFuture<'a, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}

impl<M, P> PromptRequest<'_, Standard, M, P>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    async fn send(self) -> Result<String, PromptError> {
        self.extended_details().send().await.map(|resp| resp.output)
    }
}

/// Response from a prompt request with extended details.
#[derive(Debug, Clone)]
pub struct PromptResponse {
    /// The text output from the agent.
    pub output: String,
    /// Aggregated token usage across all turns.
    pub total_usage: Usage,
}

impl PromptResponse {
    /// Creates a new prompt response.
    pub fn new(output: impl Into<String>, total_usage: Usage) -> Self {
        Self {
            output: output.into(),
            total_usage,
        }
    }
}

impl<M, P> PromptRequest<'_, Extended, M, P>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    async fn send(self) -> Result<PromptResponse, PromptError> {
        let agent_span = if tracing::Span::current().is_disabled() {
            info_span!(
                "invoke_agent",
                gen_ai.operation.name = "invoke_agent",
                gen_ai.agent.name = self.agent.name(),
                gen_ai.system_instructions = self.agent.preamble,
                gen_ai.prompt = tracing::field::Empty,
                gen_ai.completion = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let agent = self.agent;
        let chat_history = if let Some(history) = self.chat_history {
            history.push(self.prompt.clone());
            history
        } else {
            &mut vec![self.prompt.clone()]
        };

        if let Some(text) = self.prompt.rag_text() {
            agent_span.record("gen_ai.prompt", text);
        }

        let cancel_sig = CancelSignal::new();
        let mut current_max_depth = 0;
        let mut usage = Usage::new();
        let current_span_id: AtomicU64 = AtomicU64::new(0);

        let last_prompt = loop {
            let prompt = chat_history
                .last()
                .cloned()
                .expect("chat history should never be empty");

            if current_max_depth > self.max_depth + 1 {
                break prompt;
            }

            current_max_depth += 1;

            if self.max_depth > 1 {
                tracing::info!(
                    "Current conversation depth: {}/{}",
                    current_max_depth,
                    self.max_depth
                );
            }

            // Call pre-completion hook
            if let Some(ref hook) = self.hook {
                hook.on_completion_call(
                    &prompt,
                    &chat_history[..chat_history.len() - 1],
                    cancel_sig.clone(),
                )
                .await;

                if cancel_sig.is_cancelled() {
                    return Err(PromptError::prompt_cancelled(
                        chat_history.clone(),
                        cancel_sig.cancel_reason().unwrap_or("<no reason given>"),
                    ));
                }
            }

            let span = tracing::Span::current();
            let chat_span = info_span!(
                target: "crate::agent_chat",
                parent: &span,
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.agent.name = self.agent.name(),
                gen_ai.system_instructions = self.agent.preamble,
                gen_ai.provider.name = tracing::field::Empty,
                gen_ai.request.model = tracing::field::Empty,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = tracing::field::Empty,
                gen_ai.output.messages = tracing::field::Empty,
            );

            let chat_span = if current_span_id.load(Ordering::SeqCst) != 0 {
                let id = Id::from_u64(current_span_id.load(Ordering::SeqCst));
                chat_span.follows_from(id).to_owned()
            } else {
                chat_span
            };

            if let Some(id) = chat_span.id() {
                current_span_id.store(id.into_u64(), Ordering::SeqCst);
            }

            let resp = agent
                .completion(
                    prompt.clone(),
                    chat_history[..chat_history.len() - 1].to_vec(),
                )
                .await?
                .send()
                .instrument(chat_span.clone())
                .await?;

            usage += resp.usage;

            // Call post-completion hook
            if let Some(ref hook) = self.hook {
                hook.on_completion_response(&prompt, &resp, cancel_sig.clone())
                    .await;

                if cancel_sig.is_cancelled() {
                    return Err(PromptError::prompt_cancelled(
                        chat_history.clone(),
                        cancel_sig.cancel_reason().unwrap_or("<no reason given>"),
                    ));
                }
            }

            let (tool_calls, texts): (Vec<_>, Vec<_>) = resp
                .choice
                .iter()
                .partition(|choice| matches!(choice, AssistantContent::ToolCall(_)));

            chat_history.push(Message::Assistant {
                id: None,
                content: resp.choice.clone(),
            });

            // If no tool calls, return the text response
            if tool_calls.is_empty() {
                let merged_texts = texts
                    .into_iter()
                    .filter_map(|content| {
                        if let AssistantContent::Text(text) = content {
                            Some(text.text.clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                if self.max_depth > 1 {
                    tracing::info!("Depth reached: {}/{}", current_max_depth, self.max_depth);
                }

                agent_span.record("gen_ai.completion", &merged_texts);
                agent_span.record("gen_ai.usage.input_tokens", usage.input_tokens);
                agent_span.record("gen_ai.usage.output_tokens", usage.output_tokens);

                return Ok(PromptResponse::new(merged_texts, usage));
            }

            // Execute tool calls
            let hook = self.hook.clone();
            let tool_calls: Vec<AssistantContent> = tool_calls.into_iter().cloned().collect();

            let tool_content = stream::iter(tool_calls)
                .map(|choice| {
                    let hook1 = hook.clone();
                    let hook2 = hook.clone();
                    let cancel_sig1 = cancel_sig.clone();
                    let cancel_sig2 = cancel_sig.clone();

                    let tool_span = info_span!(
                        "execute_tool",
                        gen_ai.operation.name = "execute_tool",
                        gen_ai.tool.type = "function",
                        gen_ai.tool.name = tracing::field::Empty,
                        gen_ai.tool.call.id = tracing::field::Empty,
                        gen_ai.tool.call.arguments = tracing::field::Empty,
                        gen_ai.tool.call.result = tracing::field::Empty
                    );

                    let tool_span = if current_span_id.load(Ordering::SeqCst) != 0 {
                        let id = Id::from_u64(current_span_id.load(Ordering::SeqCst));
                        tool_span.follows_from(id).to_owned()
                    } else {
                        tool_span
                    };

                    if let Some(id) = tool_span.id() {
                        current_span_id.store(id.into_u64(), Ordering::SeqCst);
                    }

                    async move {
                        if let AssistantContent::ToolCall(tool_call) = choice {
                            let tool_name = &tool_call.function.name;
                            let args =
                                json_utils::value_to_json_string(&tool_call.function.arguments);

                            let tool_span = tracing::Span::current();
                            tool_span.record("gen_ai.tool.name", tool_name);
                            tool_span.record("gen_ai.tool.call.id", &tool_call.id);
                            tool_span.record("gen_ai.tool.call.arguments", &args);

                            // Call pre-tool hook
                            if let Some(hook) = hook1 {
                                let action = hook
                                    .on_tool_call(
                                        tool_name,
                                        tool_call.call_id.clone(),
                                        &args,
                                        cancel_sig1.clone(),
                                    )
                                    .await;

                                if cancel_sig1.is_cancelled() {
                                    return Err(ToolSetError::Interrupted);
                                }

                                if let ToolCallHookAction::Skip { reason } = action {
                                    tracing::info!(
                                        tool_name = tool_name,
                                        reason = reason,
                                        "Tool call rejected"
                                    );

                                    return if let Some(call_id) = tool_call.call_id.clone() {
                                        Ok(UserContent::tool_result_with_call_id(
                                            tool_call.id.clone(),
                                            call_id,
                                            OneOrMany::one(reason.into()),
                                        ))
                                    } else {
                                        Ok(UserContent::tool_result(
                                            tool_call.id.clone(),
                                            OneOrMany::one(reason.into()),
                                        ))
                                    };
                                }
                            }

                            // Execute tool
                            let output =
                                match agent.tool_server_handle.call_tool(tool_name, &args).await {
                                    Ok(res) => res,
                                    Err(e) => {
                                        tracing::warn!("Error while executing tool: {e}");
                                        e.to_string()
                                    }
                                };

                            // Call post-tool hook
                            if let Some(hook) = hook2 {
                                hook.on_tool_result(
                                    tool_name,
                                    tool_call.call_id.clone(),
                                    &args,
                                    &output,
                                    cancel_sig2.clone(),
                                )
                                .await;

                                if cancel_sig2.is_cancelled() {
                                    return Err(ToolSetError::Interrupted);
                                }
                            }

                            tool_span.record("gen_ai.tool.call.result", &output);
                            tracing::info!(
                                "executed tool {tool_name} with args {args}. result: {output}"
                            );

                            if let Some(call_id) = tool_call.call_id.clone() {
                                Ok(UserContent::tool_result_with_call_id(
                                    tool_call.id.clone(),
                                    call_id,
                                    OneOrMany::one(output.into()),
                                ))
                            } else {
                                Ok(UserContent::tool_result(
                                    tool_call.id.clone(),
                                    OneOrMany::one(output.into()),
                                ))
                            }
                        } else {
                            unreachable!("filtered for ToolCall only")
                        }
                    }
                    .instrument(tool_span)
                })
                .buffer_unordered(self.concurrency)
                .collect::<Vec<Result<UserContent, ToolSetError>>>()
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| {
                    if matches!(e, ToolSetError::Interrupted) {
                        PromptError::prompt_cancelled(
                            chat_history.clone(),
                            cancel_sig.cancel_reason().unwrap_or("<no reason given>"),
                        )
                    } else {
                        e.into()
                    }
                })?;

            chat_history.push(Message::User {
                content: OneOrMany::many(tool_content).expect("at least one tool call"),
            });
        };

        Err(PromptError::MaxDepthError {
            max_depth: self.max_depth,
            chat_history: Box::new(chat_history.clone()),
            prompt: Box::new(last_prompt),
        })
    }
}
