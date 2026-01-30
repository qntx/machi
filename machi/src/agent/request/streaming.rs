//! Streaming prompt request handling.
//!
//! This module provides the [`StreamingPromptRequest`] builder for creating
//! and executing streaming agent prompts with real-time token delivery.

use std::{pin::Pin, sync::Arc};

use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::info_span;
use tracing_futures::Instrument;

use crate::{
    completion::{
        CompletionModel, GetTokenUsage, PromptError,
        message::{
            AssistantContent, Message, Reasoning, Text, ToolResult, ToolResultContent, UserContent,
        },
        streaming::{StreamedAssistantContent, StreamedUserContent, StreamingCompletion},
    },
    core::wasm_compat::{WasmBoxedFuture, WasmCompatSend},
    core::{OneOrMany, json_utils},
};

use super::super::{Agent, StreamingPromptHook, ToolCallHookAction, error::StreamingError};
use super::CancelSignal;

#[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
pub type StreamingResult<R> =
    Pin<Box<dyn Stream<Item = Result<MultiTurnStreamItem<R>, StreamingError>> + Send>>;

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
pub type StreamingResult<R> =
    Pin<Box<dyn Stream<Item = Result<MultiTurnStreamItem<R>, StreamingError>>>>;

/// Items emitted during a multi-turn streaming conversation.
#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "camelCase")]
#[non_exhaustive]
pub enum MultiTurnStreamItem<R> {
    /// A streamed assistant content item.
    StreamAssistantItem(StreamedAssistantContent<R>),
    /// A streamed user content item (mostly for tool results).
    StreamUserItem(StreamedUserContent),
    /// The final result from the stream.
    FinalResponse(FinalResponse),
}

/// Final response from a streaming conversation.
#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FinalResponse {
    response: String,
    aggregated_usage: crate::completion::Usage,
}

impl FinalResponse {
    /// Creates an empty final response.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            response: String::new(),
            aggregated_usage: crate::completion::Usage::new(),
        }
    }

    /// Returns the response text.
    #[must_use]
    #[inline]
    pub fn response(&self) -> &str {
        &self.response
    }

    /// Returns the aggregated token usage.
    #[must_use]
    #[inline]
    pub fn usage(&self) -> &crate::completion::Usage {
        &self.aggregated_usage
    }
}

impl<R> MultiTurnStreamItem<R> {
    pub(crate) fn stream_item(item: StreamedAssistantContent<R>) -> Self {
        Self::StreamAssistantItem(item)
    }

    /// Creates a final response item.
    pub fn final_response(response: &str, aggregated_usage: crate::completion::Usage) -> Self {
        Self::FinalResponse(FinalResponse {
            response: response.to_string(),
            aggregated_usage,
        })
    }
}

/// Helper to build a cancellation error.
#[inline]
fn make_cancel_error(history: Vec<Message>, cancel_sig: &CancelSignal) -> StreamingError {
    StreamingError::Prompt(
        PromptError::prompt_cancelled(
            history,
            cancel_sig.cancel_reason().unwrap_or("<no reason given>"),
        )
        .into(),
    )
}

/// A builder for streaming prompt requests.
///
/// If you expect continuous tool calls, use `.multi_turn()` to set the maximum
/// depth, as the default is 0 (single tool round-trip).
pub struct StreamingPromptRequest<M, P>
where
    M: CompletionModel,
    P: StreamingPromptHook<M> + 'static,
{
    /// The prompt message to send.
    prompt: Message,
    /// Optional chat history.
    chat_history: Option<Vec<Message>>,
    /// Maximum depth for multi-turn conversations.
    max_depth: usize,
    /// The agent to use for execution.
    agent: Arc<Agent<M>>,
    /// Optional execution hook.
    hook: Option<P>,
}

impl<M, P> StreamingPromptRequest<M, P>
where
    M: CompletionModel + 'static,
    <M as CompletionModel>::StreamingResponse: WasmCompatSend + GetTokenUsage,
    P: StreamingPromptHook<M>,
{
    /// Creates a new streaming prompt request.
    pub fn new(agent: Arc<Agent<M>>, prompt: impl Into<Message>) -> Self {
        Self {
            prompt: prompt.into(),
            chat_history: None,
            max_depth: agent.default_max_depth.unwrap_or_default(),
            agent,
            hook: None,
        }
    }

    /// Sets the maximum depth for multi-turn conversations.
    pub fn multi_turn(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Adds chat history to the request.
    pub fn with_history(mut self, history: Vec<Message>) -> Self {
        self.chat_history = Some(history);
        self
    }

    /// Attaches an execution hook.
    pub fn with_hook<P2>(self, hook: P2) -> StreamingPromptRequest<M, P2>
    where
        P2: StreamingPromptHook<M>,
    {
        StreamingPromptRequest {
            prompt: self.prompt,
            chat_history: self.chat_history,
            max_depth: self.max_depth,
            agent: self.agent,
            hook: Some(hook),
        }
    }

    async fn send(self) -> StreamingResult<M::StreamingResponse> {
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

        let prompt = self.prompt;
        if let Some(text) = prompt.rag_text() {
            agent_span.record("gen_ai.prompt", text);
        }

        let agent = self.agent;
        let chat_history = Arc::new(RwLock::new(self.chat_history.unwrap_or_default()));

        let mut current_max_depth = 0;
        let mut last_prompt_error = String::new();
        let mut last_text_response = String::new();
        let mut is_text_response = false;
        let mut max_depth_reached = false;
        let mut aggregated_usage = crate::completion::Usage::new();
        let cancel_sig = CancelSignal::new();

        let stream = async_stream::stream! {
            let mut current_prompt = prompt.clone();
            let mut did_call_tool = false;

            'outer: loop {
                if current_max_depth > self.max_depth + 1 {
                    last_prompt_error = current_prompt.rag_text().unwrap_or_default();
                    max_depth_reached = true;
                    break;
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
                    let history_snapshot = chat_history.read().await.clone();
                    hook.on_completion_call(&current_prompt, &history_snapshot, cancel_sig.clone())
                        .await;

                    if cancel_sig.is_cancelled() {
                        yield Err(make_cancel_error(history_snapshot, &cancel_sig));
                    }
                }

                let chat_stream_span = info_span!(
                    target: "crate::agent_chat",
                    parent: tracing::Span::current(),
                    "chat_streaming",
                    gen_ai.operation.name = "chat",
                    gen_ai.agent.name = &agent.name(),
                    gen_ai.system_instructions = &agent.preamble,
                    gen_ai.provider.name = tracing::field::Empty,
                    gen_ai.request.model = tracing::field::Empty,
                    gen_ai.response.id = tracing::field::Empty,
                    gen_ai.response.model = tracing::field::Empty,
                    gen_ai.usage.output_tokens = tracing::field::Empty,
                    gen_ai.usage.input_tokens = tracing::field::Empty,
                    gen_ai.input.messages = tracing::field::Empty,
                    gen_ai.output.messages = tracing::field::Empty,
                );

                let mut stream = tracing::Instrument::instrument(
                    agent
                        .stream_completion(current_prompt.clone(), (*chat_history.read().await).clone())
                        .await?
                        .stream(),
                    chat_stream_span
                )
                .await?;

                chat_history.write().await.push(current_prompt.clone());

                let mut tool_calls = vec![];
                let mut tool_results = vec![];

                while let Some(content) = stream.next().await {
                    match content {
                        Ok(StreamedAssistantContent::Text(text)) => {
                            if !is_text_response {
                                last_text_response = String::new();
                                is_text_response = true;
                            }
                            last_text_response.push_str(&text.text);

                            if let Some(ref hook) = self.hook {
                                hook.on_text_delta(&text.text, &last_text_response, cancel_sig.clone()).await;
                                if cancel_sig.is_cancelled() {
                                    yield Err(make_cancel_error(chat_history.read().await.clone(), &cancel_sig));
                                }
                            }

                            yield Ok(MultiTurnStreamItem::stream_item(StreamedAssistantContent::Text(text)));
                            did_call_tool = false;
                        }
                        Ok(StreamedAssistantContent::ToolCall(tool_call)) => {
                            let tool_span = info_span!(
                                parent: tracing::Span::current(),
                                "execute_tool",
                                gen_ai.operation.name = "execute_tool",
                                gen_ai.tool.type = "function",
                                gen_ai.tool.name = tracing::field::Empty,
                                gen_ai.tool.call.id = tracing::field::Empty,
                                gen_ai.tool.call.arguments = tracing::field::Empty,
                                gen_ai.tool.call.result = tracing::field::Empty
                            );

                            yield Ok(MultiTurnStreamItem::stream_item(StreamedAssistantContent::ToolCall(tool_call.clone())));

                            let tc_result = async {
                                let tool_span = tracing::Span::current();
                                let tool_args = json_utils::value_to_json_string(&tool_call.function.arguments);

                                if let Some(ref hook) = self.hook {
                                    let action = hook
                                        .on_tool_call(&tool_call.function.name, tool_call.call_id.clone(), &tool_args, cancel_sig.clone())
                                        .await;

                                    if cancel_sig.is_cancelled() {
                                        return Err(make_cancel_error(chat_history.read().await.clone(), &cancel_sig));
                                    }

                                    if let ToolCallHookAction::Skip { reason } = action {
                                        tracing::info!(
                                            tool_name = tool_call.function.name.as_str(),
                                            reason = reason,
                                            "Tool call rejected"
                                        );
                                        let tool_call_msg = AssistantContent::ToolCall(tool_call.clone());
                                        tool_calls.push(tool_call_msg);
                                        tool_results.push((tool_call.id.clone(), tool_call.call_id.clone(), reason.clone()));
                                        did_call_tool = true;
                                        return Ok(reason);
                                    }
                                }

                                tool_span.record("gen_ai.tool.name", &tool_call.function.name);
                                tool_span.record("gen_ai.tool.call.arguments", &tool_args);

                                let tool_result = match agent.tool_server_handle.call_tool(&tool_call.function.name, &tool_args).await {
                                    Ok(thing) => thing,
                                    Err(e) => {
                                        tracing::warn!("Error while calling tool: {e}");
                                        e.to_string()
                                    }
                                };

                                tool_span.record("gen_ai.tool.call.result", &tool_result);

                                if let Some(ref hook) = self.hook {
                                    hook.on_tool_result(&tool_call.function.name, tool_call.call_id.clone(), &tool_args, &tool_result, cancel_sig.clone())
                                        .await;

                                    if cancel_sig.is_cancelled() {
                                        return Err(make_cancel_error(chat_history.read().await.clone(), &cancel_sig));
                                    }
                                }

                                let tool_call_msg = AssistantContent::ToolCall(tool_call.clone());
                                tool_calls.push(tool_call_msg);
                                tool_results.push((tool_call.id.clone(), tool_call.call_id.clone(), tool_result.clone()));

                                did_call_tool = true;
                                Ok(tool_result)
                            }.instrument(tool_span).await;

                            match tc_result {
                                Ok(text) => {
                                    let tr = ToolResult {
                                        id: tool_call.id,
                                        call_id: tool_call.call_id,
                                        content: OneOrMany::one(ToolResultContent::Text(Text { text })),
                                    };
                                    yield Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult(tr)));
                                }
                                Err(e) => {
                                    yield Err(e);
                                }
                            }
                        }
                        Ok(StreamedAssistantContent::ToolCallDelta { id, content }) => {
                            if let Some(ref hook) = self.hook {
                                let (name, delta) = match &content {
                                    crate::completion::streaming::ToolCallDeltaContent::Name(n) => (Some(n.as_str()), ""),
                                    crate::completion::streaming::ToolCallDeltaContent::Delta(d) => (None, d.as_str()),
                                };
                                hook.on_tool_call_delta(&id, name, delta, cancel_sig.clone()).await;

                                if cancel_sig.is_cancelled() {
                                    yield Err(make_cancel_error(chat_history.read().await.clone(), &cancel_sig));
                                }
                            }
                        }
                        Ok(StreamedAssistantContent::Reasoning(Reasoning { reasoning, id, signature })) => {
                            yield Ok(MultiTurnStreamItem::stream_item(StreamedAssistantContent::Reasoning(Reasoning { id, reasoning, signature })));
                            did_call_tool = false;
                        }
                        Ok(StreamedAssistantContent::ReasoningDelta { reasoning, id }) => {
                            yield Ok(MultiTurnStreamItem::stream_item(StreamedAssistantContent::ReasoningDelta { reasoning, id }));
                            did_call_tool = false;
                        }
                        Ok(StreamedAssistantContent::Final(final_resp)) => {
                            if let Some(usage) = final_resp.token_usage() {
                                aggregated_usage += usage;
                            }

                            if is_text_response {
                                if let Some(ref hook) = self.hook {
                                    hook.on_stream_completion_response_finish(&prompt, &final_resp, cancel_sig.clone()).await;

                                    if cancel_sig.is_cancelled() {
                                        yield Err(make_cancel_error(chat_history.read().await.clone(), &cancel_sig));
                                    }
                                }

                                tracing::Span::current().record("gen_ai.completion", &last_text_response);
                                yield Ok(MultiTurnStreamItem::stream_item(StreamedAssistantContent::Final(final_resp)));
                                is_text_response = false;
                            }
                        }
                        Err(e) => {
                            yield Err(e.into());
                            break 'outer;
                        }
                    }
                }

                // Add tool calls to chat history
                if !tool_calls.is_empty() {
                    chat_history.write().await.push(Message::Assistant {
                        id: None,
                        content: OneOrMany::many(tool_calls.clone()).expect("non-empty"),
                    });
                }

                // Add tool results to chat history
                for (id, call_id, tool_result) in tool_results {
                    if let Some(call_id) = call_id {
                        chat_history.write().await.push(Message::User {
                            content: OneOrMany::one(UserContent::tool_result_with_call_id(
                                &id,
                                call_id.clone(),
                                OneOrMany::one(ToolResultContent::text(&tool_result)),
                            )),
                        });
                    } else {
                        chat_history.write().await.push(Message::User {
                            content: OneOrMany::one(UserContent::tool_result(
                                &id,
                                OneOrMany::one(ToolResultContent::text(&tool_result)),
                            )),
                        });
                    }
                }

                // Update current prompt
                current_prompt = match chat_history.write().await.pop() {
                    Some(prompt) => prompt,
                    None => unreachable!("chat history should never be empty"),
                };

                if !did_call_tool {
                    let current_span = tracing::Span::current();
                    current_span.record("gen_ai.usage.input_tokens", aggregated_usage.input_tokens);
                    current_span.record("gen_ai.usage.output_tokens", aggregated_usage.output_tokens);
                    tracing::info!("Agent multi-turn stream finished");
                    yield Ok(MultiTurnStreamItem::final_response(&last_text_response, aggregated_usage));
                    break;
                }
            }

            if max_depth_reached {
                yield Err(Box::new(PromptError::MaxDepthError {
                    max_depth: self.max_depth,
                    chat_history: Box::new((*chat_history.read().await).clone()),
                    prompt: Box::new(last_prompt_error.clone().into()),
                }).into());
            }
        };

        Box::pin(stream.instrument(agent_span))
    }
}

impl<M, P> IntoFuture for StreamingPromptRequest<M, P>
where
    M: CompletionModel + 'static,
    <M as CompletionModel>::StreamingResponse: WasmCompatSend,
    P: StreamingPromptHook<M> + 'static,
{
    type Output = StreamingResult<M::StreamingResponse>;
    type IntoFuture = WasmBoxedFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move { self.send().await })
    }
}

/// Helper function to stream a completion request to stdout.
pub async fn stream_to_stdout<R>(
    stream: &mut StreamingResult<R>,
) -> Result<FinalResponse, std::io::Error> {
    let mut final_res = FinalResponse::empty();

    print!("Response: ");
    while let Some(content) = stream.next().await {
        match content {
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                Text { text },
            ))) => {
                print!("{text}");
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Reasoning(
                Reasoning { reasoning, .. },
            ))) => {
                let reasoning = reasoning.join("\n");
                print!("{reasoning}");
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
            Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                final_res = res;
            }
            Err(err) => {
                eprintln!("Error: {err}");
            }
            _ => {}
        }
    }

    Ok(final_res)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::ProviderClient;
    use crate::client::completion::CompletionClient;
    use crate::completion::streaming::StreamingPrompt;
    use crate::providers::anthropic;
    use futures::StreamExt;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::time::Duration;

    /// Background task that logs periodically to detect span leakage.
    async fn background_logger(stop: Arc<AtomicBool>, leak_count: Arc<AtomicU32>) {
        let mut interval = tokio::time::interval(Duration::from_millis(50));
        let mut count = 0u32;

        while !stop.load(Ordering::Relaxed) {
            interval.tick().await;
            count += 1;

            tracing::event!(
                target: "background_logger",
                tracing::Level::INFO,
                count = count,
                "Background tick"
            );

            let current = tracing::Span::current();
            if !current.is_disabled() && !current.is_none() {
                leak_count.fetch_add(1, Ordering::Relaxed);
            }
        }

        tracing::info!(target: "background_logger", total_ticks = count, "Background logger stopped");
    }

    /// Test that span context doesn't leak to concurrent tasks during streaming.
    #[tokio::test(flavor = "current_thread")]
    #[ignore = "This requires an API key"]
    async fn test_span_context_isolation() {
        let stop = Arc::new(AtomicBool::new(false));
        let leak_count = Arc::new(AtomicU32::new(0));

        let bg_stop = stop.clone();
        let bg_leak = leak_count.clone();
        let bg_handle = tokio::spawn(async move {
            background_logger(bg_stop, bg_leak).await;
        });

        tokio::time::sleep(Duration::from_millis(100)).await;

        let client = anthropic::Client::from_env();
        let agent = client
            .agent(anthropic::completion::CLAUDE_3_5_HAIKU)
            .preamble("You are a helpful assistant.")
            .temperature(0.1)
            .max_tokens(100)
            .build();

        let mut stream = agent
            .stream_prompt("Say 'hello world' and nothing else.")
            .await;

        let mut full_content = String::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                    text,
                ))) => {
                    full_content.push_str(&text.text);
                }
                Ok(MultiTurnStreamItem::FinalResponse(_)) => {
                    break;
                }
                Err(e) => {
                    tracing::warn!("Error: {:?}", e);
                    break;
                }
                _ => {}
            }
        }

        tracing::info!("Got response: {:?}", full_content);

        stop.store(true, Ordering::Relaxed);
        bg_handle.await.unwrap();

        let leaks = leak_count.load(Ordering::Relaxed);
        assert_eq!(
            leaks, 0,
            "SPAN LEAK DETECTED: Background logger was inside unexpected spans {leaks} times."
        );
    }
}
