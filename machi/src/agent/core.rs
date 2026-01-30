//! Core Agent structure and trait implementations.
//!
//! This module contains the main [`Agent`] struct which represents an LLM agent
//! that combines a completion model with a system prompt, context documents, and tools.

use std::{collections::HashMap, sync::Arc};

use futures::{StreamExt, TryStreamExt, stream};
use tokio::sync::RwLock;

use crate::{
    completion::{
        Chat, Completion, CompletionError, CompletionModel, CompletionRequestBuilder, Document,
        GetTokenUsage, Message, Prompt, PromptError,
        message::ToolChoice,
        streaming::{StreamingChat, StreamingCompletion, StreamingPrompt},
    },
    core::wasm_compat::WasmCompatSend,
    store::{VectorStoreError, request::VectorSearchRequest},
    tool::server::ToolServerHandle,
};

use super::request::{PromptRequest, StreamingPromptRequest, prompt::Standard};

const UNKNOWN_AGENT_NAME: &str = "Unnamed Agent";

/// Type alias for dynamic context stores.
pub type DynamicContextStore = Arc<
    RwLock<
        Vec<(
            usize,
            Box<dyn crate::store::VectorStoreIndexDyn + Send + Sync>,
        )>,
    >,
>;

/// An LLM agent that combines a completion model with configuration and tools.
///
/// An agent encapsulates:
/// - A completion model (e.g., GPT-4, Claude)
/// - A system prompt (preamble)
/// - Static and dynamic context documents
/// - Tools that the agent can use
/// - Model parameters (temperature, max_tokens, etc.)
///
/// # Example
/// ```rust,ignore
/// use machi::{completion::Prompt, providers::openai};
///
/// let openai = openai::Client::from_env();
///
/// let comedian_agent = openai
///     .agent("gpt-4o")
///     .preamble("You are a comedian here to entertain the user using humour and jokes.")
///     .temperature(0.9)
///     .build();
///
/// let response = comedian_agent.prompt("Entertain me!")
///     .await
///     .expect("Failed to prompt the agent");
/// ```
#[derive(Clone)]
#[non_exhaustive]
pub struct Agent<M>
where
    M: CompletionModel,
{
    /// Name of the agent used for logging and debugging.
    pub name: Option<String>,
    /// Agent description. Useful for sub-agents in workflows.
    pub description: Option<String>,
    /// The completion model to use.
    pub model: Arc<M>,
    /// System prompt (preamble).
    pub preamble: Option<String>,
    /// Static context documents always available to the agent.
    pub static_context: Vec<Document>,
    /// Temperature setting for the model.
    pub temperature: Option<f64>,
    /// Maximum number of tokens for completions.
    pub max_tokens: Option<u64>,
    /// Additional model-specific parameters.
    pub additional_params: Option<serde_json::Value>,
    /// Handle to the tool server for tool execution.
    pub tool_server_handle: ToolServerHandle,
    /// Dynamic context stores with sample counts.
    pub dynamic_context: DynamicContextStore,
    /// Tool choice configuration.
    pub tool_choice: Option<ToolChoice>,
    /// Default maximum depth for multi-turn conversations.
    pub default_max_depth: Option<usize>,
}

impl<M> Agent<M>
where
    M: CompletionModel,
{
    /// Returns the name of the agent, or a default if not set.
    pub(crate) fn name(&self) -> &str {
        self.name.as_deref().unwrap_or(UNKNOWN_AGENT_NAME)
    }
}

impl<M> Completion<M> for Agent<M>
where
    M: CompletionModel,
{
    async fn completion(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: Vec<Message>,
    ) -> Result<CompletionRequestBuilder<M>, CompletionError> {
        let prompt = prompt.into();

        // Find the latest message containing RAG text
        let rag_text = prompt
            .rag_text()
            .or_else(|| chat_history.iter().rev().find_map(Message::rag_text));

        // Build the base completion request
        let request = self
            .model
            .completion_request(prompt)
            .messages(chat_history)
            .temperature_opt(self.temperature)
            .max_tokens_opt(self.max_tokens)
            .additional_params_opt(self.additional_params.clone())
            .documents(self.static_context.clone());

        // Add preamble if present
        let request = match &self.preamble {
            Some(preamble) => request.preamble(preamble.to_owned()),
            None => request,
        };

        // Add tool choice if present
        let request = match &self.tool_choice {
            Some(tool_choice) => request.tool_choice(tool_choice.clone()),
            None => request,
        };

        // Fetch dynamic context and tools if RAG text is available
        let request = if let Some(text) = &rag_text {
            let dynamic_context = stream::iter(self.dynamic_context.read().await.iter())
                .then(|(num_sample, index)| async {
                    let req = VectorSearchRequest::builder()
                        .query(text)
                        .samples(*num_sample as u64)
                        .build()
                        .expect("VectorSearchRequest build should not fail");

                    Ok::<_, VectorStoreError>(
                        index
                            .top_n(req)
                            .await?
                            .into_iter()
                            .map(|(_, id, doc)| {
                                let text = serde_json::to_string_pretty(&doc)
                                    .unwrap_or_else(|_| doc.to_string());
                                Document {
                                    id,
                                    text,
                                    additional_props: HashMap::new(),
                                }
                            })
                            .collect::<Vec<_>>(),
                    )
                })
                .try_fold(vec![], |mut acc, docs| async {
                    acc.extend(docs);
                    Ok(acc)
                })
                .await
                .map_err(|e| CompletionError::RequestError(Box::new(e)))?;

            let tooldefs = self
                .tool_server_handle
                .get_tool_defs(Some(text.clone()))
                .await
                .map_err(|_| {
                    CompletionError::RequestError("Failed to get tool definitions".into())
                })?;

            request.documents(dynamic_context).tools(tooldefs)
        } else {
            let tooldefs = self
                .tool_server_handle
                .get_tool_defs(None)
                .await
                .map_err(|_| {
                    CompletionError::RequestError("Failed to get tool definitions".into())
                })?;

            request.tools(tooldefs)
        };

        Ok(request)
    }
}

#[allow(refining_impl_trait)]
impl<M> Prompt for Agent<M>
where
    M: CompletionModel,
{
    fn prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> PromptRequest<'_, Standard, M, ()> {
        PromptRequest::new(self, prompt)
    }
}

#[allow(refining_impl_trait)]
impl<M> Prompt for &Agent<M>
where
    M: CompletionModel,
{
    #[tracing::instrument(skip(self, prompt), fields(agent_name = self.name()))]
    fn prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> PromptRequest<'_, Standard, M, ()> {
        PromptRequest::new(*self, prompt)
    }
}

#[allow(refining_impl_trait)]
impl<M> Chat for Agent<M>
where
    M: CompletionModel,
{
    #[tracing::instrument(skip(self, prompt, chat_history), fields(agent_name = self.name()))]
    async fn chat(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        mut chat_history: Vec<Message>,
    ) -> Result<String, PromptError> {
        PromptRequest::new(self, prompt)
            .with_history(&mut chat_history)
            .await
    }
}

impl<M> StreamingCompletion<M> for Agent<M>
where
    M: CompletionModel,
{
    async fn stream_completion(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: Vec<Message>,
    ) -> Result<CompletionRequestBuilder<M>, CompletionError> {
        self.completion(prompt, chat_history).await
    }
}

impl<M> StreamingPrompt<M, M::StreamingResponse> for Agent<M>
where
    M: CompletionModel + 'static,
    M::StreamingResponse: GetTokenUsage,
{
    fn stream_prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> StreamingPromptRequest<M, ()> {
        let arc = Arc::new(self.clone());
        StreamingPromptRequest::new(arc, prompt)
    }
}

impl<M> StreamingChat<M, M::StreamingResponse> for Agent<M>
where
    M: CompletionModel + 'static,
    M::StreamingResponse: GetTokenUsage,
{
    fn stream_chat(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: Vec<Message>,
    ) -> StreamingPromptRequest<M, ()> {
        let arc = Arc::new(self.clone());
        StreamingPromptRequest::new(arc, prompt).with_history(chat_history)
    }
}
