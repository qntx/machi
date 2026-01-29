//! Core completion traits.
//!
//! This module defines the fundamental traits for completion operations:
//! - [`Prompt`] - High-level one-shot prompt interface
//! - [`Chat`] - High-level chat interface with history
//! - [`Completion`] - Low-level completion interface
//! - [`CompletionModel`] - Completion model implementation trait
//! - [`GetTokenUsage`] - Token usage extraction trait

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::core::wasm_compat::{WasmCompatSend, WasmCompatSync};
use crate::message::Message;

use super::errors::{CompletionError, PromptError};
use super::request::{CompletionRequest, CompletionRequestBuilder, CompletionResponse, Usage};
use super::streaming::StreamingCompletionResponse;

/// Trait defining a high-level LLM simple prompt interface (i.e.: prompt in, response out).
pub trait Prompt: WasmCompatSend + WasmCompatSync {
    /// Send a simple prompt to the underlying completion model.
    ///
    /// If the completion model's response is a message, then it is returned as a string.
    ///
    /// If the completion model's response is a tool call, then the tool is called and
    /// the result is returned as a string.
    ///
    /// If the tool does not exist, or the tool call fails, then an error is returned.
    fn prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> impl std::future::IntoFuture<Output = Result<String, PromptError>, IntoFuture: WasmCompatSend>;
}

/// Trait defining a high-level LLM chat interface (i.e.: prompt and chat history in, response out).
pub trait Chat: WasmCompatSend + WasmCompatSync {
    /// Send a prompt with optional chat history to the underlying completion model.
    ///
    /// If the completion model's response is a message, then it is returned as a string.
    ///
    /// If the completion model's response is a tool call, then the tool is called and the result
    /// is returned as a string.
    ///
    /// If the tool does not exist, or the tool call fails, then an error is returned.
    fn chat(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: Vec<Message>,
    ) -> impl std::future::IntoFuture<Output = Result<String, PromptError>, IntoFuture: WasmCompatSend>;
}

/// Trait defining a low-level LLM completion interface
pub trait Completion<M: CompletionModel> {
    /// Generates a completion request builder for the given `prompt` and `chat_history`.
    /// This function is meant to be called by the user to further customize the
    /// request at prompt time before sending it.
    ///
    /// ?IMPORTANT: The type that implements this trait might have already
    /// populated fields in the builder (the exact fields depend on the type).
    /// For fields that have already been set by the model, calling the corresponding
    /// method on the builder will overwrite the value set by the model.
    ///
    /// For example, the request builder returned by [`Agent::completion`](crate::agent::Agent::completion) will already
    /// contain the `preamble` provided when creating the agent.
    fn completion(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: Vec<Message>,
    ) -> impl std::future::Future<Output = Result<CompletionRequestBuilder<M>, CompletionError>>
    + WasmCompatSend;
}

/// Trait defining a completion model that can be used to generate completion responses.
/// This trait is meant to be implemented by the user to define a custom completion model,
/// either from a third party provider (e.g.: OpenAI) or a local model.
pub trait CompletionModel: Clone + WasmCompatSend + WasmCompatSync {
    /// The raw response type returned by the underlying completion model.
    type Response: WasmCompatSend + WasmCompatSync + Serialize + DeserializeOwned;
    /// The raw response type returned by the underlying completion model when streaming.
    type StreamingResponse: Clone
        + Unpin
        + WasmCompatSend
        + WasmCompatSync
        + Serialize
        + DeserializeOwned
        + GetTokenUsage;

    type Client;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self;

    /// Generates a completion response for the given completion request.
    fn completion(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<
        Output = Result<CompletionResponse<Self::Response>, CompletionError>,
    > + WasmCompatSend;

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<
        Output = Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>,
    > + WasmCompatSend;

    /// Generates a completion request builder for the given `prompt`.
    fn completion_request(&self, prompt: impl Into<Message>) -> CompletionRequestBuilder<Self> {
        CompletionRequestBuilder::new(self.clone(), prompt)
    }
}

/// A trait for grabbing the token usage of a completion response.
///
/// Primarily designed for streamed completion responses in streamed multi-turn, as otherwise it would be impossible to do.
pub trait GetTokenUsage {
    fn token_usage(&self) -> Option<Usage>;
}

impl GetTokenUsage for () {
    fn token_usage(&self) -> Option<Usage> {
        None
    }
}

impl<T> GetTokenUsage for Option<T>
where
    T: GetTokenUsage,
{
    fn token_usage(&self) -> Option<Usage> {
        if let Some(usage) = self {
            usage.token_usage()
        } else {
            None
        }
    }
}
