//! LLM sampling contracts for the Ovo kernel.
//!
//! - Always: [`LlmSampler`], [`MockSampler`], wire helpers in [`openai_compat`].
//! - Decorators: [`RetryingSampler`], [`BreakerSampler`].
//! - Feature `openai`: [`OpenAiCompatSampler`] HTTP client.
//! - Feature `ollama`: [`OllamaSampler`] HTTP client.

#![forbid(unsafe_code)]

pub mod breaker;
pub mod breaker_sampler;
pub mod mock;
pub mod openai_compat;
pub mod retry;
pub mod retrying;
pub mod sample;
pub mod sampler;
pub mod stream;

#[cfg(feature = "ollama")]
pub mod ollama;

pub use breaker::{Admission, BreakerConfig, BreakerOutcome, BreakerState, CircuitBreaker};
pub use breaker_sampler::BreakerSampler;
pub use mock::MockSampler;
#[cfg(feature = "ollama")]
pub use ollama::{OllamaConfig, OllamaSampler, build_ollama_chat_body, parse_ollama_chat_response};
#[cfg(feature = "openai")]
pub use openai_compat::OpenAiCompatSampler;
pub use openai_compat::{
    OpenAiCompatConfig, build_chat_completions_body, parse_chat_completions_response,
};
pub use retry::{
    DEFAULT_MAX_ATTEMPTS, HttpRetryClass, MAX_RETRY_AFTER, MAX_RETRY_BACKOFF,
    RATE_LIMIT_RETRY_THRESHOLD, RetryContext, RetryDecision, RetryPolicy, backoff_for_attempt,
    classify_http_status, decide_retry, error_code_for_http, is_empty_response,
};
pub use retrying::{DEFAULT_IDLE_TIMEOUT, RetryingSampler};
pub use sample::{SampleRequest, SampleResponse, ToolChoice};
pub use sampler::{LlmSampler, response_to_stream};
pub use stream::{SampleEvent, SampleStream};
