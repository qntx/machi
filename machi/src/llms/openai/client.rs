//! `OpenAI` API client implementation.
//!
//! Core types (`ChatRequest`, `Message`, `ToolDefinition`, `ResponseFormat`)
//! serialize directly to `OpenAI`'s expected JSON format — no intermediate
//! wire types are needed for the request path.

use std::sync::Arc;
use std::time::Duration;

use reqwest_middleware::ClientWithMiddleware;
use serde::Deserialize;
use serde_json::Value;

use super::config::OpenAIConfig;
use crate::chat::ChatRequest;
use crate::error::Result;
use crate::llms::LlmError;

/// `OpenAI` error response.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIErrorResponse {
    pub error: OpenAIError,
}

/// `OpenAI` error details.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIError {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

/// `OpenAI` API client.
///
/// Uses [`ClientWithMiddleware`] internally, which acts as a plain HTTP
/// client when no middleware is registered. When constructed via
/// [`from_wallet`](Self::from_wallet), the x402 payment middleware is
/// added so that HTTP 402 responses are handled transparently.
#[derive(Debug, Clone)]
pub struct OpenAI {
    pub(crate) config: Arc<OpenAIConfig>,
    pub(crate) client: ClientWithMiddleware,
}

impl OpenAI {
    /// Create a new `OpenAI` client with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the API key is empty or the HTTP client fails to build.
    pub fn new(config: OpenAIConfig) -> Result<Self> {
        if config.api_key.is_empty() {
            return Err(LlmError::auth("openai", "API key is required").into());
        }

        let client = Self::build_http_client(&config)?;

        Ok(Self {
            config: Arc::new(config),
            client: reqwest_middleware::ClientBuilder::new(client).build(),
        })
    }

    /// Create an x402-enabled client from an [`EvmWallet`](crate::wallet::EvmWallet).
    ///
    /// The returned client transparently handles HTTP 402 responses by
    /// signing ERC-3009 payment authorizations using the wallet's signer.
    /// Default base URL is `https://llm.qntx.fun/v1`.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client fails to build.
    #[cfg(feature = "x402")]
    pub fn from_wallet(wallet: &crate::wallet::EvmWallet) -> Result<Self> {
        Self::from_wallet_with(wallet, OpenAIConfig::x402())
    }

    /// Create an x402-enabled client with custom configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client fails to build.
    #[cfg(feature = "x402")]
    pub fn from_wallet_with(
        wallet: &crate::wallet::EvmWallet,
        config: OpenAIConfig,
    ) -> Result<Self> {
        use r402::scheme::PreferChain;
        use r402::chain::ChainIdPattern;
        use r402_evm::Eip155ExactClient;
        use r402_http::client::{WithPayments, X402Client};

        let http = Self::build_http_client(&config)?;
        let signer = Arc::new(wallet.signer().clone());

        // Prefer the wallet's chain so the middleware selects a network
        // where the wallet actually holds funds, instead of blindly
        // picking the first entry in the server's `accepts` list.
        let prefer = PreferChain::new(vec![
            ChainIdPattern::exact("eip155", wallet.chain_id().to_string()),
        ]);
        let x402 = X402Client::new()
            .register(Eip155ExactClient::new(signer))
            .with_selector(prefer);
        let client = http.with_payments(x402);

        tracing::debug!(
            address = %wallet.address(),
            chain = %wallet.chain_name(),
            base_url = %config.base_url,
            "x402-enabled OpenAI client created",
        );

        Ok(Self {
            config: Arc::new(config),
            client,
        })
    }

    /// Create a client from environment variables.
    ///
    /// # Errors
    ///
    /// Returns an error if required environment variables are missing or the client fails to build.
    pub fn from_env() -> Result<Self> {
        let config = OpenAIConfig::from_env()?;
        Self::new(config)
    }

    /// Build the underlying [`reqwest::Client`] with timeout from config.
    fn build_http_client(config: &OpenAIConfig) -> Result<reqwest::Client> {
        let mut builder = reqwest::Client::builder();
        if let Some(timeout) = config.timeout_secs {
            builder = builder.timeout(Duration::from_secs(timeout));
        }
        builder
            .build()
            .map_err(|e| LlmError::internal(format!("Failed to create HTTP client: {e}")).into())
    }

    /// Get the API key.
    #[must_use]
    pub fn api_key(&self) -> &str {
        &self.config.api_key
    }

    /// Get the base URL.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    /// Get the default model.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Build the chat completions URL.
    pub(crate) fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.config.base_url)
    }

    /// Build the audio speech URL.
    pub(crate) fn speech_url(&self) -> String {
        format!("{}/audio/speech", self.config.base_url)
    }

    /// Build the audio transcriptions URL.
    pub(crate) fn transcriptions_url(&self) -> String {
        format!("{}/audio/transcriptions", self.config.base_url)
    }

    /// Build the embeddings URL.
    pub(crate) fn embeddings_url(&self) -> String {
        format!("{}/embeddings", self.config.base_url)
    }

    /// Build request headers for JSON requests.
    pub(crate) fn build_request(&self, url: &str) -> reqwest_middleware::RequestBuilder {
        let mut req = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json");

        if let Some(org) = &self.config.organization {
            req = req.header("OpenAI-Organization", org);
        }

        req
    }

    /// Build request headers for multipart requests.
    pub(crate) fn build_multipart_request(&self, url: &str) -> reqwest_middleware::RequestBuilder {
        let mut req = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.config.api_key));

        if let Some(org) = &self.config.organization {
            req = req.header("OpenAI-Organization", org);
        }

        req
    }

    /// Serialize a [`ChatRequest`] to a JSON [`Value`] for the `OpenAI` API.
    ///
    /// Core types already carry the correct serde attributes, so this is a
    /// thin wrapper that fills in the default model and adds `stream_options`
    /// when streaming.
    pub(crate) fn build_chat_body(&self, request: &ChatRequest, streaming: bool) -> Result<Value> {
        let mut body = serde_json::to_value(request)
            .map_err(|e| LlmError::internal(format!("Failed to serialize request: {e}")))?;

        // Fill default model from config when the request omits it.
        if request.model.is_empty() {
            body["model"] = Value::String(self.config.model.clone());
        }

        // Enable streaming with usage reporting.
        if streaming {
            body["stream"] = Value::Bool(true);
            body["stream_options"] = serde_json::json!({"include_usage": true});
        }

        Ok(body)
    }

    /// Parse an error response from `OpenAI`.
    pub(crate) fn parse_error(status: u16, body: &str) -> LlmError {
        if let Ok(error_response) = serde_json::from_str::<OpenAIErrorResponse>(body) {
            let error = error_response.error;
            let code = error.code.unwrap_or_else(|| error.error_type.clone());

            return match status {
                401 => LlmError::auth("openai", error.message),
                429 => LlmError::rate_limited("openai"),
                400 if error.message.contains("context_length") => {
                    // Attempt to extract token counts from the error message.
                    let (used, max) = parse_context_length_tokens(&error.message);
                    LlmError::context_exceeded(used, max)
                }
                _ => LlmError::provider_code("openai", code, error.message),
            };
        }

        LlmError::http_status(status, body.to_owned())
    }
}

/// Extract token counts from an `OpenAI` context-length error message.
///
/// `OpenAI` error messages typically look like:
/// "This model's maximum context length is 8192 tokens. However, your messages resulted in 9500 tokens."
/// Returns `(used, max)`, defaulting to `(0, 0)` if parsing fails.
fn parse_context_length_tokens(message: &str) -> (usize, usize) {
    let mut max = 0usize;
    let mut used = 0usize;

    // Look for "maximum context length is <N> tokens"
    if let Some(pos) = message.find("maximum context length is ") {
        let after = &message[pos + "maximum context length is ".len()..];
        if let Some(end) = after.find(|c: char| !c.is_ascii_digit()) {
            max = after[..end].parse().unwrap_or(0);
        }
    }

    // Look for "resulted in <N> tokens" or "your messages resulted in <N> tokens"
    if let Some(pos) = message.find("resulted in ") {
        let after = &message[pos + "resulted in ".len()..];
        if let Some(end) = after.find(|c: char| !c.is_ascii_digit()) {
            used = after[..end].parse().unwrap_or(0);
        }
    }

    (used, max)
}
