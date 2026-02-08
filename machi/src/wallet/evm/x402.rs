//! x402 protocol integration for [`EvmWallet`](super::EvmWallet).
//!
//! Provides transparent HTTP 402 payment handling using the wallet's
//! [`PrivateKeySigner`] to sign ERC-3009 `transferWithAuthorization`
//! payment authorizations.
//!
//! # Architecture
//!
//! ```text
//! Agent calls x402_fetch tool
//!   → X402HttpClient.get(url)
//!     → reqwest + x402-reqwest middleware
//!       ├→ GET /resource → 402 Payment Required
//!       ├→ V2Eip155ExactClient signs payment with PrivateKeySigner
//!       └→ GET /resource + X-PAYMENT header → 200 OK
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use machi::wallet::EvmWallet;
//! use machi::wallet::x402::X402HttpClient;
//!
//! # async fn example() -> machi::Result<()> {
//! let wallet = EvmWallet::from_private_key("0x...", "https://base-rpc.example.com").await?;
//! let client = X402HttpClient::from_wallet(&wallet);
//! let body = client.get("https://api.example.com/paid-resource").await?;
//! println!("{body}");
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::Value;
use tracing::debug;
use x402_chain_eip155::{V1Eip155ExactClient, V2Eip155ExactClient};
use x402_reqwest::{ReqwestWithPayments, ReqwestWithPaymentsBuild, X402Client};

use super::wallet::EvmWallet;
use crate::tool::{BoxedTool, DynTool, ToolDefinition, ToolError};
use crate::wallet::WalletError;

/// HTTP client with transparent x402 payment capabilities.
///
/// Wraps [`reqwest`] with [`x402_reqwest`] middleware so that any request
/// returning HTTP 402 is automatically retried with a signed payment header.
/// Uses the [`EvmWallet`]'s [`PrivateKeySigner`](alloy::signers::local::PrivateKeySigner)
/// to sign ERC-3009 payment authorizations — **no gas required**.
///
/// Registers both V1 and V2 EIP-155 exact scheme clients for maximum
/// compatibility with x402 servers.
#[derive(Debug, Clone)]
pub struct X402HttpClient {
    inner: reqwest_middleware::ClientWithMiddleware,
}

impl X402HttpClient {
    /// Create an x402-enabled HTTP client from an [`EvmWallet`].
    ///
    /// Extracts the wallet's signer and registers V1 + V2 EIP-155
    /// exact payment scheme clients.
    #[must_use]
    pub fn from_wallet(wallet: &EvmWallet) -> Self {
        let signer = Arc::new(wallet.signer().clone());

        let x402_client = X402Client::new()
            .register(V1Eip155ExactClient::new(Arc::clone(&signer)))
            .register(V2Eip155ExactClient::new(signer));

        let inner = Client::new().with_payments(x402_client).build();

        debug!(
            address = %wallet.address(),
            chain = %wallet.chain_name(),
            "x402 HTTP client created",
        );

        Self { inner }
    }

    /// Create an x402-enabled HTTP client from a shared [`EvmWallet`] reference.
    #[must_use]
    pub fn from_wallet_arc(wallet: &Arc<EvmWallet>) -> Self {
        Self::from_wallet(wallet)
    }

    /// Reference to the underlying [`reqwest_middleware::ClientWithMiddleware`].
    #[must_use]
    pub const fn client(&self) -> &reqwest_middleware::ClientWithMiddleware {
        &self.inner
    }

    /// Perform a GET request with transparent x402 payment handling.
    ///
    /// If the server responds with HTTP 402, the middleware automatically
    /// signs a payment and retries the request.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails or payment cannot be completed.
    pub async fn get(&self, url: &str) -> Result<String, WalletError> {
        let response = self
            .inner
            .get(url)
            .send()
            .await
            .map_err(|e| WalletError::payment(format!("GET request failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            return Err(WalletError::payment(format!(
                "request returned status {status}"
            )));
        }

        response
            .text()
            .await
            .map_err(|e| WalletError::payment(format!("failed to read response body: {e}")))
    }

    /// Perform a POST request with transparent x402 payment handling.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails or payment cannot be completed.
    pub async fn post(&self, url: &str, body: impl Into<String>) -> Result<String, WalletError> {
        let response = self
            .inner
            .post(url)
            .header("content-type", "application/json")
            .body(body.into())
            .send()
            .await
            .map_err(|e| WalletError::payment(format!("POST request failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            return Err(WalletError::payment(format!(
                "request returned status {status}"
            )));
        }

        response
            .text()
            .await
            .map_err(|e| WalletError::payment(format!("failed to read response body: {e}")))
    }
}

/// Agent tool: fetch a URL with automatic x402 payment.
///
/// When the target server returns HTTP 402 Payment Required, this tool
/// transparently signs and submits an ERC-3009 payment authorization
/// using the wallet's EVM signer. No gas is consumed for payment signing.
#[derive(Debug)]
pub(crate) struct X402FetchTool(Arc<X402HttpClient>);

impl X402FetchTool {
    /// Create a new x402 fetch tool from a shared HTTP client.
    pub const fn new(client: Arc<X402HttpClient>) -> Self {
        Self(client)
    }
}

#[async_trait]
impl DynTool for X402FetchTool {
    fn name(&self) -> &'static str {
        "x402_fetch"
    }

    fn description(&self) -> String {
        String::from(
            "Fetch a URL with automatic x402 payment. If the server requires \
             payment (HTTP 402), this tool transparently signs an ERC-3009 \
             payment authorization using the wallet and retries the request. \
             Supports GET and POST methods.",
        )
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            self.name(),
            self.description(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch (must include scheme, e.g. https://...)."
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST"],
                        "description": "HTTP method. Defaults to GET."
                    },
                    "body": {
                        "type": "string",
                        "description": "Request body for POST requests (JSON string)."
                    }
                },
                "required": ["url"],
                "additionalProperties": false
            }),
        )
    }

    async fn call_json(&self, args: Value) -> Result<Value, ToolError> {
        const MAX_BODY_LEN: usize = 100_000;

        let url = args
            .get("url")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::invalid_args("missing required field 'url'"))?;

        let method = args.get("method").and_then(Value::as_str).unwrap_or("GET");

        let body = match method.to_uppercase().as_str() {
            "POST" => {
                let post_body = args.get("body").and_then(Value::as_str).unwrap_or("{}");
                self.0.post(url, post_body).await?
            }
            _ => self.0.get(url).await?,
        };

        // Truncate very large responses to keep context manageable.
        let truncated = body.len() > MAX_BODY_LEN;
        let content = if truncated {
            &body[..MAX_BODY_LEN]
        } else {
            body.as_str()
        };

        Ok(serde_json::json!({
            "url": url,
            "method": method.to_uppercase(),
            "content": content,
            "truncated": truncated,
            "content_length": body.len(),
        }))
    }
}

/// Create x402 tools from a shared wallet reference.
pub(crate) fn create_tools(wallet: &Arc<EvmWallet>) -> Vec<BoxedTool> {
    let client = Arc::new(X402HttpClient::from_wallet_arc(wallet));
    vec![Box::new(X402FetchTool::new(client))]
}
