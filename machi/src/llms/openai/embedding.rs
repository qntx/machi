//! `OpenAI` Embedding API implementation.

use async_trait::async_trait;
use serde::Deserialize;

use super::client::OpenAI;
use crate::embedding::{
    Embedding, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
};
use crate::error::{LlmError, Result};

/// `OpenAI` embedding data (kept because the field is `embedding`, not `vector`).
#[derive(Debug, Clone, Deserialize)]
#[allow(clippy::missing_docs_in_private_items)]
struct OpenAIEmbeddingData {
    pub embedding: Vec<f32>,
    pub index: usize,
}

/// `OpenAI` embedding response wrapper.
#[derive(Debug, Clone, Deserialize)]
#[allow(clippy::missing_docs_in_private_items)]
struct OpenAIEmbeddingResponse {
    pub data: Vec<OpenAIEmbeddingData>,
    pub model: String,
    /// Deserialized directly into the core [`EmbeddingUsage`] type.
    pub usage: Option<EmbeddingUsage>,
}

/// Default embedding model for `OpenAI`.
const DEFAULT_EMBEDDING_MODEL: &str = "text-embedding-3-small";
/// Default embedding dimension for text-embedding-3-small.
const DEFAULT_EMBEDDING_DIMENSION: usize = 1536;

#[async_trait]
impl EmbeddingProvider for OpenAI {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        let url = self.embeddings_url();

        // EmbeddingRequest serializes directly to the OpenAI-expected format.
        let response = self.build_request(&url).json(request).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(Self::parse_error(status.as_u16(), &error_text).into());
        }

        let response_text = response.text().await?;
        let parsed: OpenAIEmbeddingResponse =
            serde_json::from_str(&response_text).map_err(|e| {
                LlmError::response_format(
                    "valid OpenAI embedding response",
                    format!("parse error: {e}, response: {response_text}"),
                )
            })?;

        let embeddings = parsed
            .data
            .into_iter()
            .map(|d| Embedding::new(d.embedding, d.index))
            .collect();

        let usage = parsed.usage;

        Ok(EmbeddingResponse {
            embeddings,
            model: Some(parsed.model),
            usage,
        })
    }

    fn default_embedding_model(&self) -> &str {
        DEFAULT_EMBEDDING_MODEL
    }

    fn embedding_dimension(&self) -> Option<usize> {
        Some(DEFAULT_EMBEDDING_DIMENSION)
    }
}
