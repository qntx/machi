//! OpenAI Embedding API implementation.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::embedding::{
    Embedding, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
};
use crate::error::{LlmError, Result};

use super::client::OpenAI;

/// OpenAI embedding request.
#[derive(Debug, Clone, Serialize)]
struct OpenAIEmbeddingRequest {
    pub model: String,
    pub input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
}

/// OpenAI embedding data.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIEmbeddingData {
    pub embedding: Vec<f32>,
    pub index: usize,
}

/// OpenAI embedding response.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIEmbeddingResponse {
    pub data: Vec<OpenAIEmbeddingData>,
    pub model: String,
    pub usage: Option<OpenAIEmbeddingUsage>,
}

/// OpenAI embedding usage statistics.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIEmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

/// Default embedding model for OpenAI.
const DEFAULT_EMBEDDING_MODEL: &str = "text-embedding-3-small";
/// Default embedding dimension for text-embedding-3-small.
const DEFAULT_EMBEDDING_DIMENSION: usize = 1536;

#[async_trait]
impl EmbeddingProvider for OpenAI {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        let url = self.embeddings_url();

        let body = OpenAIEmbeddingRequest {
            model: request.model.clone(),
            input: request.input.clone(),
            encoding_format: request.encoding_format.map(|f| f.as_str().to_owned()),
            dimensions: request.dimensions,
        };

        let response = self.build_request(&url).json(&body).send().await?;

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

        let (usage, total_tokens) = if let Some(u) = parsed.usage {
            (
                Some(EmbeddingUsage {
                    prompt_tokens: u.prompt_tokens,
                    total_tokens: u.total_tokens,
                }),
                Some(u.total_tokens),
            )
        } else {
            (None, None)
        };

        Ok(EmbeddingResponse {
            embeddings,
            model: Some(parsed.model),
            usage,
            total_tokens,
        })
    }

    fn default_embedding_model(&self) -> &str {
        DEFAULT_EMBEDDING_MODEL
    }

    fn embedding_dimension(&self) -> Option<usize> {
        Some(DEFAULT_EMBEDDING_DIMENSION)
    }
}
