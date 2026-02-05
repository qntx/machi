//! Ollama Embedding API implementation.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::embedding::{Embedding, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse};
use crate::error::{LlmError, Result};

use super::client::Ollama;

/// Default embedding model for Ollama.
const DEFAULT_EMBEDDING_MODEL: &str = "nomic-embed-text";

/// Ollama embedding request.
#[derive(Debug, Clone, Serialize)]
struct OllamaEmbeddingRequest {
    model: String,
    input: Vec<String>,
}

/// Ollama embedding response.
#[derive(Debug, Clone, Deserialize)]
struct OllamaEmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
}

#[async_trait]
impl EmbeddingProvider for Ollama {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        let url = self.embeddings_url();

        let body = OllamaEmbeddingRequest {
            model: request.model.clone(),
            input: request.input.clone(),
        };

        let response = self.client().post(&url).json(&body).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(Self::parse_error(status.as_u16(), &error_text).into());
        }

        let response_text = response.text().await?;
        let parsed: OllamaEmbeddingResponse =
            serde_json::from_str(&response_text).map_err(|e| {
                LlmError::response_format(
                    "valid Ollama embedding response",
                    format!("parse error: {e}, response: {response_text}"),
                )
            })?;

        let embeddings = parsed
            .embeddings
            .into_iter()
            .enumerate()
            .map(|(i, vector)| Embedding::new(vector, i))
            .collect();

        Ok(EmbeddingResponse {
            embeddings,
            model: Some(request.model.clone()),
            usage: None,
            total_tokens: parsed.prompt_eval_count,
        })
    }

    fn default_embedding_model(&self) -> &str {
        DEFAULT_EMBEDDING_MODEL
    }

    fn embedding_dimension(&self) -> Option<usize> {
        // Dimension depends on the model, so we don't specify a default
        None
    }
}
