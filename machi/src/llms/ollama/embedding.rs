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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    mod ollama_embedding_request {
        use super::*;

        #[test]
        fn serializes_correctly() {
            let request = OllamaEmbeddingRequest {
                model: "nomic-embed-text".to_owned(),
                input: vec!["Hello, world!".to_owned()],
            };

            let json = serde_json::to_string(&request).unwrap();

            assert!(json.contains("\"model\":\"nomic-embed-text\""));
            assert!(json.contains("\"input\":[\"Hello, world!\"]"));
        }

        #[test]
        fn serializes_multiple_inputs() {
            let request = OllamaEmbeddingRequest {
                model: "nomic-embed-text".to_owned(),
                input: vec![
                    "First text".to_owned(),
                    "Second text".to_owned(),
                    "Third text".to_owned(),
                ],
            };

            let json = serde_json::to_string(&request).unwrap();

            assert!(json.contains("First text"));
            assert!(json.contains("Second text"));
            assert!(json.contains("Third text"));
        }

        #[test]
        fn serializes_empty_input() {
            let request = OllamaEmbeddingRequest {
                model: "nomic-embed-text".to_owned(),
                input: vec![],
            };

            let json = serde_json::to_string(&request).unwrap();

            assert!(json.contains("\"input\":[]"));
        }
    }

    mod ollama_embedding_response {
        use super::*;

        #[test]
        fn deserializes_basic_response() {
            let json = r#"{
                "embeddings": [[0.1, 0.2, 0.3]],
                "prompt_eval_count": 5
            }"#;

            let response: OllamaEmbeddingResponse = serde_json::from_str(json).unwrap();

            assert_eq!(response.embeddings.len(), 1);
            assert_eq!(response.embeddings[0].len(), 3);
            assert_eq!(response.prompt_eval_count, Some(5));
        }

        #[test]
        fn deserializes_multiple_embeddings() {
            let json = r#"{
                "embeddings": [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6]
                ],
                "prompt_eval_count": 10
            }"#;

            let response: OllamaEmbeddingResponse = serde_json::from_str(json).unwrap();

            assert_eq!(response.embeddings.len(), 2);
        }

        #[test]
        fn deserializes_without_prompt_eval_count() {
            let json = r#"{
                "embeddings": [[0.1, 0.2, 0.3]]
            }"#;

            let response: OllamaEmbeddingResponse = serde_json::from_str(json).unwrap();

            assert!(response.prompt_eval_count.is_none());
        }

        #[test]
        fn deserializes_high_dimensional_embedding() {
            // Simulate a 768-dimensional embedding (common for many models)
            let vector: Vec<f32> = (0..768).map(|i| i as f32 * 0.001).collect();
            let json = format!(
                r#"{{"embeddings": [{}]}}"#,
                serde_json::to_string(&vector).unwrap()
            );

            let response: OllamaEmbeddingResponse = serde_json::from_str(&json).unwrap();

            assert_eq!(response.embeddings[0].len(), 768);
        }

        #[test]
        fn deserializes_empty_embeddings() {
            let json = r#"{"embeddings": []}"#;

            let response: OllamaEmbeddingResponse = serde_json::from_str(json).unwrap();

            assert!(response.embeddings.is_empty());
        }
    }

    mod embedding_provider_impl {
        use super::*;

        #[test]
        fn default_embedding_model_returns_nomic() {
            let client = Ollama::with_defaults().unwrap();

            assert_eq!(client.default_embedding_model(), "nomic-embed-text");
        }

        #[test]
        fn default_embedding_model_matches_constant() {
            let client = Ollama::with_defaults().unwrap();

            assert_eq!(client.default_embedding_model(), DEFAULT_EMBEDDING_MODEL);
        }

        #[test]
        fn embedding_dimension_is_none() {
            let client = Ollama::with_defaults().unwrap();

            // Ollama doesn't specify a fixed dimension since it depends on the model
            assert!(client.embedding_dimension().is_none());
        }
    }

    mod embedding_conversion {
        use super::*;

        #[test]
        fn converts_response_to_embedding_objects() {
            // Simulate the conversion logic from the embed function
            let embeddings_data = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];

            let embeddings: Vec<Embedding> = embeddings_data
                .into_iter()
                .enumerate()
                .map(|(i, vector)| Embedding::new(vector, i))
                .collect();

            assert_eq!(embeddings.len(), 2);
            assert_eq!(embeddings[0].index, 0);
            assert_eq!(embeddings[1].index, 1);
        }

        #[test]
        fn embedding_preserves_vector_values() {
            let vector = vec![0.123, 0.456, 0.789];
            let embedding = Embedding::new(vector.clone(), 0);

            assert_eq!(embedding.vector, vector);
        }
    }

    mod realistic_responses {
        use super::*;

        #[test]
        fn parses_nomic_embed_text_response() {
            // Nomic-embed-text produces 768-dimensional embeddings
            let vector: Vec<f32> = (0..768).map(|i| (i as f32 / 768.0) - 0.5).collect();
            let json = format!(
                r#"{{
                    "embeddings": [{}],
                    "prompt_eval_count": 12
                }}"#,
                serde_json::to_string(&vector).unwrap()
            );

            let response: OllamaEmbeddingResponse = serde_json::from_str(&json).unwrap();

            assert_eq!(response.embeddings.len(), 1);
            assert_eq!(response.embeddings[0].len(), 768);
            assert_eq!(response.prompt_eval_count, Some(12));
        }

        #[test]
        fn parses_batch_embedding_response() {
            let v1: Vec<f32> = vec![0.1; 384];
            let v2: Vec<f32> = vec![0.2; 384];
            let v3: Vec<f32> = vec![0.3; 384];

            let json = format!(
                r#"{{
                    "embeddings": [{}, {}, {}],
                    "prompt_eval_count": 45
                }}"#,
                serde_json::to_string(&v1).unwrap(),
                serde_json::to_string(&v2).unwrap(),
                serde_json::to_string(&v3).unwrap()
            );

            let response: OllamaEmbeddingResponse = serde_json::from_str(&json).unwrap();

            assert_eq!(response.embeddings.len(), 3);
            assert_eq!(response.prompt_eval_count, Some(45));
        }

        #[test]
        fn parses_mxbai_embed_large_response() {
            // mxbai-embed-large produces 1024-dimensional embeddings
            let vector: Vec<f32> = (0..1024).map(|i| (i as f32 / 1024.0) - 0.5).collect();
            let json = format!(
                r#"{{
                    "embeddings": [{}],
                    "prompt_eval_count": 8
                }}"#,
                serde_json::to_string(&vector).unwrap()
            );

            let response: OllamaEmbeddingResponse = serde_json::from_str(&json).unwrap();

            assert_eq!(response.embeddings[0].len(), 1024);
        }
    }

    mod constants {
        use super::*;

        #[test]
        fn default_model_is_nomic_embed_text() {
            assert_eq!(DEFAULT_EMBEDDING_MODEL, "nomic-embed-text");
        }
    }
}
