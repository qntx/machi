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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    mod constants {
        use super::*;

        #[test]
        fn default_embedding_model_is_text_embedding_3_small() {
            assert_eq!(DEFAULT_EMBEDDING_MODEL, "text-embedding-3-small");
        }

        #[test]
        fn default_embedding_dimension_is_1536() {
            assert_eq!(DEFAULT_EMBEDDING_DIMENSION, 1536);
        }
    }

    mod openai_embedding_request {
        use super::*;

        #[test]
        fn serializes_required_fields() {
            let req = OpenAIEmbeddingRequest {
                model: "text-embedding-3-small".to_owned(),
                input: vec!["Hello world".to_owned()],
                encoding_format: None,
                dimensions: None,
            };

            let json = serde_json::to_value(&req).unwrap();

            assert_eq!(json["model"], "text-embedding-3-small");
            assert!(json["input"].is_array());
            assert_eq!(json["input"][0], "Hello world");
        }

        #[test]
        fn skips_none_optional_fields() {
            let req = OpenAIEmbeddingRequest {
                model: "model".to_owned(),
                input: vec!["text".to_owned()],
                encoding_format: None,
                dimensions: None,
            };

            let json = serde_json::to_string(&req).unwrap();

            assert!(!json.contains("encoding_format"));
            assert!(!json.contains("dimensions"));
        }

        #[test]
        fn includes_encoding_format_when_set() {
            let req = OpenAIEmbeddingRequest {
                model: "model".to_owned(),
                input: vec!["text".to_owned()],
                encoding_format: Some("float".to_owned()),
                dimensions: None,
            };

            let json = serde_json::to_value(&req).unwrap();

            assert_eq!(json["encoding_format"], "float");
        }

        #[test]
        fn includes_dimensions_when_set() {
            let req = OpenAIEmbeddingRequest {
                model: "model".to_owned(),
                input: vec!["text".to_owned()],
                encoding_format: None,
                dimensions: Some(256),
            };

            let json = serde_json::to_value(&req).unwrap();

            assert_eq!(json["dimensions"], 256);
        }

        #[test]
        fn handles_multiple_inputs() {
            let req = OpenAIEmbeddingRequest {
                model: "model".to_owned(),
                input: vec![
                    "First text".to_owned(),
                    "Second text".to_owned(),
                    "Third text".to_owned(),
                ],
                encoding_format: None,
                dimensions: None,
            };

            let json = serde_json::to_value(&req).unwrap();
            let inputs = json["input"].as_array().unwrap();

            assert_eq!(inputs.len(), 3);
            assert_eq!(inputs[0], "First text");
            assert_eq!(inputs[1], "Second text");
            assert_eq!(inputs[2], "Third text");
        }

        #[test]
        fn handles_empty_input_array() {
            let req = OpenAIEmbeddingRequest {
                model: "model".to_owned(),
                input: vec![],
                encoding_format: None,
                dimensions: None,
            };

            let json = serde_json::to_value(&req).unwrap();

            assert!(json["input"].as_array().unwrap().is_empty());
        }

        #[test]
        fn handles_unicode_input() {
            let req = OpenAIEmbeddingRequest {
                model: "model".to_owned(),
                input: vec!["你好世界 🌍".to_owned()],
                encoding_format: None,
                dimensions: None,
            };

            let json = serde_json::to_value(&req).unwrap();

            assert_eq!(json["input"][0], "你好世界 🌍");
        }
    }

    mod openai_embedding_data {
        use super::*;

        #[test]
        fn deserializes_embedding_data() {
            let json = r#"{"embedding": [0.1, 0.2, 0.3], "index": 0}"#;
            let data: OpenAIEmbeddingData = serde_json::from_str(json).unwrap();

            assert_eq!(data.embedding, vec![0.1, 0.2, 0.3]);
            assert_eq!(data.index, 0);
        }

        #[test]
        fn handles_large_embedding_vector() {
            let embedding: Vec<f32> = (0..1536).map(|i| i as f32 * 0.001).collect();
            let json = serde_json::json!({
                "embedding": embedding,
                "index": 5
            });

            let data: OpenAIEmbeddingData = serde_json::from_value(json).unwrap();

            assert_eq!(data.embedding.len(), 1536);
            assert_eq!(data.index, 5);
        }

        #[test]
        fn handles_negative_values() {
            let json = r#"{"embedding": [-0.5, 0.0, 0.5], "index": 0}"#;
            let data: OpenAIEmbeddingData = serde_json::from_str(json).unwrap();

            assert_eq!(data.embedding[0], -0.5);
            assert_eq!(data.embedding[1], 0.0);
            assert_eq!(data.embedding[2], 0.5);
        }
    }

    mod openai_embedding_response {
        use super::*;

        #[test]
        fn deserializes_full_response() {
            let json = r#"{
                "data": [
                    {"embedding": [0.1, 0.2], "index": 0}
                ],
                "model": "text-embedding-3-small",
                "usage": {
                    "prompt_tokens": 10,
                    "total_tokens": 10
                }
            }"#;

            let response: OpenAIEmbeddingResponse = serde_json::from_str(json).unwrap();

            assert_eq!(response.data.len(), 1);
            assert_eq!(response.model, "text-embedding-3-small");
            assert!(response.usage.is_some());
        }

        #[test]
        fn handles_multiple_embeddings() {
            let json = r#"{
                "data": [
                    {"embedding": [0.1], "index": 0},
                    {"embedding": [0.2], "index": 1},
                    {"embedding": [0.3], "index": 2}
                ],
                "model": "text-embedding-3-small",
                "usage": null
            }"#;

            let response: OpenAIEmbeddingResponse = serde_json::from_str(json).unwrap();

            assert_eq!(response.data.len(), 3);
            assert_eq!(response.data[0].index, 0);
            assert_eq!(response.data[1].index, 1);
            assert_eq!(response.data[2].index, 2);
        }

        #[test]
        fn handles_missing_usage() {
            let json = r#"{
                "data": [],
                "model": "model"
            }"#;

            let response: OpenAIEmbeddingResponse = serde_json::from_str(json).unwrap();

            assert!(response.usage.is_none());
        }
    }

    mod openai_embedding_usage {
        use super::*;

        #[test]
        fn deserializes_usage() {
            let json = r#"{"prompt_tokens": 100, "total_tokens": 100}"#;
            let usage: OpenAIEmbeddingUsage = serde_json::from_str(json).unwrap();

            assert_eq!(usage.prompt_tokens, 100);
            assert_eq!(usage.total_tokens, 100);
        }

        #[test]
        fn handles_large_token_counts() {
            let json = r#"{"prompt_tokens": 1000000, "total_tokens": 1000000}"#;
            let usage: OpenAIEmbeddingUsage = serde_json::from_str(json).unwrap();

            assert_eq!(usage.prompt_tokens, 1_000_000);
            assert_eq!(usage.total_tokens, 1_000_000);
        }
    }

    mod embedding_provider_impl {
        use super::*;
        use crate::llms::openai::OpenAIConfig;

        fn test_client() -> OpenAI {
            OpenAI::new(OpenAIConfig::new("test-key")).unwrap()
        }

        #[test]
        fn default_embedding_model_returns_correct_value() {
            use crate::embedding::EmbeddingProvider;

            let client = test_client();
            assert_eq!(client.default_embedding_model(), "text-embedding-3-small");
        }

        #[test]
        fn embedding_dimension_returns_1536() {
            use crate::embedding::EmbeddingProvider;

            let client = test_client();
            assert_eq!(client.embedding_dimension(), Some(1536));
        }
    }

    mod response_conversion {
        use super::*;

        #[test]
        fn converts_response_to_embedding() {
            // Simulate what happens in embed() when converting response
            let data = OpenAIEmbeddingData {
                embedding: vec![0.1, 0.2, 0.3],
                index: 0,
            };

            let embedding = Embedding::new(data.embedding.clone(), data.index);

            assert_eq!(embedding.vector, vec![0.1, 0.2, 0.3]);
            assert_eq!(embedding.index, 0);
        }

        #[test]
        fn converts_usage_to_embedding_usage() {
            let openai_usage = OpenAIEmbeddingUsage {
                prompt_tokens: 50,
                total_tokens: 50,
            };

            let usage = EmbeddingUsage {
                prompt_tokens: openai_usage.prompt_tokens,
                total_tokens: openai_usage.total_tokens,
            };

            assert_eq!(usage.prompt_tokens, 50);
            assert_eq!(usage.total_tokens, 50);
        }
    }

    mod edge_cases {
        use super::*;

        #[test]
        fn handles_empty_embedding_vector() {
            let json = r#"{"embedding": [], "index": 0}"#;
            let data: OpenAIEmbeddingData = serde_json::from_str(json).unwrap();

            assert!(data.embedding.is_empty());
        }

        #[test]
        fn handles_very_small_float_values() {
            let json = r#"{"embedding": [1e-10, -1e-10], "index": 0}"#;
            let data: OpenAIEmbeddingData = serde_json::from_str(json).unwrap();

            assert!((data.embedding[0] - 1e-10).abs() < 1e-15);
            assert!((data.embedding[1] - (-1e-10)).abs() < 1e-15);
        }

        #[test]
        fn request_model_names() {
            let models = [
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ];

            for model in models {
                let req = OpenAIEmbeddingRequest {
                    model: model.to_owned(),
                    input: vec!["test".to_owned()],
                    encoding_format: None,
                    dimensions: None,
                };
                let json = serde_json::to_value(&req).unwrap();
                assert_eq!(json["model"], model);
            }
        }

        #[test]
        fn encoding_formats() {
            for format in ["float", "base64"] {
                let req = OpenAIEmbeddingRequest {
                    model: "model".to_owned(),
                    input: vec!["test".to_owned()],
                    encoding_format: Some(format.to_owned()),
                    dimensions: None,
                };
                let json = serde_json::to_value(&req).unwrap();
                assert_eq!(json["encoding_format"], format);
            }
        }

        #[test]
        fn custom_dimensions() {
            for dim in [256_u32, 512, 1024, 1536, 3072] {
                let req = OpenAIEmbeddingRequest {
                    model: "text-embedding-3-large".to_owned(),
                    input: vec!["test".to_owned()],
                    encoding_format: None,
                    dimensions: Some(dim),
                };
                let json = serde_json::to_value(&req).unwrap();
                assert_eq!(json["dimensions"], dim);
            }
        }
    }
}
