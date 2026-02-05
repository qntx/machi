//! Embedding provider trait and types.
//!
//! This module defines the interface for text embedding operations,
//! which convert text into dense vector representations.
//!
//! # Example
//!
//! ```rust,ignore
//! use machi::prelude::*;
//!
//! // Single text embedding
//! let embedding = provider.embed_single("text-embedding-3-small", "Hello, world!").await?;
//! println!("Dimension: {}", embedding.dimension());
//!
//! // Batch embedding
//! let request = EmbeddingRequest::new("text-embedding-3-small", vec![
//!     "First text".to_string(),
//!     "Second text".to_string(),
//! ]).dimensions(256);
//! let response = provider.embed(&request).await?;
//!
//! // Compute similarity
//! let similarity = response.embeddings[0].cosine_similarity(&response.embeddings[1]);
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Encoding format for embedding output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EncodingFormat {
    /// Float format (default) - returns vectors as arrays of floats.
    #[default]
    Float,
    /// Base64 format - returns vectors as base64-encoded strings.
    Base64,
}

impl EncodingFormat {
    /// Get the format string for API requests.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Float => "float",
            Self::Base64 => "base64",
        }
    }
}

/// Request for generating embeddings.
///
/// # Models
/// - `text-embedding-3-small`: 1536 dimensions (default), fast and cost-effective
/// - `text-embedding-3-large`: 3072 dimensions, higher quality
/// - `text-embedding-ada-002`: 1536 dimensions (legacy)
///
/// # Limits
/// - Max input tokens: 8192 per input
/// - Max total tokens: 300,000 across all inputs in a single request
/// - Max array size: 2048 inputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// Model to use for embedding.
    pub model: String,
    /// Input texts to embed.
    pub input: Vec<String>,
    /// Encoding format for the output vectors.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<EncodingFormat>,
    /// Number of dimensions for output vectors (text-embedding-3 models only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    /// Unique identifier for end-user monitoring and abuse detection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl EmbeddingRequest {
    /// Create a new embedding request.
    #[must_use]
    pub fn new(model: impl Into<String>, input: Vec<String>) -> Self {
        Self {
            model: model.into(),
            input,
            encoding_format: None,
            dimensions: None,
            user: None,
        }
    }

    /// Create a request for a single text.
    #[must_use]
    pub fn single(model: impl Into<String>, text: impl Into<String>) -> Self {
        Self::new(model, vec![text.into()])
    }

    /// Set the encoding format.
    #[must_use]
    pub const fn encoding_format(mut self, format: EncodingFormat) -> Self {
        self.encoding_format = Some(format);
        self
    }

    /// Set the output dimensions (text-embedding-3 models only).
    ///
    /// Lower dimensions reduce storage and improve search speed,
    /// but may slightly reduce quality.
    #[must_use]
    pub const fn dimensions(mut self, dims: u32) -> Self {
        self.dimensions = Some(dims);
        self
    }

    /// Set the user identifier for monitoring and abuse detection.
    #[must_use]
    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }
}

/// A single embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// The embedding vector.
    pub vector: Vec<f32>,
    /// Index of the input text this embedding corresponds to.
    pub index: usize,
}

impl Embedding {
    /// Create a new embedding.
    #[must_use]
    pub const fn new(vector: Vec<f32>, index: usize) -> Self {
        Self { vector, index }
    }

    /// Get the dimension of the embedding.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.vector.len()
    }

    /// Compute cosine similarity with another embedding.
    #[must_use]
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        if self.vector.len() != other.vector.len() {
            return 0.0;
        }

        let dot_product: f32 = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }

    /// Compute Euclidean distance to another embedding.
    #[must_use]
    pub fn euclidean_distance(&self, other: &Self) -> f32 {
        if self.vector.len() != other.vector.len() {
            return f32::MAX;
        }

        self.vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

/// Token usage statistics for embedding requests.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    /// Number of tokens in the input prompt.
    pub prompt_tokens: u32,
    /// Total tokens used (same as prompt_tokens for embeddings).
    pub total_tokens: u32,
}

/// Response from an embedding request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// The generated embeddings.
    pub embeddings: Vec<Embedding>,
    /// Model used for embedding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Token usage statistics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<EmbeddingUsage>,
    /// Total tokens used (convenience field, same as usage.total_tokens).
    #[serde(skip)]
    pub total_tokens: Option<u32>,
}

impl EmbeddingResponse {
    /// Create a new embedding response.
    #[must_use]
    pub const fn new(embeddings: Vec<Embedding>) -> Self {
        Self {
            embeddings,
            model: None,
            usage: None,
            total_tokens: None,
        }
    }

    /// Set the model name.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the usage statistics.
    #[must_use]
    pub const fn with_usage(mut self, prompt_tokens: u32, total_tokens: u32) -> Self {
        self.usage = Some(EmbeddingUsage {
            prompt_tokens,
            total_tokens,
        });
        self.total_tokens = Some(total_tokens);
        self
    }

    /// Get the first embedding vector.
    #[must_use]
    pub fn first(&self) -> Option<&Embedding> {
        self.embeddings.first()
    }

    /// Get all embedding vectors.
    #[must_use]
    pub fn vectors(&self) -> Vec<&Vec<f32>> {
        self.embeddings.iter().map(|e| &e.vector).collect()
    }

    /// Get the total number of tokens used.
    #[must_use]
    pub fn tokens_used(&self) -> Option<u32> {
        self.total_tokens
            .or_else(|| self.usage.as_ref().map(|u| u.total_tokens))
    }
}

/// Trait for providers that support text embeddings.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embeddings for the given texts.
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse>;

    /// Generate embedding for a single text.
    async fn embed_single(&self, model: &str, text: &str) -> Result<Embedding> {
        let request = EmbeddingRequest::single(model, text);
        let response = self.embed(&request).await?;
        response.embeddings.into_iter().next().ok_or_else(|| {
            crate::error::LlmError::response_format("embedding", "empty response").into()
        })
    }

    /// Get the default embedding model name.
    fn default_embedding_model(&self) -> &str;

    /// Get the embedding dimension for the default model.
    fn embedding_dimension(&self) -> Option<usize> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod encoding_format {
        use super::*;

        #[test]
        fn default_is_float() {
            assert_eq!(EncodingFormat::default(), EncodingFormat::Float);
        }

        #[test]
        fn as_str_returns_correct_values() {
            assert_eq!(EncodingFormat::Float.as_str(), "float");
            assert_eq!(EncodingFormat::Base64.as_str(), "base64");
        }

        #[test]
        fn serde_uses_lowercase() {
            assert_eq!(
                serde_json::to_string(&EncodingFormat::Float).unwrap(),
                r#""float""#
            );
            assert_eq!(
                serde_json::to_string(&EncodingFormat::Base64).unwrap(),
                r#""base64""#
            );
        }

        #[test]
        fn serde_roundtrip() {
            for format in [EncodingFormat::Float, EncodingFormat::Base64] {
                let json = serde_json::to_string(&format).unwrap();
                let parsed: EncodingFormat = serde_json::from_str(&json).unwrap();
                assert_eq!(parsed, format);
            }
        }

        #[test]
        fn copy_trait() {
            let f1 = EncodingFormat::Float;
            let f2 = f1;
            assert_eq!(f1, f2);
        }
    }

    mod embedding_request {
        use super::*;

        #[test]
        fn new_creates_with_model_and_input() {
            let req = EmbeddingRequest::new(
                "text-embedding-3-small",
                vec!["hello".into(), "world".into()],
            );

            assert_eq!(req.model, "text-embedding-3-small");
            assert_eq!(req.input.len(), 2);
            assert!(req.encoding_format.is_none());
            assert!(req.dimensions.is_none());
            assert!(req.user.is_none());
        }

        #[test]
        fn single_creates_with_one_input() {
            let req = EmbeddingRequest::single("text-embedding-3-small", "hello");

            assert_eq!(req.model, "text-embedding-3-small");
            assert_eq!(req.input.len(), 1);
            assert_eq!(req.input[0], "hello");
        }

        #[test]
        fn encoding_format_sets_value() {
            let req =
                EmbeddingRequest::single("model", "text").encoding_format(EncodingFormat::Base64);
            assert_eq!(req.encoding_format, Some(EncodingFormat::Base64));
        }

        #[test]
        fn dimensions_sets_value() {
            let req = EmbeddingRequest::single("model", "text").dimensions(256);
            assert_eq!(req.dimensions, Some(256));
        }

        #[test]
        fn user_sets_value() {
            let req = EmbeddingRequest::single("model", "text").user("user-123");
            assert_eq!(req.user.as_deref(), Some("user-123"));
        }

        #[test]
        fn builder_chain() {
            let req = EmbeddingRequest::new("text-embedding-3-large", vec!["test".into()])
                .encoding_format(EncodingFormat::Float)
                .dimensions(512)
                .user("user-abc");

            assert_eq!(req.model, "text-embedding-3-large");
            assert_eq!(req.encoding_format, Some(EncodingFormat::Float));
            assert_eq!(req.dimensions, Some(512));
            assert_eq!(req.user.as_deref(), Some("user-abc"));
        }

        #[test]
        fn serde_skips_none_values() {
            let req = EmbeddingRequest::single("model", "text");
            let json = serde_json::to_string(&req).unwrap();

            assert!(json.contains("model"));
            assert!(json.contains("input"));
            assert!(!json.contains("encoding_format"));
            assert!(!json.contains("dimensions"));
            assert!(!json.contains("user"));
        }

        #[test]
        fn serde_roundtrip() {
            let req = EmbeddingRequest::new("model", vec!["a".into(), "b".into()]).dimensions(256);

            let json = serde_json::to_string(&req).unwrap();
            let parsed: EmbeddingRequest = serde_json::from_str(&json).unwrap();

            assert_eq!(parsed.model, req.model);
            assert_eq!(parsed.input, req.input);
            assert_eq!(parsed.dimensions, req.dimensions);
        }
    }

    mod embedding {
        use super::*;

        #[test]
        fn new_creates_embedding() {
            let emb = Embedding::new(vec![1.0, 2.0, 3.0], 5);

            assert_eq!(emb.vector, vec![1.0, 2.0, 3.0]);
            assert_eq!(emb.index, 5);
        }

        #[test]
        fn dimension_returns_vector_length() {
            let emb = Embedding::new(vec![0.0; 1536], 0);
            assert_eq!(emb.dimension(), 1536);

            let empty = Embedding::new(vec![], 0);
            assert_eq!(empty.dimension(), 0);
        }

        #[test]
        fn cosine_similarity_identical_vectors() {
            let e1 = Embedding::new(vec![1.0, 0.0, 0.0], 0);
            let e2 = Embedding::new(vec![1.0, 0.0, 0.0], 1);

            let sim = e1.cosine_similarity(&e2);
            assert!((sim - 1.0).abs() < 1e-6);
        }

        #[test]
        fn cosine_similarity_orthogonal_vectors() {
            let e1 = Embedding::new(vec![1.0, 0.0, 0.0], 0);
            let e2 = Embedding::new(vec![0.0, 1.0, 0.0], 1);

            let sim = e1.cosine_similarity(&e2);
            assert!(sim.abs() < 1e-6);
        }

        #[test]
        fn cosine_similarity_opposite_vectors() {
            let e1 = Embedding::new(vec![1.0, 0.0], 0);
            let e2 = Embedding::new(vec![-1.0, 0.0], 1);

            let sim = e1.cosine_similarity(&e2);
            assert!((sim + 1.0).abs() < 1e-6);
        }

        #[test]
        fn cosine_similarity_different_dimensions() {
            let e1 = Embedding::new(vec![1.0, 0.0], 0);
            let e2 = Embedding::new(vec![1.0, 0.0, 0.0], 1);

            assert_eq!(e1.cosine_similarity(&e2), 0.0);
        }

        #[test]
        fn cosine_similarity_zero_vector() {
            let e1 = Embedding::new(vec![0.0, 0.0, 0.0], 0);
            let e2 = Embedding::new(vec![1.0, 0.0, 0.0], 1);

            assert_eq!(e1.cosine_similarity(&e2), 0.0);
        }

        #[test]
        fn cosine_similarity_normalized_vectors() {
            let e1 = Embedding::new(vec![0.6, 0.8], 0);
            let e2 = Embedding::new(vec![0.8, 0.6], 1);

            let sim = e1.cosine_similarity(&e2);
            let expected = 0.6 * 0.8 + 0.8 * 0.6; // = 0.96
            assert!((sim - expected).abs() < 1e-6);
        }

        #[test]
        fn euclidean_distance_same_point() {
            let e1 = Embedding::new(vec![1.0, 2.0, 3.0], 0);
            let e2 = Embedding::new(vec![1.0, 2.0, 3.0], 1);

            assert!(e1.euclidean_distance(&e2).abs() < 1e-6);
        }

        #[test]
        fn euclidean_distance_3_4_5_triangle() {
            let e1 = Embedding::new(vec![0.0, 0.0], 0);
            let e2 = Embedding::new(vec![3.0, 4.0], 1);

            let dist = e1.euclidean_distance(&e2);
            assert!((dist - 5.0).abs() < 1e-6);
        }

        #[test]
        fn euclidean_distance_different_dimensions() {
            let e1 = Embedding::new(vec![1.0, 0.0], 0);
            let e2 = Embedding::new(vec![1.0, 0.0, 0.0], 1);

            assert_eq!(e1.euclidean_distance(&e2), f32::MAX);
        }

        #[test]
        fn euclidean_distance_unit_vectors() {
            let e1 = Embedding::new(vec![1.0, 0.0], 0);
            let e2 = Embedding::new(vec![0.0, 1.0], 1);

            let dist = e1.euclidean_distance(&e2);
            let expected = 2.0_f32.sqrt();
            assert!((dist - expected).abs() < 1e-6);
        }

        #[test]
        fn serde_roundtrip() {
            let emb = Embedding::new(vec![0.1, 0.2, 0.3], 42);
            let json = serde_json::to_string(&emb).unwrap();
            let parsed: Embedding = serde_json::from_str(&json).unwrap();

            assert_eq!(parsed.index, 42);
            assert_eq!(parsed.vector.len(), 3);
        }
    }

    mod embedding_usage {
        use super::*;

        #[test]
        fn default_is_zero() {
            let usage = EmbeddingUsage::default();
            assert_eq!(usage.prompt_tokens, 0);
            assert_eq!(usage.total_tokens, 0);
        }

        #[test]
        fn serde_roundtrip() {
            let usage = EmbeddingUsage {
                prompt_tokens: 100,
                total_tokens: 100,
            };

            let json = serde_json::to_string(&usage).unwrap();
            let parsed: EmbeddingUsage = serde_json::from_str(&json).unwrap();

            assert_eq!(parsed.prompt_tokens, 100);
            assert_eq!(parsed.total_tokens, 100);
        }

        #[test]
        fn copy_trait() {
            let u1 = EmbeddingUsage {
                prompt_tokens: 50,
                total_tokens: 50,
            };
            let u2 = u1;
            assert_eq!(u1.prompt_tokens, u2.prompt_tokens);
        }
    }

    mod embedding_response {
        use super::*;

        #[test]
        fn new_creates_with_embeddings() {
            let embeddings = vec![
                Embedding::new(vec![1.0, 2.0], 0),
                Embedding::new(vec![3.0, 4.0], 1),
            ];
            let resp = EmbeddingResponse::new(embeddings);

            assert_eq!(resp.embeddings.len(), 2);
            assert!(resp.model.is_none());
            assert!(resp.usage.is_none());
            assert!(resp.total_tokens.is_none());
        }

        #[test]
        fn default_is_empty() {
            let resp = EmbeddingResponse::default();
            assert!(resp.embeddings.is_empty());
            assert!(resp.model.is_none());
        }

        #[test]
        fn with_model_sets_value() {
            let resp = EmbeddingResponse::new(vec![]).with_model("text-embedding-3-small");
            assert_eq!(resp.model.as_deref(), Some("text-embedding-3-small"));
        }

        #[test]
        fn with_usage_sets_values() {
            let resp = EmbeddingResponse::new(vec![]).with_usage(50, 50);

            let usage = resp.usage.unwrap();
            assert_eq!(usage.prompt_tokens, 50);
            assert_eq!(usage.total_tokens, 50);
            assert_eq!(resp.total_tokens, Some(50));
        }

        #[test]
        fn first_returns_first_embedding() {
            let embeddings = vec![Embedding::new(vec![1.0], 0), Embedding::new(vec![2.0], 1)];
            let resp = EmbeddingResponse::new(embeddings);

            let first = resp.first().unwrap();
            assert_eq!(first.index, 0);
            assert_eq!(first.vector, vec![1.0]);
        }

        #[test]
        fn first_returns_none_for_empty() {
            let resp = EmbeddingResponse::new(vec![]);
            assert!(resp.first().is_none());
        }

        #[test]
        fn vectors_returns_all_vectors() {
            let embeddings = vec![
                Embedding::new(vec![1.0, 2.0], 0),
                Embedding::new(vec![3.0, 4.0], 1),
            ];
            let resp = EmbeddingResponse::new(embeddings);

            let vectors = resp.vectors();
            assert_eq!(vectors.len(), 2);
            assert_eq!(*vectors[0], vec![1.0, 2.0]);
            assert_eq!(*vectors[1], vec![3.0, 4.0]);
        }

        #[test]
        fn tokens_used_from_total_tokens() {
            let mut resp = EmbeddingResponse::new(vec![]);
            resp.total_tokens = Some(100);

            assert_eq!(resp.tokens_used(), Some(100));
        }

        #[test]
        fn tokens_used_from_usage() {
            let resp = EmbeddingResponse::new(vec![]).with_usage(75, 75);
            assert_eq!(resp.tokens_used(), Some(75));
        }

        #[test]
        fn tokens_used_prefers_total_tokens() {
            let mut resp = EmbeddingResponse::new(vec![]).with_usage(50, 50);
            resp.total_tokens = Some(100);

            assert_eq!(resp.tokens_used(), Some(100));
        }

        #[test]
        fn tokens_used_returns_none_when_empty() {
            let resp = EmbeddingResponse::new(vec![]);
            assert!(resp.tokens_used().is_none());
        }

        #[test]
        fn builder_chain() {
            let resp = EmbeddingResponse::new(vec![Embedding::new(vec![1.0], 0)])
                .with_model("model-name")
                .with_usage(10, 10);

            assert_eq!(resp.embeddings.len(), 1);
            assert_eq!(resp.model.as_deref(), Some("model-name"));
            assert!(resp.usage.is_some());
        }

        #[test]
        fn serde_skips_none_values() {
            let resp = EmbeddingResponse::new(vec![]);
            let json = serde_json::to_string(&resp).unwrap();

            assert!(json.contains("embeddings"));
            assert!(!json.contains("model"));
            assert!(!json.contains("usage"));
        }

        #[test]
        fn serde_skips_total_tokens() {
            let mut resp = EmbeddingResponse::new(vec![]);
            resp.total_tokens = Some(100);

            let json = serde_json::to_string(&resp).unwrap();
            assert!(!json.contains("total_tokens"));
        }
    }

    mod integration {
        use super::*;

        #[test]
        fn embedding_request_json_structure() {
            let req = EmbeddingRequest::new("text-embedding-3-small", vec!["hello".into()])
                .dimensions(256)
                .encoding_format(EncodingFormat::Float);

            let json: serde_json::Value = serde_json::to_value(&req).unwrap();

            assert_eq!(json["model"], "text-embedding-3-small");
            assert_eq!(json["input"].as_array().unwrap().len(), 1);
            assert_eq!(json["dimensions"], 256);
            assert_eq!(json["encoding_format"], "float");
        }

        #[test]
        fn similarity_search_workflow() {
            let query = Embedding::new(vec![1.0, 0.0, 0.0], 0);
            let docs = vec![
                Embedding::new(vec![0.9, 0.1, 0.0], 0),
                Embedding::new(vec![0.0, 1.0, 0.0], 1),
                Embedding::new(vec![0.7, 0.7, 0.0], 2),
            ];

            let mut scores: Vec<(usize, f32)> = docs
                .iter()
                .enumerate()
                .map(|(i, doc)| (i, query.cosine_similarity(doc)))
                .collect();

            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            assert_eq!(scores[0].0, 0);
            assert_eq!(scores[1].0, 2);
            assert_eq!(scores[2].0, 1);
        }

        #[test]
        fn response_with_multiple_embeddings() {
            let embeddings = vec![
                Embedding::new(vec![0.1, 0.2], 0),
                Embedding::new(vec![0.3, 0.4], 1),
                Embedding::new(vec![0.5, 0.6], 2),
            ];

            let resp = EmbeddingResponse::new(embeddings)
                .with_model("text-embedding-3-small")
                .with_usage(30, 30);

            assert_eq!(resp.embeddings.len(), 3);
            assert_eq!(resp.vectors().len(), 3);
            assert_eq!(resp.tokens_used(), Some(30));

            let sim = resp.embeddings[0].cosine_similarity(&resp.embeddings[1]);
            assert!(sim > 0.0);
        }
    }
}
