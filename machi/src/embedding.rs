//! Embedding provider trait and types.
//!
//! This module defines the interface for text embedding operations,
//! which convert text into dense vector representations.
//!
//! # Examples
//!
//! ```rust
//! use machi::embedding::{Embedding, EmbeddingRequest};
//!
//! // Build a request
//! let request = EmbeddingRequest::new("text-embedding-3-small", vec![
//!     "First text".to_string(),
//!     "Second text".to_string(),
//! ]).dimensions(256);
//!
//! // Compute cosine similarity between two embeddings
//! let a = Embedding::new(vec![1.0, 0.0, 0.0], 0);
//! let b = Embedding::new(vec![0.0, 1.0, 0.0], 1);
//! assert_eq!(a.cosine_similarity(&b), 0.0);
//! assert_eq!(a.dimension(), 3);
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
    /// Total tokens used (same as `prompt_tokens` for embeddings).
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
}

impl EmbeddingResponse {
    /// Create a new embedding response.
    #[must_use]
    pub const fn new(embeddings: Vec<Embedding>) -> Self {
        Self {
            embeddings,
            model: None,
            usage: None,
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
        self.usage.as_ref().map(|u| u.total_tokens)
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
