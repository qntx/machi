//! Core traits for the embedding module.
//!
//! This module defines the fundamental traits for embedding operations:
//! - [`EmbeddingModel`] - Trait for embedding models that generate text embeddings
//! - [`ImageEmbeddingModel`] - Trait for embedding models that generate image embeddings
//! - [`VectorDistance`] - Trait for computing distances between embedding vectors

use crate::core::wasm_compat::*;

use super::{Embedding, EmbeddingError};

/// Trait for embedding models that can generate embeddings for documents.
pub trait EmbeddingModel: WasmCompatSend + WasmCompatSync {
    /// The maximum number of documents that can be embedded in a single request.
    const MAX_DOCUMENTS: usize;

    type Client;

    fn make(client: &Self::Client, model: impl Into<String>, dims: Option<usize>) -> Self;

    /// The number of dimensions in the embedding vector.
    fn ndims(&self) -> usize;

    /// Embed multiple text documents in a single request
    fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + WasmCompatSend,
    ) -> impl std::future::Future<Output = Result<Vec<Embedding>, EmbeddingError>> + WasmCompatSend;

    /// Embed a single text document.
    fn embed_text(
        &self,
        text: &str,
    ) -> impl std::future::Future<Output = Result<Embedding, EmbeddingError>> + WasmCompatSend {
        async {
            Ok(self
                .embed_texts(vec![text.to_string()])
                .await?
                .pop()
                .expect("There should be at least one embedding"))
        }
    }
}

/// Trait for embedding models that can generate embeddings for images.
pub trait ImageEmbeddingModel: Clone + WasmCompatSend + WasmCompatSync {
    /// The maximum number of images that can be embedded in a single request.
    const MAX_DOCUMENTS: usize;

    /// The number of dimensions in the embedding vector.
    fn ndims(&self) -> usize;

    /// Embed multiple images in a single request from bytes.
    fn embed_images(
        &self,
        images: impl IntoIterator<Item = Vec<u8>> + WasmCompatSend,
    ) -> impl std::future::Future<Output = Result<Vec<Embedding>, EmbeddingError>> + Send;

    /// Embed a single image from bytes.
    fn embed_image<'a>(
        &'a self,
        bytes: &'a [u8],
    ) -> impl std::future::Future<Output = Result<Embedding, EmbeddingError>> + WasmCompatSend {
        async move {
            Ok(self
                .embed_images(vec![bytes.to_owned()])
                .await?
                .pop()
                .expect("There should be at least one embedding"))
        }
    }
}

/// Trait for computing distances between embedding vectors.
pub trait VectorDistance {
    /// Get dot product of two embedding vectors
    fn dot_product(&self, other: &Self) -> f64;

    /// Get cosine similarity of two embedding vectors.
    /// If `normalized` is true, the dot product is returned.
    fn cosine_similarity(&self, other: &Self, normalized: bool) -> f64;

    /// Get angular distance of two embedding vectors.
    fn angular_distance(&self, other: &Self, normalized: bool) -> f64;

    /// Get euclidean distance of two embedding vectors.
    fn euclidean_distance(&self, other: &Self) -> f64;

    /// Get manhattan distance of two embedding vectors.
    fn manhattan_distance(&self, other: &Self) -> f64;

    /// Get chebyshev distance of two embedding vectors.
    fn chebyshev_distance(&self, other: &Self) -> f64;
}
