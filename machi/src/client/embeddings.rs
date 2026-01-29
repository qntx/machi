use crate::embedding::Embed;
use crate::embedding::{EmbeddingModel, EmbeddingsBuilder};

/// A provider client with embedding capabilities.
/// Clone is required for conversions between client types.
pub trait EmbeddingsClient {
    /// The type of EmbeddingModel used by the Client
    type EmbeddingModel: EmbeddingModel;

    /// Create an embedding model with the given model.
    fn embedding_model(&self, model: impl Into<String>) -> Self::EmbeddingModel;

    /// Create an embedding model with the given model identifier string and the number of dimensions.
    fn embedding_model_with_ndims(
        &self,
        model: impl Into<String>,
        ndims: usize,
    ) -> Self::EmbeddingModel;

    /// Create an embedding builder with the given embedding model.
    fn embeddings<D: Embed>(
        &self,
        model: impl Into<String>,
    ) -> EmbeddingsBuilder<Self::EmbeddingModel, D> {
        EmbeddingsBuilder::new(self.embedding_model(model))
    }

    /// Create an embedding builder with the given name and dimensions.
    fn embeddings_with_ndims<D: Embed>(
        &self,
        model: &str,
        ndims: usize,
    ) -> EmbeddingsBuilder<Self::EmbeddingModel, D> {
        EmbeddingsBuilder::new(self.embedding_model_with_ndims(model, ndims))
    }
}
