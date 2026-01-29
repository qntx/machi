//! This module provides functionality for working with embeddings.
//! Embeddings are numerical representations of documents or other objects, typically used in
//! natural language processing (NLP) tasks such as text classification, information retrieval,
//! and document similarity.

use serde::{Deserialize, Serialize};

pub mod builder;
pub mod distance;
pub mod embed;
pub mod errors;
pub mod tool;
pub mod traits;

pub use builder::EmbeddingsBuilder;
pub use distance::VectorDistance;
pub use embed::{Embed, EmbedError, TextEmbedder, to_texts};
pub use errors::EmbeddingError;
pub use tool::ToolSchema;
pub use traits::{EmbeddingModel, ImageEmbeddingModel};

/// Struct that holds a single document and its embedding.
#[derive(Clone, Default, Deserialize, Serialize, Debug)]
pub struct Embedding {
    /// The document that was embedded. Used for debugging.
    pub document: String,
    /// The embedding vector
    pub vec: Vec<f64>,
}

impl PartialEq for Embedding {
    fn eq(&self, other: &Self) -> bool {
        self.document == other.document
    }
}

impl Eq for Embedding {}
