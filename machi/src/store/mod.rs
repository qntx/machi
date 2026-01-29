//! Vector store abstractions for semantic search and retrieval.
//!
//! # Core Traits
//!
//! - [`VectorStoreIndex`]: Query a vector store for similar documents.
//! - [`InsertDocuments`]: Insert documents and their embeddings.
//! - [`VectorStoreIndexDyn`]: Type-erased version for dynamic contexts.
//!
//! Use [`VectorSearchRequest`] to build queries. See [`request`] for filtering.
//!
//! Types implementing [`VectorStoreIndex`] automatically implement [`Tool`].

pub mod builder;
pub mod errors;
pub mod in_memory_store;
pub mod lsh;
pub mod request;
pub mod traits;

pub use errors::VectorStoreError;
pub use request::VectorSearchRequest;
pub use traits::{InsertDocuments, TopNResults, VectorStoreIndex, VectorStoreIndexDyn};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    completion::ToolDefinition,
    core::wasm_compat::{WasmCompatSend, WasmCompatSync},
    store::request::SearchFilter,
    tool::Tool,
};

/// The output of vector store queries invoked via [`Tool`]
#[derive(Serialize, Deserialize, Debug)]
pub struct VectorStoreOutput {
    pub score: f64,
    pub id: String,
    pub document: Value,
}

impl<T, F> Tool for T
where
    F: SearchFilter<Value = serde_json::Value>
        + WasmCompatSend
        + WasmCompatSync
        + for<'de> Deserialize<'de>,
    T: VectorStoreIndex<Filter = F>,
{
    const NAME: &'static str = "search_vector_store";

    type Error = VectorStoreError;
    type Args = VectorSearchRequest<F>;
    type Output = Vec<VectorStoreOutput>;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description:
                "Retrieves the most relevant documents from a vector store based on a query."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query string to search for relevant documents in the vector store."
                    },
                    "samples": {
                        "type": "integer",
                        "description": "The maxinum number of samples / documents to retrieve.",
                        "default": 5,
                        "minimum": 1
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Similarity search threshold. If present, any result with a distance less than this may be omitted from the final result."
                    }
                },
                "required": ["query", "samples"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let results = self.top_n(args).await?;
        Ok(results
            .into_iter()
            .map(|(score, id, document)| VectorStoreOutput {
                score,
                id,
                document,
            })
            .collect())
    }
}

/// Index strategy for the super::InMemoryVectorStore
#[derive(Clone, Debug)]
pub enum IndexStrategy {
    /// Checks all documents in the vector store to find the most relevant documents.
    BruteForce,

    /// Uses LSH to find candidates then computes exact distances.
    LSH {
        /// Number of tables to use for LSH.
        num_tables: usize,
        /// Number of hyperplanes to use for LSH.
        num_hyperplanes: usize,
    },
}

impl Default for IndexStrategy {
    fn default() -> Self {
        Self::BruteForce
    }
}
