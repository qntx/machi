//! Core vector store traits.
//!
//! This module defines the fundamental traits for vector store operations:
//! - [`InsertDocuments`] - Insert documents and embeddings
//! - [`VectorStoreIndex`] - Query by similarity
//! - [`VectorStoreIndexDyn`] - Type-erased version for dynamic dispatch

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    Embed, OneOrMany,
    core::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
    embedding::Embedding,
    store::request::{Filter, SearchFilter},
};

use super::errors::VectorStoreError;
use super::request::VectorSearchRequest;

/// Trait for inserting documents and embeddings into a vector store.
pub trait InsertDocuments: WasmCompatSend + WasmCompatSync {
    fn insert_documents<Doc: Serialize + Embed + WasmCompatSend>(
        &self,
        documents: Vec<(Doc, OneOrMany<Embedding>)>,
    ) -> impl std::future::Future<Output = Result<(), VectorStoreError>> + WasmCompatSend;
}

/// Trait for querying a vector store by similarity.
pub trait VectorStoreIndex: WasmCompatSend + WasmCompatSync {
    /// The filter type for this backend.
    type Filter: SearchFilter + WasmCompatSend + WasmCompatSync;

    /// Returns the top N most similar documents as `(score, id, document)` tuples.
    fn top_n<T: for<'a> Deserialize<'a> + WasmCompatSend>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, String, T)>, VectorStoreError>>
    + WasmCompatSend;

    /// Returns the top N most similar document IDs as `(score, id)` tuples.
    fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, String)>, VectorStoreError>> + WasmCompatSend;
}

pub type TopNResults = Result<Vec<(f64, String, Value)>, VectorStoreError>;

/// Type-erased [`VectorStoreIndex`] for dynamic dispatch.
pub trait VectorStoreIndexDyn: WasmCompatSend + WasmCompatSync {
    fn top_n<'a>(
        &'a self,
        req: VectorSearchRequest<Filter<serde_json::Value>>,
    ) -> WasmBoxedFuture<'a, TopNResults>;

    fn top_n_ids<'a>(
        &'a self,
        req: VectorSearchRequest<Filter<serde_json::Value>>,
    ) -> WasmBoxedFuture<'a, Result<Vec<(f64, String)>, VectorStoreError>>;
}

impl<I: VectorStoreIndex<Filter = F>, F> VectorStoreIndexDyn for I
where
    F: std::fmt::Debug
        + Clone
        + SearchFilter<Value = serde_json::Value>
        + WasmCompatSend
        + WasmCompatSync
        + Serialize
        + for<'de> Deserialize<'de>
        + 'static,
{
    fn top_n<'a>(
        &'a self,
        req: VectorSearchRequest<Filter<serde_json::Value>>,
    ) -> WasmBoxedFuture<'a, TopNResults> {
        let req = req.map_filter(Filter::interpret);

        Box::pin(async move {
            Ok(self
                .top_n::<serde_json::Value>(req)
                .await?
                .into_iter()
                .map(|(score, id, doc)| (score, id, prune_document(doc).unwrap_or_default()))
                .collect::<Vec<_>>())
        })
    }

    fn top_n_ids<'a>(
        &'a self,
        req: VectorSearchRequest<Filter<serde_json::Value>>,
    ) -> WasmBoxedFuture<'a, Result<Vec<(f64, String)>, VectorStoreError>> {
        let req = req.map_filter(Filter::interpret);

        Box::pin(self.top_n_ids(req))
    }
}

fn prune_document(document: serde_json::Value) -> Option<serde_json::Value> {
    match document {
        Value::Object(mut map) => {
            let new_map = map
                .iter_mut()
                .filter_map(|(key, value)| {
                    prune_document(value.take()).map(|value| (key.clone(), value))
                })
                .collect::<serde_json::Map<_, _>>();

            Some(Value::Object(new_map))
        }
        Value::Array(vec) if vec.len() > 400 => None,
        Value::Array(vec) => Some(Value::Array(
            vec.into_iter().filter_map(prune_document).collect(),
        )),
        Value::Number(num) => Some(Value::Number(num)),
        Value::String(s) => Some(Value::String(s)),
        Value::Bool(b) => Some(Value::Bool(b)),
        Value::Null => Some(Value::Null),
    }
}
