//! Core traits for the embedding module.
//!
//! This module defines the [`Embed`] trait which must be implemented for types
//! that can be embedded by the [`crate::embedding::EmbeddingsBuilder`].

/// Error type used for when the [Embed::embed] method of the [Embed] trait fails.
/// Used by default implementations of [Embed] for common types.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct EmbedError(#[from] Box<dyn std::error::Error + Send + Sync>);

impl EmbedError {
    pub fn new<E: std::error::Error + Send + Sync + 'static>(error: E) -> Self {
        EmbedError(Box::new(error))
    }
}

/// Derive this trait for objects that need to be converted to vector embeddings.
/// The [Embed::embed] method accumulates string values that need to be embedded by adding them to the [TextEmbedder].
/// If an error occurs, the method should return [EmbedError].
/// # Example
/// ```rust
/// use std::env;
///
/// use serde::{Deserialize, Serialize};
/// use crate::{Embed, embedding::{TextEmbedder, EmbedError}};
///
/// struct WordDefinition {
///     id: String,
///     word: String,
///     definitions: String,
/// }
///
/// impl Embed for WordDefinition {
///     fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
///        // Embeddings only need to be generated for `definition` field.
///        // Split the definitions by comma and collect them into a vector of strings.
///        // That way, different embeddings can be generated for each definition in the `definitions` string.
///        self.definitions
///            .split(",")
///            .for_each(|s| {
///                embedder.embed(s.to_string());
///            });
///
///        Ok(())
///     }
/// }
///
/// let fake_definition = WordDefinition {
///    id: "1".to_string(),
///    word: "apple".to_string(),
///    definitions: "a fruit, a tech company".to_string(),
/// };
///
/// assert_eq!(embedding::to_texts(fake_definition).unwrap(), vec!["a fruit", " a tech company"]);
/// ```
pub trait Embed {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError>;
}

/// Accumulates string values that need to be embedded.
/// Used by the [Embed] trait.
#[derive(Default)]
pub struct TextEmbedder {
    pub(crate) texts: Vec<String>,
}

impl TextEmbedder {
    /// Adds input `text` string to the list of texts in the [TextEmbedder] that need to be embedded.
    pub fn embed(&mut self, text: String) {
        self.texts.push(text);
    }
}

/// Utility function that returns a vector of strings that need to be embedded for a
/// given object that implements the [Embed] trait.
pub fn to_texts(item: impl Embed) -> Result<Vec<String>, EmbedError> {
    let mut embedder = TextEmbedder::default();
    item.embed(&mut embedder)?;
    Ok(embedder.texts)
}
