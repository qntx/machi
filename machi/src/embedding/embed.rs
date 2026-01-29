//! The module defines the [Embed] trait, which must be implemented for types
//! that can be embedded by the [crate::embedding::EmbeddingsBuilder].
//!
//! The module also defines the [EmbedError] struct which is used for when the [Embed::embed]
//! method of the [Embed] trait fails.
//!
//! The module also defines the [TextEmbedder] struct which accumulates string values that need to be embedded.
//! It is used directly with the [Embed] trait.
//!
//! Finally, the module implements [Embed] for many common primitive types.

// Re-export core types from traits module
pub use super::traits::{Embed, EmbedError, TextEmbedder, to_texts};

impl Embed for String {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.clone());
        Ok(())
    }
}

impl Embed for &str {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for i8 {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for i16 {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for i32 {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for i64 {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for i128 {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for f32 {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for f64 {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for bool {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for char {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for serde_json::Value {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(serde_json::to_string(self).map_err(EmbedError::new)?);
        Ok(())
    }
}

impl<T: Embed> Embed for &T {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        (*self).embed(embedder)
    }
}

impl<T: Embed> Embed for Vec<T> {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        for item in self {
            item.embed(embedder).map_err(EmbedError::new)?;
        }
        Ok(())
    }
}
