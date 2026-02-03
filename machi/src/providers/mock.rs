//! Mock model implementation for testing.
//!
//! This module provides a simple mock model that returns predefined responses,
//! useful for unit testing without making real API calls.

use super::common::{GenerateOptions, Model, ModelResponse};
use crate::error::AgentError;
use crate::message::ChatMessage;
use async_trait::async_trait;

/// A simple mock model for testing.
///
/// Returns predefined responses in sequence, cycling through them.
///
/// # Example
///
/// ```rust,ignore
/// use machi::prelude::*;
///
/// let model = MockModel::new(vec!["Hello!".to_string(), "Goodbye!".to_string()]);
/// // First call returns "Hello!", second returns "Goodbye!", third returns "Hello!" again...
/// ```
#[derive(Debug)]
pub struct MockModel {
    model_id: String,
    responses: Vec<String>,
    response_index: std::sync::atomic::AtomicUsize,
}

impl MockModel {
    /// Create a new mock model with predefined responses.
    #[must_use]
    pub fn new(responses: Vec<String>) -> Self {
        Self {
            model_id: "mock-model".to_string(),
            responses,
            response_index: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Create a mock model with a custom model ID.
    #[must_use]
    pub fn with_model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = model_id.into();
        self
    }
}

#[async_trait]
impl Model for MockModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn generate(
        &self,
        _messages: Vec<ChatMessage>,
        _options: GenerateOptions,
    ) -> Result<ModelResponse, AgentError> {
        let index = self
            .response_index
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let response = self
            .responses
            .get(index % self.responses.len())
            .cloned()
            .unwrap_or_else(|| "No response".to_string());

        Ok(ModelResponse::new(ChatMessage::assistant(response)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_model_cycles_responses() {
        let model = MockModel::new(vec!["first".to_string(), "second".to_string()]);

        let options = GenerateOptions::default();
        let messages = vec![];

        let r1 = model
            .generate(messages.clone(), options.clone())
            .await
            .expect("generate should succeed");
        assert_eq!(r1.text(), Some("first".to_string()));

        let r2 = model
            .generate(messages.clone(), options.clone())
            .await
            .expect("generate should succeed");
        assert_eq!(r2.text(), Some("second".to_string()));

        let r3 = model
            .generate(messages.clone(), options.clone())
            .await
            .expect("generate should succeed");
        assert_eq!(r3.text(), Some("first".to_string()));
    }

    #[test]
    fn test_mock_model_custom_id() {
        let model = MockModel::new(vec!["test".to_string()]).with_model_id("custom-mock");
        assert_eq!(model.model_id(), "custom-mock");
    }
}
