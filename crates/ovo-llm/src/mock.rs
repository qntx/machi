//! Scripted sampler for offline tests.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use ovo_types::{ErrorCode, Message, OvoError, Role, Usage};

use crate::sample::{SampleRequest, SampleResponse};
use crate::sampler::LlmSampler;

/// Queue of scripted responses (FIFO) plus optional prompt-keyed responses.
#[derive(Debug, Default)]
pub struct MockSampler {
    responses: Mutex<Vec<Result<SampleResponse, OvoError>>>,
    /// Exact match on the last user message text → response text.
    by_user_text: Mutex<HashMap<String, String>>,
}

impl MockSampler {
    /// Empty mock.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a FIFO response.
    pub fn push(&self, response: SampleResponse) {
        self.responses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(Ok(response));
    }

    /// Push a FIFO error (for retry / breaker tests).
    pub fn push_error(&self, error: OvoError) {
        self.responses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(Err(error));
    }

    /// Convenience: final assistant text (FIFO).
    pub fn push_text(&self, text: impl Into<String>) {
        self.push(SampleResponse {
            message: Message::assistant(text),
            usage: Usage::new(1, 1),
            stop_reason: Some("stop".into()),
        });
    }

    /// Map an exact user prompt to an assistant reply (safe under concurrent spawns).
    pub fn map_user_text(&self, user_text: impl Into<String>, reply: impl Into<String>) {
        self.by_user_text
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(user_text.into(), reply.into());
    }

    /// Convenience: assistant tool calls (FIFO).
    pub fn push_tools(&self, message: Message) {
        self.push(SampleResponse {
            message,
            usage: Usage::new(1, 1),
            stop_reason: Some("tool_calls".into()),
        });
    }

    fn last_user_text(request: &SampleRequest) -> Option<String> {
        request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(Message::text)
    }
}

#[async_trait]
impl LlmSampler for MockSampler {
    async fn sample(&self, request: SampleRequest) -> Result<SampleResponse, OvoError> {
        if request.cancel.is_cancelled() {
            return Err(OvoError::new(ErrorCode::LlmCancelled, "sample cancelled"));
        }
        if request.deadline.is_some_and(|d| d.is_expired()) {
            return Err(OvoError::new(
                ErrorCode::LlmCancelled,
                "sample deadline expired",
            ));
        }

        if let Some(user) = Self::last_user_text(&request) {
            let mapped = self
                .by_user_text
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&user);
            if let Some(text) = mapped {
                return Ok(SampleResponse {
                    message: Message::assistant(text),
                    usage: Usage::new(1, 1),
                    stop_reason: Some("stop".into()),
                });
            }
        }

        let mut guard = self
            .responses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if guard.is_empty() {
            return Err(OvoError::new(
                ErrorCode::LlmInvalidResponse,
                "mock sampler has no scripted responses left",
            ));
        }
        guard.remove(0)
    }
}

#[cfg(test)]
mod tests {
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::sample::ToolChoice;

    #[tokio::test]
    async fn pops_in_order() {
        let mock = MockSampler::new();
        mock.push_text("a");
        mock.push_text("b");
        let req = SampleRequest {
            model: "mock".into(),
            messages: vec![Message::user("hi")],
            tools: vec![],
            tool_choice: ToolChoice::default(),
            response_format: None,
            max_output_tokens: None,
            temperature: None,
            cancel: CancellationToken::new(),
            deadline: None,
        };
        let a = mock.sample(req.clone()).await.expect("a");
        let b = mock.sample(req).await.expect("b");
        assert_eq!(a.message.text(), "a");
        assert_eq!(b.message.text(), "b");
    }
}
