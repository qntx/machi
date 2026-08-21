//! [`LlmSampler`] trait.

use async_trait::async_trait;
use futures::stream;
use ovo_types::OvoError;

use crate::sample::{SampleRequest, SampleResponse};
use crate::stream::{SampleEvent, SampleStream};

/// Abstraction over model providers.
#[async_trait]
pub trait LlmSampler: Send + Sync {
    /// Perform one non-streaming sample.
    async fn sample(&self, request: SampleRequest) -> Result<SampleResponse, OvoError>;

    /// Streaming sample. Default: call [`Self::sample`] and emit a single
    /// [`SampleEvent::Completed`] (plus usage when non-zero).
    async fn sample_stream(&self, request: SampleRequest) -> Result<SampleStream, OvoError> {
        let response = self.sample(request).await?;
        Ok(response_to_stream(response))
    }
}

/// Convert a complete response into a short stream.
#[must_use]
pub fn response_to_stream(response: SampleResponse) -> SampleStream {
    let mut events = Vec::with_capacity(3);
    let text = response.message.text();
    if !text.is_empty() && response.message.tool_calls.is_empty() {
        events.push(SampleEvent::TextDelta { text });
    }
    if !response.message.tool_calls.is_empty() {
        events.push(SampleEvent::ToolCalls {
            message: response.message.clone(),
        });
    }
    if response.usage.total_tokens > 0
        || response.usage.input_tokens > 0
        || response.usage.output_tokens > 0
    {
        events.push(SampleEvent::Usage(response.usage));
    }
    events.push(SampleEvent::Completed {
        message: response.message,
        stop_reason: response.stop_reason,
    });
    Box::pin(stream::iter(events))
}
