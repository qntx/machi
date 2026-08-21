//! [`BreakerSampler`]: decorator that gates samples through a [`CircuitBreaker`].

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures::Stream;
use ovo_types::{ErrorCode, OvoError};

use crate::breaker::{Admission, BreakerOutcome, CircuitBreaker};
use crate::sample::{SampleRequest, SampleResponse};
use crate::sampler::LlmSampler;
use crate::stream::{SampleEvent, SampleStream};

/// Sampler wrapper that refuses traffic while the breaker is open.
#[derive(Debug, Clone)]
pub struct BreakerSampler<S> {
    inner: Arc<S>,
    breaker: Arc<CircuitBreaker>,
    /// Stable key for multi-endpoint registries (metrics / logs).
    endpoint: String,
}

impl<S> BreakerSampler<S> {
    /// Wrap `inner` with a shared breaker.
    #[must_use]
    pub fn new(inner: Arc<S>, breaker: Arc<CircuitBreaker>, endpoint: impl Into<String>) -> Self {
        Self {
            inner,
            breaker,
            endpoint: endpoint.into(),
        }
    }

    /// Endpoint label.
    #[must_use]
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Shared breaker.
    #[must_use]
    pub fn breaker(&self) -> &Arc<CircuitBreaker> {
        &self.breaker
    }
}

#[async_trait]
impl<S: LlmSampler + 'static> LlmSampler for BreakerSampler<S> {
    async fn sample(&self, request: SampleRequest) -> Result<SampleResponse, OvoError> {
        self.admit()?;
        match self.inner.sample(request).await {
            Ok(r) => {
                self.breaker.record(BreakerOutcome::Success);
                Ok(r)
            }
            Err(e) => {
                self.breaker.record(BreakerOutcome::Failure);
                Err(e)
            }
        }
    }

    async fn sample_stream(&self, request: SampleRequest) -> Result<SampleStream, OvoError> {
        self.admit()?;
        match self.inner.sample_stream(request).await {
            Ok(inner) => {
                // Record terminal outcome from stream events — not on open.
                Ok(Box::pin(BreakerStream {
                    inner,
                    breaker: Arc::clone(&self.breaker),
                    recorded: AtomicBool::new(false),
                }))
            }
            Err(e) => {
                self.breaker.record(BreakerOutcome::Failure);
                Err(e)
            }
        }
    }
}

impl<S> BreakerSampler<S> {
    fn admit(&self) -> Result<(), OvoError> {
        match self.breaker.check() {
            Admission::Allow => Ok(()),
            Admission::Reject { retry_after } => Err(OvoError::new(
                ErrorCode::LlmProvider,
                format!(
                    "circuit breaker open for endpoint '{}'; retry after {}ms",
                    self.endpoint,
                    retry_after.as_millis()
                ),
            )
            .with_retry(ovo_types::RetryClass::Backoff)),
        }
    }
}

/// Records breaker outcome from stream terminal events (not on open).
///
/// `SampleStream` is `Pin<Box<dyn Stream>>` (always `Unpin`), so field projection
/// needs no pin projection crate.
struct BreakerStream {
    inner: SampleStream,
    breaker: Arc<CircuitBreaker>,
    recorded: AtomicBool,
}

impl Stream for BreakerStream {
    type Item = SampleEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(ev)) => {
                let outcome = match &ev {
                    SampleEvent::Completed { .. } => Some(BreakerOutcome::Success),
                    SampleEvent::Failed { .. } => Some(BreakerOutcome::Failure),
                    _ => None,
                };
                if let Some(outcome) = outcome
                    && !self.recorded.swap(true, Ordering::Relaxed)
                {
                    self.breaker.record(outcome);
                }
                Poll::Ready(Some(ev))
            }
            Poll::Ready(None) => {
                // Abrupt end without Completed/Failed: treat as failure for half-open honesty.
                if !self.recorded.swap(true, Ordering::Relaxed) {
                    self.breaker.record(BreakerOutcome::Failure);
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
