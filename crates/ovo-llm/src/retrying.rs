//! [`RetryingSampler`]: decorator that applies [`crate::retry::RetryPolicy`].

#![allow(
    clippy::duration_suboptimal_units,
    reason = "DEFAULT_IDLE_TIMEOUT uses secs for portability"
)]

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use ovo_types::{ErrorCode, OvoError};
use tokio::time::{sleep, timeout};
use tokio_util::sync::CancellationToken;

use crate::retry::{RetryContext, RetryDecision, RetryPolicy, decide_retry, is_empty_response};
use crate::sample::{SampleRequest, SampleResponse};
use crate::sampler::LlmSampler;
use crate::stream::{SampleEvent, SampleStream};

/// Default per-chunk idle timeout for streams.
pub const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(5 * 60);

/// Wraps an [`LlmSampler`] with retry / backoff / empty-response handling.
///
/// Providers stay transport-only; this decorator owns the resilience policy.
#[derive(Debug, Clone)]
pub struct RetryingSampler<S> {
    inner: Arc<S>,
    policy: RetryPolicy,
    idle_timeout: Duration,
}

impl<S> RetryingSampler<S> {
    /// Wrap `inner` with the default policy and 300s idle timeout.
    #[must_use]
    pub fn new(inner: Arc<S>) -> Self {
        Self {
            inner,
            policy: RetryPolicy::default(),
            idle_timeout: DEFAULT_IDLE_TIMEOUT,
        }
    }

    /// Custom policy.
    #[must_use]
    pub fn with_policy(mut self, policy: RetryPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Per-chunk idle timeout for streaming samples.
    #[must_use]
    pub const fn with_idle_timeout(mut self, idle: Duration) -> Self {
        self.idle_timeout = idle;
        self
    }

    /// Borrow the policy.
    #[must_use]
    pub const fn policy(&self) -> &RetryPolicy {
        &self.policy
    }
}

#[async_trait]
impl<S: LlmSampler + 'static> LlmSampler for RetryingSampler<S> {
    async fn sample(&self, request: SampleRequest) -> Result<SampleResponse, OvoError> {
        let mut attempt = 0u32;
        let mut rate_limit_retries = 0u32;
        loop {
            if request.cancel.is_cancelled() {
                return Err(OvoError::llm_cancelled("sample cancelled"));
            }
            match self.inner.sample(request.clone()).await {
                Ok(response) => {
                    if is_empty_response(&response.message) {
                        let err =
                            OvoError::new(ErrorCode::LlmEmptyResponse, "empty model response");
                        match decide_retry(
                            &self.policy,
                            &err,
                            &retry_ctx(attempt, rate_limit_retries, &err),
                        ) {
                            RetryDecision::Fatal => return Err(err),
                            RetryDecision::Retry { backoff, .. } => {
                                attempt = attempt.saturating_add(1);
                                sleep_cancellable(backoff, &request.cancel).await?;
                                continue;
                            }
                        }
                    }
                    return Ok(response);
                }
                Err(err) => {
                    let is_rate = matches!(err.code(), ErrorCode::LlmRateLimit);
                    match decide_retry(
                        &self.policy,
                        &err,
                        &retry_ctx(attempt, rate_limit_retries, &err),
                    ) {
                        RetryDecision::Fatal => return Err(err),
                        RetryDecision::Retry { backoff, .. } => {
                            if is_rate {
                                rate_limit_retries = rate_limit_retries.saturating_add(1);
                            }
                            attempt = attempt.saturating_add(1);
                            sleep_cancellable(backoff, &request.cancel).await?;
                        }
                    }
                }
            }
        }
    }

    async fn sample_stream(&self, request: SampleRequest) -> Result<SampleStream, OvoError> {
        let mut attempt = 0u32;
        let mut rate_limit_retries = 0u32;
        loop {
            if request.cancel.is_cancelled() {
                return Err(OvoError::llm_cancelled("sample stream cancelled"));
            }
            match self.inner.sample_stream(request.clone()).await {
                Ok(stream) => {
                    return Ok(idle_timeout_stream(
                        stream,
                        self.idle_timeout,
                        request.cancel.clone(),
                    ));
                }
                Err(err) => {
                    let is_rate = matches!(err.code(), ErrorCode::LlmRateLimit);
                    match decide_retry(
                        &self.policy,
                        &err,
                        &retry_ctx(attempt, rate_limit_retries, &err),
                    ) {
                        RetryDecision::Fatal => return Err(err),
                        RetryDecision::Retry { backoff, .. } => {
                            if is_rate {
                                rate_limit_retries = rate_limit_retries.saturating_add(1);
                            }
                            attempt = attempt.saturating_add(1);
                            sleep_cancellable(backoff, &request.cancel).await?;
                        }
                    }
                }
            }
        }
    }
}

fn retry_ctx(attempt: u32, rate_limit_retries: u32, err: &OvoError) -> RetryContext {
    RetryContext {
        attempt,
        rate_limit_retries,
        retry_after: err.retry_after(),
        x_should_retry: err.x_should_retry(),
        http_status: err.http_status(),
    }
}

async fn sleep_cancellable(dur: Duration, cancel: &CancellationToken) -> Result<(), OvoError> {
    if dur.is_zero() {
        return Ok(());
    }
    tokio::select! {
        () = sleep(dur) => Ok(()),
        () = cancel.cancelled() => Err(OvoError::llm_cancelled("retry sleep cancelled")),
    }
}

struct IdleState {
    stream: Option<SampleStream>,
    idle: Duration,
    cancel: CancellationToken,
}

/// Fail if no event arrives within `idle` between chunks.
fn idle_timeout_stream(
    stream: SampleStream,
    idle: Duration,
    cancel: CancellationToken,
) -> SampleStream {
    let state = IdleState {
        stream: Some(stream),
        idle,
        cancel,
    };
    Box::pin(futures::stream::unfold(state, |mut st| async move {
        let mut stream = st.stream.take()?;
        if st.cancel.is_cancelled() {
            return Some((
                SampleEvent::Failed {
                    message: "sample stream cancelled".into(),
                },
                IdleState {
                    stream: None,
                    idle: st.idle,
                    cancel: st.cancel,
                },
            ));
        }
        match timeout(st.idle, stream.next()).await {
            Ok(Some(ev)) => {
                let terminal = matches!(
                    ev,
                    SampleEvent::Completed { .. } | SampleEvent::Failed { .. }
                );
                let next_stream = if terminal { None } else { Some(stream) };
                Some((
                    ev,
                    IdleState {
                        stream: next_stream,
                        idle: st.idle,
                        cancel: st.cancel,
                    },
                ))
            }
            Ok(None) => None,
            Err(_) => Some((
                SampleEvent::Failed {
                    message: format!(
                        "idle timeout after {}s between stream chunks",
                        st.idle.as_secs()
                    ),
                },
                IdleState {
                    stream: None,
                    idle: st.idle,
                    cancel: st.cancel,
                },
            )),
        }
    }))
}

#[cfg(test)]
#[allow(clippy::expect_used, reason = "unit tests")]
mod tests {
    use ovo_types::Message;

    use super::*;
    use crate::mock::MockSampler;
    use crate::retry::RetryPolicy;
    use crate::sample::ToolChoice;

    fn req(text: &str) -> SampleRequest {
        SampleRequest {
            model: "mock".into(),
            messages: vec![Message::user(text)],
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            response_format: None,
            max_output_tokens: None,
            temperature: None,
            cancel: CancellationToken::new(),
            deadline: None,
        }
    }

    #[tokio::test]
    async fn retries_then_succeeds() {
        let mock = Arc::new(MockSampler::new());
        // Fail once then succeed — MockSampler uses FIFO for unmapped prompts.
        mock.push_error(OvoError::new(ErrorCode::LlmProvider, "transient"));
        mock.push_text("ok");
        let sampler = RetryingSampler::new(mock).with_policy(RetryPolicy::for_tests());
        let out = sampler.sample(req("hi")).await.expect("sample");
        assert_eq!(out.message.text(), "ok");
    }

    #[tokio::test]
    async fn empty_response_retries() {
        let mock = Arc::new(MockSampler::new());
        mock.push_text("");
        mock.push_text("recovered");
        let sampler = RetryingSampler::new(mock).with_policy(RetryPolicy::for_tests());
        let out = sampler.sample(req("e")).await.expect("sample");
        assert_eq!(out.message.text(), "recovered");
    }
}
