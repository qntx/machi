//! Retry classification and backoff for sampler decorators.
//!
//!
//! Pure logic — no I/O. [`crate::retrying::RetryingSampler`] applies the policy.

#![allow(
    clippy::duration_suboptimal_units,
    reason = "policy constants use secs for stable readability across toolchains"
)]

use std::time::Duration;

use ovo_types::{ErrorCode, OvoError, RetryClass};

/// After this many 429 retries, escalate instead of waiting again.
pub const RATE_LIMIT_RETRY_THRESHOLD: u32 = 2;

/// Default max attempts (including the first try). Attempt index reaching this is fatal.
pub const DEFAULT_MAX_ATTEMPTS: u32 = 15;

/// Cap for exponential backoff (and non-429 Retry-After clamp).
pub const MAX_RETRY_BACKOFF: Duration = Duration::from_secs(30);

/// Cap when honoring full `Retry-After` on 429.
pub const MAX_RETRY_AFTER: Duration = Duration::from_secs(120);

/// Base exponential backoff start (attempt 1 → 2s).
const BACKOFF_BASE_SECS: u64 = 2;

/// Configurable retry policy for [`crate::retrying::RetryingSampler`].
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    /// Max sample attempts (including first). Default [`DEFAULT_MAX_ATTEMPTS`].
    pub max_attempts: u32,
    /// Max 429 retries (separate budget). Default [`RATE_LIMIT_RETRY_THRESHOLD`].
    pub rate_limit_max: u32,
    /// Cap for exponential backoff.
    pub max_backoff: Duration,
    /// Cap for full Retry-After waits on 429.
    pub max_retry_after: Duration,
    /// Disable jitter (tests).
    pub jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            rate_limit_max: RATE_LIMIT_RETRY_THRESHOLD,
            max_backoff: MAX_RETRY_BACKOFF,
            max_retry_after: MAX_RETRY_AFTER,
            jitter: true,
        }
    }
}

impl RetryPolicy {
    /// Test-friendly policy: few attempts, no jitter, short caps.
    #[must_use]
    pub fn for_tests() -> Self {
        Self {
            max_attempts: 5,
            rate_limit_max: 2,
            max_backoff: Duration::ZERO,
            max_retry_after: Duration::ZERO,
            jitter: false,
        }
    }
}

/// Decision from the classifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetryDecision {
    /// Do not retry; surface the error.
    Fatal,
    /// Sleep then retry.
    Retry {
        /// Backoff before next attempt.
        backoff: Duration,
        /// Human reason for telemetry / [`crate::stream::SampleEvent::Retrying`].
        reason: String,
    },
}

/// Context for one failed attempt.
#[derive(Debug, Clone, Copy)]
pub struct RetryContext {
    /// 0-based attempt index of the failure (0 = first try failed).
    pub attempt: u32,
    /// How many 429 retries have already been consumed.
    pub rate_limit_retries: u32,
    /// Optional `Retry-After` from the provider (seconds).
    pub retry_after: Option<Duration>,
    /// Optional `x-should-retry` header (`Some(false)` is fatal).
    pub x_should_retry: Option<bool>,
    /// HTTP status when known.
    pub http_status: Option<u16>,
}

/// Classify an HTTP status for the edge-client policy.
///
/// - `x-should-retry: false` → fatal  
/// - 400/401/403/404/422 → fatal  
/// - 525/526 → fatal  
/// - 429 → rate-limited retry  
/// - other 5xx → retry  
/// - else → fatal (unknown 4xx)
#[must_use]
pub fn classify_http_status(status: u16, x_should_retry: Option<bool>) -> HttpRetryClass {
    if x_should_retry == Some(false) {
        return HttpRetryClass::Fatal;
    }
    match status {
        400 | 401 | 403 | 404 | 422 => HttpRetryClass::Fatal,
        525 | 526 => HttpRetryClass::Fatal,
        429 => HttpRetryClass::RateLimited,
        s if (500..600).contains(&s) => HttpRetryClass::Retry,
        _ if x_should_retry == Some(true) => HttpRetryClass::Retry,
        _ => HttpRetryClass::Fatal,
    }
}

/// HTTP classification outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpRetryClass {
    /// Do not retry.
    Fatal,
    /// Retry with exponential backoff.
    Retry,
    /// Retry with 429 budget + Retry-After.
    RateLimited,
}

/// Map HTTP class + status into a [`OvoError`] code.
#[must_use]
pub fn error_code_for_http(status: u16, class: HttpRetryClass) -> ErrorCode {
    match (status, class) {
        (401 | 403, _) => ErrorCode::LlmAuth,
        (429, _) | (_, HttpRetryClass::RateLimited) => ErrorCode::LlmRateLimit,
        _ => ErrorCode::LlmProvider,
    }
}

/// Decide whether to retry after a [`OvoError`].
#[must_use]
pub fn decide_retry(policy: &RetryPolicy, err: &OvoError, ctx: &RetryContext) -> RetryDecision {
    let next_attempt = ctx.attempt.saturating_add(1);
    if next_attempt >= policy.max_attempts {
        return RetryDecision::Fatal;
    }

    // Explicit server hint on the error path is not always present; honor RetryClass.
    match err.code() {
        ErrorCode::LlmIdleTimeout | ErrorCode::LlmTruncated => {
            return RetryDecision::Fatal;
        }
        // Auth: honor AuthRefresh once (transport may refresh credentials), else fatal.
        ErrorCode::LlmAuth => {
            if err.retry_class() == RetryClass::AuthRefresh && ctx.attempt == 0 {
                return RetryDecision::Retry {
                    backoff: backoff_for_attempt(policy, next_attempt),
                    reason: "auth_refresh".into(),
                };
            }
            return RetryDecision::Fatal;
        }
        ErrorCode::LlmCancelled => return RetryDecision::Fatal,
        ErrorCode::LlmEmptyResponse => {
            let backoff = backoff_for_attempt(policy, next_attempt);
            return RetryDecision::Retry {
                backoff,
                reason: "empty_response".into(),
            };
        }
        ErrorCode::LlmRateLimit => {
            if ctx.rate_limit_retries >= policy.rate_limit_max {
                return RetryDecision::Fatal;
            }
            let wait = ctx
                .retry_after
                .unwrap_or_else(|| backoff_for_attempt(policy, next_attempt))
                .min(policy.max_retry_after);
            return RetryDecision::Retry {
                backoff: wait,
                reason: "rate_limited".into(),
            };
        }
        ErrorCode::LlmProvider if err.retry_class() == RetryClass::Backoff => {
            let backoff = ctx
                .retry_after
                .map(|d| d.min(policy.max_backoff))
                .unwrap_or_else(|| backoff_for_attempt(policy, next_attempt));
            return RetryDecision::Retry {
                backoff,
                reason: "provider".into(),
            };
        }
        _ => {}
    }

    if let Some(status) = ctx.http_status {
        match classify_http_status(status, ctx.x_should_retry) {
            HttpRetryClass::Fatal => return RetryDecision::Fatal,
            HttpRetryClass::RateLimited => {
                if ctx.rate_limit_retries >= policy.rate_limit_max {
                    return RetryDecision::Fatal;
                }
                let wait = ctx
                    .retry_after
                    .unwrap_or_else(|| backoff_for_attempt(policy, next_attempt))
                    .min(policy.max_retry_after);
                return RetryDecision::Retry {
                    backoff: wait,
                    reason: format!("http_{status}"),
                };
            }
            HttpRetryClass::Retry => {
                let backoff = backoff_for_attempt(policy, next_attempt);
                return RetryDecision::Retry {
                    backoff,
                    reason: format!("http_{status}"),
                };
            }
        }
    }

    if err.retry_class() == RetryClass::Backoff {
        let backoff = backoff_for_attempt(policy, next_attempt);
        return RetryDecision::Retry {
            backoff,
            reason: err.code().as_str().into(),
        };
    }

    RetryDecision::Fatal
}

/// Exponential backoff 2^attempt seconds from 2s, capped, with optional ±20% jitter.
#[must_use]
pub fn backoff_for_attempt(policy: &RetryPolicy, attempt: u32) -> Duration {
    // attempt 1 → 2s, 2 → 4s, …
    let shift = attempt.saturating_sub(1).min(16);
    let base_ms = (BACKOFF_BASE_SECS.saturating_mul(1000))
        .checked_shl(shift)
        .unwrap_or(u64::MAX)
        .min(u64::try_from(policy.max_backoff.as_millis()).unwrap_or(u64::MAX));
    let base = Duration::from_millis(base_ms);
    if policy.jitter { jittered(base) } else { base }
}

fn jittered(base: Duration) -> Duration {
    use std::hash::{Hash, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQ: AtomicU64 = AtomicU64::new(0);
    let base_ms = u64::try_from(base.as_millis()).unwrap_or(u64::MAX);
    let range = (base_ms / 5).max(1);
    let mut hasher = std::hash::DefaultHasher::new();
    SEQ.fetch_add(1, Ordering::Relaxed).hash(&mut hasher);
    base_ms.hash(&mut hasher);
    let j = hasher.finish() % (range.saturating_mul(2).saturating_add(1));
    // ±20% without i64: pick offset in [0, 2*range] then subtract range via saturating.
    let ms = if j >= range {
        base_ms.saturating_add(j - range)
    } else {
        base_ms.saturating_sub(range - j)
    };
    Duration::from_millis(ms.max(1))
}

/// True when a completed response is empty (retryable `EmptyResponse`).
#[must_use]
pub fn is_empty_response(message: &ovo_types::Message) -> bool {
    message.tool_calls.is_empty() && message.text().trim().is_empty()
}

#[cfg(test)]
#[allow(clippy::expect_used, reason = "unit tests")]
mod tests {
    use super::*;

    #[test]
    fn fatal_4xx_table() {
        for s in [400_u16, 401, 403, 404, 422] {
            assert_eq!(
                classify_http_status(s, None),
                HttpRetryClass::Fatal,
                "status {s}"
            );
        }
    }

    #[test]
    fn retry_5xx_except_tls() {
        assert_eq!(classify_http_status(500, None), HttpRetryClass::Retry);
        assert_eq!(classify_http_status(503, None), HttpRetryClass::Retry);
        assert_eq!(classify_http_status(525, None), HttpRetryClass::Fatal);
        assert_eq!(classify_http_status(526, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn rate_limit_and_header_hint() {
        assert_eq!(classify_http_status(429, None), HttpRetryClass::RateLimited);
        assert_eq!(
            classify_http_status(500, Some(false)),
            HttpRetryClass::Fatal
        );
        assert_eq!(classify_http_status(418, Some(true)), HttpRetryClass::Retry);
    }

    #[test]
    fn empty_response_is_retried() {
        let policy = RetryPolicy::for_tests();
        let err = OvoError::new(ErrorCode::LlmEmptyResponse, "empty");
        let d = decide_retry(
            &policy,
            &err,
            &RetryContext {
                attempt: 0,
                rate_limit_retries: 0,
                retry_after: None,
                x_should_retry: None,
                http_status: None,
            },
        );
        assert!(matches!(d, RetryDecision::Retry { .. }));
    }

    #[test]
    fn rate_limit_budget_exhausted() {
        let policy = RetryPolicy::for_tests();
        let err = OvoError::new(ErrorCode::LlmRateLimit, "429");
        let d = decide_retry(
            &policy,
            &err,
            &RetryContext {
                attempt: 0,
                rate_limit_retries: policy.rate_limit_max,
                retry_after: Some(Duration::from_secs(5)),
                x_should_retry: None,
                http_status: Some(429),
            },
        );
        assert_eq!(d, RetryDecision::Fatal);
    }

    #[test]
    fn idle_timeout_fatal() {
        let policy = RetryPolicy::default();
        let err = OvoError::new(ErrorCode::LlmIdleTimeout, "idle");
        let d = decide_retry(
            &policy,
            &err,
            &RetryContext {
                attempt: 0,
                rate_limit_retries: 0,
                retry_after: None,
                x_should_retry: None,
                http_status: None,
            },
        );
        assert_eq!(d, RetryDecision::Fatal);
    }

    #[test]
    fn auth_refresh_once_then_fatal() {
        let policy = RetryPolicy::for_tests();
        let err = OvoError::new(ErrorCode::LlmAuth, "401").with_retry(RetryClass::AuthRefresh);
        let first = decide_retry(
            &policy,
            &err,
            &RetryContext {
                attempt: 0,
                rate_limit_retries: 0,
                retry_after: None,
                x_should_retry: None,
                http_status: Some(401),
            },
        );
        assert!(
            matches!(first, RetryDecision::Retry { ref reason, .. } if reason == "auth_refresh"),
            "{first:?}"
        );
        let second = decide_retry(
            &policy,
            &err,
            &RetryContext {
                attempt: 1,
                rate_limit_retries: 0,
                retry_after: None,
                x_should_retry: None,
                http_status: Some(401),
            },
        );
        assert_eq!(second, RetryDecision::Fatal);
    }

    #[test]
    fn rate_limit_honors_retry_after_from_context() {
        let policy = RetryPolicy {
            max_retry_after: Duration::from_secs(60),
            jitter: false,
            ..RetryPolicy::for_tests()
        };
        let err = OvoError::new(ErrorCode::LlmRateLimit, "429")
            .with_http_status(429)
            .with_retry_after(Duration::from_secs(9));
        let d = decide_retry(
            &policy,
            &err,
            &RetryContext {
                attempt: 0,
                rate_limit_retries: 0,
                retry_after: err.retry_after(),
                x_should_retry: None,
                http_status: err.http_status(),
            },
        );
        assert!(
            matches!(
                d,
                RetryDecision::Retry {
                    backoff,
                    ..
                } if backoff == Duration::from_secs(9)
            ),
            "{d:?}"
        );
    }

    #[test]
    fn backoff_first_attempt_near_two_seconds_without_jitter() {
        let policy = RetryPolicy {
            jitter: false,
            ..RetryPolicy::default()
        };
        assert_eq!(backoff_for_attempt(&policy, 1), Duration::from_secs(2));
        assert_eq!(backoff_for_attempt(&policy, 2), Duration::from_secs(4));
        assert_eq!(backoff_for_attempt(&policy, 10), MAX_RETRY_BACKOFF);
    }
}

include!("http_status_matrix.rs");
