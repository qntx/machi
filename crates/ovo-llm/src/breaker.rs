//! Windowed circuit breaker for sampler endpoints.

#![allow(
    clippy::duration_suboptimal_units,
    reason = "window/open defaults use secs for stable readability"
)]

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Outcome recorded after a probe or live call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerOutcome {
    /// Successful sample.
    Success,
    /// Failed sample (counts toward error rate).
    Failure,
}

/// Breaker state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerState {
    /// Accepting traffic.
    Closed,
    /// Rejecting traffic until cooldown elapses.
    Open,
    /// Limited probes allowed.
    HalfOpen,
}

/// Configuration for [`CircuitBreaker`].
#[derive(Debug, Clone, Copy)]
pub struct BreakerConfig {
    /// Minimum samples in the window before opening.
    pub min_samples: u32,
    /// Error rate threshold in `(0, 1]` (e.g. `0.5` = 50%).
    pub error_rate_threshold: f64,
    /// Sliding window duration.
    pub window: Duration,
    /// How long to stay open before half-open.
    pub open_duration: Duration,
    /// Max concurrent probes in half-open.
    pub half_open_max_probes: u32,
}

impl Default for BreakerConfig {
    fn default() -> Self {
        Self {
            min_samples: 5,
            error_rate_threshold: 0.5,
            window: Duration::from_secs(60),
            open_duration: Duration::from_secs(30),
            half_open_max_probes: 1,
        }
    }
}

/// Admission decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    /// Call may proceed.
    Allow,
    /// Breaker is open; retry after this delay.
    Reject {
        /// Suggested wait.
        retry_after: Duration,
    },
}

#[derive(Debug)]
struct Inner {
    state: BreakerState,
    events: VecDeque<(Instant, bool)>,
    opened_at: Option<Instant>,
    half_open_inflight: u32,
}

/// Thread-safe windowed circuit breaker.
#[derive(Debug)]
pub struct CircuitBreaker {
    config: BreakerConfig,
    inner: Mutex<Inner>,
}

impl CircuitBreaker {
    /// Create with config.
    #[must_use]
    pub fn new(config: BreakerConfig) -> Self {
        Self {
            config,
            inner: Mutex::new(Inner {
                state: BreakerState::Closed,
                events: VecDeque::new(),
                opened_at: None,
                half_open_inflight: 0,
            }),
        }
    }

    /// Current state (for tests / metrics).
    #[must_use]
    pub fn state(&self) -> BreakerState {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .state
    }

    /// Check whether a call is admitted.
    #[must_use]
    #[allow(
        clippy::significant_drop_tightening,
        reason = "admission mutates breaker state under one lock"
    )]
    pub fn check(&self) -> Admission {
        let mut g = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let now = Instant::now();
        self.prune(&mut g, now);
        match g.state {
            BreakerState::Closed => Admission::Allow,
            BreakerState::Open => self.check_open(&mut g, now),
            BreakerState::HalfOpen => self.check_half_open(&mut g),
        }
    }

    fn check_open(&self, g: &mut Inner, now: Instant) -> Admission {
        let cooled = g
            .opened_at
            .is_some_and(|opened| now.duration_since(opened) >= self.config.open_duration);
        if cooled {
            g.state = BreakerState::HalfOpen;
            g.half_open_inflight = 1;
            return Admission::Allow;
        }
        let remaining = g
            .opened_at
            .map(|t| {
                self.config
                    .open_duration
                    .saturating_sub(now.duration_since(t))
            })
            .unwrap_or(self.config.open_duration);
        Admission::Reject {
            retry_after: remaining,
        }
    }

    fn check_half_open(&self, g: &mut Inner) -> Admission {
        if g.half_open_inflight < self.config.half_open_max_probes {
            g.half_open_inflight = g.half_open_inflight.saturating_add(1);
            Admission::Allow
        } else {
            Admission::Reject {
                retry_after: Duration::from_secs(1),
            }
        }
    }

    /// Record an outcome after a call.
    #[allow(
        clippy::significant_drop_tightening,
        reason = "state transition needs the mutex for the whole update"
    )]
    pub fn record(&self, outcome: BreakerOutcome) {
        let now = Instant::now();
        let success = matches!(outcome, BreakerOutcome::Success);
        let mut g = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.prune(&mut g, now);
        g.events.push_back((now, success));
        self.apply_outcome(&mut g, now, success);
    }

    fn apply_outcome(&self, g: &mut Inner, now: Instant, success: bool) {
        match g.state {
            BreakerState::HalfOpen => {
                g.half_open_inflight = g.half_open_inflight.saturating_sub(1);
                if success {
                    g.state = BreakerState::Closed;
                    g.opened_at = None;
                    g.events.clear();
                } else {
                    g.state = BreakerState::Open;
                    g.opened_at = Some(now);
                }
            }
            BreakerState::Closed => {
                if self.should_open(g) {
                    g.state = BreakerState::Open;
                    g.opened_at = Some(now);
                }
            }
            BreakerState::Open => {}
        }
    }

    fn prune(&self, g: &mut Inner, now: Instant) {
        let window = self.config.window;
        while g
            .events
            .front()
            .is_some_and(|(t, _)| now.duration_since(*t) > window)
        {
            g.events.pop_front();
        }
    }

    fn should_open(&self, g: &Inner) -> bool {
        let n = g.events.len();
        if n < self.config.min_samples as usize {
            return false;
        }
        let failures = g.events.iter().filter(|(_, ok)| !*ok).count();
        let rate = failures as f64 / n as f64;
        rate >= self.config.error_rate_threshold
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, reason = "unit tests")]
mod tests {
    use super::*;

    #[test]
    fn opens_after_error_rate() {
        let cb = CircuitBreaker::new(BreakerConfig {
            min_samples: 4,
            error_rate_threshold: 0.5,
            window: Duration::from_secs(60),
            open_duration: Duration::from_secs(30),
            half_open_max_probes: 1,
        });
        assert_eq!(cb.check(), Admission::Allow);
        for _ in 0..2 {
            cb.record(BreakerOutcome::Success);
        }
        for _ in 0..2 {
            cb.record(BreakerOutcome::Failure);
        }
        assert_eq!(cb.state(), BreakerState::Open);
        assert!(matches!(cb.check(), Admission::Reject { .. }));
    }

    #[tokio::test]
    async fn half_open_success_closes() {
        let cb = CircuitBreaker::new(BreakerConfig {
            min_samples: 2,
            error_rate_threshold: 0.5,
            window: Duration::from_secs(60),
            open_duration: Duration::from_secs(0),
            half_open_max_probes: 1,
        });
        cb.record(BreakerOutcome::Failure);
        cb.record(BreakerOutcome::Failure);
        assert_eq!(cb.state(), BreakerState::Open);
        // open_duration is zero → next check immediately enters half-open.
        assert_eq!(cb.check(), Admission::Allow);
        assert_eq!(cb.state(), BreakerState::HalfOpen);
        cb.record(BreakerOutcome::Success);
        assert_eq!(cb.state(), BreakerState::Closed);
    }
}
