//! Absolute deadlines for cooperative cancellation.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// An absolute deadline for a unit of work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Deadline {
    instant: Instant,
}

impl Deadline {
    /// Deadline after `duration` from now.
    #[must_use]
    pub fn after(duration: Duration) -> Self {
        Self {
            instant: Instant::now() + duration,
        }
    }

    /// Deadline at a specific instant.
    #[must_use]
    pub const fn at(instant: Instant) -> Self {
        Self { instant }
    }

    /// Returns true when the deadline has passed.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        Instant::now() >= self.instant
    }

    /// Remaining time, or zero if expired.
    #[must_use]
    pub fn remaining(&self) -> Duration {
        self.instant.saturating_duration_since(Instant::now())
    }

    /// Underlying instant.
    #[must_use]
    pub const fn instant(&self) -> Instant {
        self.instant
    }
}

/// Serializable duration budget (not wall-clock) for configs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DurationBudget {
    /// Milliseconds.
    pub millis: u64,
}

impl DurationBudget {
    /// Construct from milliseconds.
    #[must_use]
    pub const fn from_millis(millis: u64) -> Self {
        Self { millis }
    }

    /// Convert to [`Duration`].
    #[must_use]
    pub const fn as_duration(self) -> Duration {
        Duration::from_millis(self.millis)
    }

    /// Convert to a live [`Deadline`].
    #[must_use]
    pub fn to_deadline(self) -> Deadline {
        Deadline::after(self.as_duration())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expired_zero() {
        let d = Deadline::after(Duration::ZERO);
        assert!(d.remaining().is_zero() || d.is_expired());
    }
}
