//! Turn lifecycle contributor port.

use ovo_types::{ErrorCode, OvoError, RunId};

/// Why a turn aborted without a normal completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnAbortReason {
    /// Cancel token fired.
    Cancelled,
    /// Deadline exceeded.
    Deadline,
    /// Max steps exceeded.
    MaxSteps,
    /// Stationarity hard stop.
    Stationarity,
    /// Other typed error.
    Error {
        /// Error code.
        code: ErrorCode,
        /// Message.
        message: String,
    },
}

impl TurnAbortReason {
    /// Build from a [`OvoError`].
    #[must_use]
    pub fn from_error(err: &OvoError) -> Self {
        match err.code() {
            ErrorCode::RuntimeCancelled | ErrorCode::LlmCancelled | ErrorCode::HostCancelled => {
                Self::Cancelled
            }
            ErrorCode::RuntimeDeadline => Self::Deadline,
            ErrorCode::RuntimeMaxSteps => Self::MaxSteps,
            ErrorCode::RuntimeStationarity => Self::Stationarity,
            code => Self::Error {
                code,
                message: err.message().to_owned(),
            },
        }
    }
}

/// Lifecycle hooks for a single turn. Default methods are no-ops.
pub trait TurnLifecycleContributor: Send + Sync {
    /// Called when a turn begins (after run id is assigned).
    fn on_turn_start(&self, _run_id: &RunId) {}
    /// Called when a turn completes successfully.
    fn on_turn_done(&self, _run_id: &RunId, _steps: usize) {}
    /// Called when a turn aborts (cancel / deadline / max steps / stationarity / …).
    fn on_turn_abort(&self, _run_id: &RunId, _reason: &TurnAbortReason) {}
    /// Called when a turn ends with a returned error (non-cancelled path).
    fn on_turn_error(&self, _run_id: &RunId, _err: &OvoError) {}
}

/// No-op contributor.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopLifecycle;

impl TurnLifecycleContributor for NoopLifecycle {}

/// Fan-out to a list of contributors.
#[derive(Default)]
pub struct LifecycleFanout {
    contributors: Vec<std::sync::Arc<dyn TurnLifecycleContributor>>,
}

impl std::fmt::Debug for LifecycleFanout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LifecycleFanout")
            .field("contributors", &self.contributors.len())
            .finish()
    }
}

impl LifecycleFanout {
    /// Empty fanout.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a contributor.
    #[must_use]
    pub fn push(mut self, c: std::sync::Arc<dyn TurnLifecycleContributor>) -> Self {
        self.contributors.push(c);
        self
    }
}

impl TurnLifecycleContributor for LifecycleFanout {
    fn on_turn_start(&self, run_id: &RunId) {
        for c in &self.contributors {
            c.on_turn_start(run_id);
        }
    }

    fn on_turn_done(&self, run_id: &RunId, steps: usize) {
        for c in &self.contributors {
            c.on_turn_done(run_id, steps);
        }
    }

    fn on_turn_abort(&self, run_id: &RunId, reason: &TurnAbortReason) {
        for c in &self.contributors {
            c.on_turn_abort(run_id, reason);
        }
    }

    fn on_turn_error(&self, run_id: &RunId, err: &OvoError) {
        for c in &self.contributors {
            c.on_turn_error(run_id, err);
        }
    }
}
