//! Callback system for agent events.
//!
//! This module provides a callback system that allows users to hook into
//! various agent events like step completion, tool calls, and errors.

use crate::memory::{ActionStep, PlanningStep, TaskStep};
use std::fmt;
use std::sync::Arc;

/// Type alias for a boxed callback function.
pub type BoxedCallback = Box<dyn Fn(&StepEvent) + Send + Sync>;

/// Events that can be emitted during agent execution.
#[derive(Debug)]
pub enum StepEvent {
    /// A new task has started.
    TaskStarted(TaskStep),
    /// A planning step has completed.
    PlanningComplete(PlanningStep),
    /// An action step is starting.
    ActionStarting {
        /// Step number.
        step_number: usize,
    },
    /// An action step has completed.
    ActionComplete(ActionStep),
    /// An error occurred during a step.
    StepError {
        /// Step number.
        step_number: usize,
        /// Error message.
        error: String,
    },
    /// Agent execution has completed.
    AgentComplete {
        /// Whether the run was successful.
        success: bool,
        /// Total steps taken.
        total_steps: usize,
    },
}

impl fmt::Display for StepEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TaskStarted(task) => write!(f, "Task started: {}", task.task),
            Self::PlanningComplete(plan) => write!(f, "Planning complete: {}", plan.plan),
            Self::ActionStarting { step_number } => write!(f, "Starting step {}", step_number),
            Self::ActionComplete(step) => {
                write!(f, "Step {} complete", step.step_number)?;
                if let Some(obs) = &step.observations {
                    write!(f, " - Observations: {}", obs)?;
                }
                Ok(())
            }
            Self::StepError { step_number, error } => {
                write!(f, "Step {} error: {}", step_number, error)
            }
            Self::AgentComplete {
                success,
                total_steps,
            } => write!(
                f,
                "Agent {} after {} steps",
                if *success { "completed" } else { "failed" },
                total_steps
            ),
        }
    }
}

/// A collection of callbacks for agent events.
#[derive(Default)]
pub struct CallbackManager {
    /// Callbacks to invoke on each event.
    callbacks: Vec<Arc<BoxedCallback>>,
}

impl CallbackManager {
    /// Create a new empty callback manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a callback.
    pub fn add<F>(&mut self, callback: F)
    where
        F: Fn(&StepEvent) + Send + Sync + 'static,
    {
        self.callbacks.push(Arc::new(Box::new(callback)));
    }

    /// Emit an event to all callbacks.
    pub fn emit(&self, event: &StepEvent) {
        for callback in &self.callbacks {
            callback(event);
        }
    }

    /// Check if there are any callbacks registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.callbacks.is_empty()
    }

    /// Get the number of registered callbacks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.callbacks.len()
    }
}

impl fmt::Debug for CallbackManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CallbackManager")
            .field("callback_count", &self.callbacks.len())
            .finish()
    }
}

/// A simple logging callback that prints events to stdout.
pub fn logging_callback(event: &StepEvent) {
    println!("[Agent] {}", event);
}

/// Create a callback that logs events with a custom prefix.
pub fn prefixed_logging_callback(prefix: String) -> impl Fn(&StepEvent) + Send + Sync {
    move |event| {
        println!("[{}] {}", prefix, event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_callback_manager() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let mut manager = CallbackManager::new();
        manager.add(move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        assert_eq!(manager.len(), 1);
        assert!(!manager.is_empty());

        manager.emit(&StepEvent::ActionStarting { step_number: 1 });
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        manager.emit(&StepEvent::ActionStarting { step_number: 2 });
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }
}
