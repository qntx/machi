//! Built-in callback handlers for common use cases.

use super::context::CallbackContext;
use crate::memory::{ActionStep, FinalAnswerStep, MemoryStep, PlanningStep, TaskStep};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use tracing::{debug, info, warn};

/// Configuration for the logging handler.
#[derive(Debug, Clone)]
pub struct LoggingConfig {
    /// Whether to log task steps.
    pub log_tasks: bool,
    /// Whether to log action steps.
    pub log_actions: bool,
    /// Whether to log planning steps.
    pub log_planning: bool,
    /// Whether to log final answers.
    pub log_final_answers: bool,
    /// Whether to include detailed information.
    pub detailed: bool,
    /// Custom prefix for log messages.
    pub prefix: Option<String>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            log_tasks: true,
            log_actions: true,
            log_planning: true,
            log_final_answers: true,
            detailed: false,
            prefix: None,
        }
    }
}

impl LoggingConfig {
    /// Create a minimal logging config (only errors and final answers).
    #[must_use]
    pub const fn minimal() -> Self {
        Self {
            log_tasks: false,
            log_actions: false,
            log_planning: false,
            log_final_answers: true,
            detailed: false,
            prefix: None,
        }
    }

    /// Create a verbose logging config.
    #[must_use]
    pub const fn verbose() -> Self {
        Self {
            log_tasks: true,
            log_actions: true,
            log_planning: true,
            log_final_answers: true,
            detailed: true,
            prefix: None,
        }
    }

    /// Set a custom prefix.
    #[must_use]
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }
}

/// Create a logging handler with the given configuration.
///
/// Uses the `tracing` crate for structured logging.
pub fn logging_handler(
    config: LoggingConfig,
) -> impl Fn(&dyn MemoryStep, &CallbackContext) + Send + Sync + 'static {
    move |step, ctx| {
        let prefix = config.prefix.as_deref().unwrap_or("Agent");

        if let Some(task) = step.as_any().downcast_ref::<TaskStep>() {
            if config.log_tasks {
                info!(
                    target: "machi::callback",
                    prefix = prefix,
                    task = %task.task.chars().take(100).collect::<String>(),
                    has_images = task.has_images(),
                    "[{prefix}] New task started"
                );
            }
        } else if let Some(action) = step.as_any().downcast_ref::<ActionStep>() {
            if config.log_actions {
                let tool_names: Vec<_> = action
                    .tool_calls
                    .as_ref()
                    .map(|calls| calls.iter().map(|c| c.name.as_str()).collect())
                    .unwrap_or_default();

                if config.detailed {
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "duration is always non-negative and within u64 range"
                    )]
                    let duration_ms = action.timing.duration_secs().map(|d| (d * 1000.0) as u64);
                    info!(
                        target: "machi::callback",
                        prefix = prefix,
                        step = action.step_number,
                        tools = ?tool_names,
                        has_error = action.error.is_some(),
                        is_final = action.is_final_answer,
                        duration_ms = duration_ms,
                        "[{prefix}] Step {step} completed",
                        step = action.step_number
                    );
                } else {
                    debug!(
                        target: "machi::callback",
                        prefix = prefix,
                        step = action.step_number,
                        "[{prefix}] Step {step} completed",
                        step = action.step_number
                    );
                }

                if action.error.is_some() {
                    warn!(
                        target: "machi::callback",
                        prefix = prefix,
                        step = action.step_number,
                        error = ?action.error,
                        "[{prefix}] Step {step} encountered error",
                        step = action.step_number
                    );
                }
            }
        } else if let Some(planning) = step.as_any().downcast_ref::<PlanningStep>() {
            if config.log_planning {
                debug!(
                    target: "machi::callback",
                    prefix = prefix,
                    plan_length = planning.plan.len(),
                    "[{prefix}] Planning step completed"
                );
            }
        } else if let Some(final_answer) = step.as_any().downcast_ref::<FinalAnswerStep>()
            && config.log_final_answers
        {
            info!(
                target: "machi::callback",
                prefix = prefix,
                step = ctx.step_number,
                output_type = std::any::type_name_of_val(&final_answer.output),
                "[{prefix}] Final answer provided"
            );
        }
    }
}

/// Metrics collector for tracking agent execution statistics.
#[derive(Debug, Default)]
pub struct MetricsCollector {
    /// Total steps executed.
    pub steps: AtomicUsize,
    /// Total tool calls made.
    pub tool_calls: AtomicUsize,
    /// Total errors encountered.
    pub errors: AtomicUsize,
    /// Total input tokens.
    pub input_tokens: AtomicU64,
    /// Total output tokens.
    pub output_tokens: AtomicU64,
    /// Tasks started.
    pub tasks: AtomicUsize,
    /// Final answers provided.
    pub final_answers: AtomicUsize,
}

impl MetricsCollector {
    /// Create a new metrics collector.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all metrics.
    pub fn reset(&self) {
        self.steps.store(0, Ordering::SeqCst);
        self.tool_calls.store(0, Ordering::SeqCst);
        self.errors.store(0, Ordering::SeqCst);
        self.input_tokens.store(0, Ordering::SeqCst);
        self.output_tokens.store(0, Ordering::SeqCst);
        self.tasks.store(0, Ordering::SeqCst);
        self.final_answers.store(0, Ordering::SeqCst);
    }

    /// Get a snapshot of current metrics.
    #[must_use]
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            steps: self.steps.load(Ordering::SeqCst),
            tool_calls: self.tool_calls.load(Ordering::SeqCst),
            errors: self.errors.load(Ordering::SeqCst),
            input_tokens: self.input_tokens.load(Ordering::SeqCst),
            output_tokens: self.output_tokens.load(Ordering::SeqCst),
            tasks: self.tasks.load(Ordering::SeqCst),
            final_answers: self.final_answers.load(Ordering::SeqCst),
        }
    }
}

/// Snapshot of metrics at a point in time.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct MetricsSnapshot {
    /// Total steps executed.
    pub steps: usize,
    /// Total tool calls made.
    pub tool_calls: usize,
    /// Total errors encountered.
    pub errors: usize,
    /// Total input tokens.
    pub input_tokens: u64,
    /// Total output tokens.
    pub output_tokens: u64,
    /// Tasks started.
    pub tasks: usize,
    /// Final answers provided.
    pub final_answers: usize,
}

impl MetricsSnapshot {
    /// Total tokens (input + output).
    #[must_use]
    pub const fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    /// Convert to `RunMetrics` with duration.
    #[must_use]
    pub fn to_run_metrics(self, duration: std::time::Duration) -> RunMetrics {
        RunMetrics {
            steps: self.steps,
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            duration: Some(duration),
            tool_calls: self.tool_calls,
            errors: self.errors,
        }
    }
}

impl std::fmt::Display for MetricsSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Metrics:")?;
        writeln!(f, "  Steps:      {}", self.steps)?;
        writeln!(
            f,
            "  Tokens:     {} (in: {}, out: {})",
            self.total_tokens(),
            self.input_tokens,
            self.output_tokens
        )?;
        writeln!(f, "  Tool calls: {}", self.tool_calls)?;
        writeln!(f, "  Errors:     {}", self.errors)?;
        Ok(())
    }
}

/// Metrics collected during an agent run, including duration.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct RunMetrics {
    /// Total steps executed.
    pub steps: usize,
    /// Total input tokens.
    pub input_tokens: u64,
    /// Total output tokens.
    pub output_tokens: u64,
    /// Total duration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<std::time::Duration>,
    /// Tool calls made.
    pub tool_calls: usize,
    /// Errors encountered.
    pub errors: usize,
}

impl RunMetrics {
    /// Total tokens (input + output).
    #[must_use]
    pub const fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    /// Tokens per second rate.
    #[must_use]
    pub fn tokens_per_second(&self) -> Option<f64> {
        self.duration.map(|d| {
            let secs = d.as_secs_f64();
            if secs > 0.0 {
                self.total_tokens() as f64 / secs
            } else {
                0.0
            }
        })
    }
}

impl std::fmt::Display for RunMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Agent Run Metrics")?;
        writeln!(f, "  Steps:      {}", self.steps)?;
        writeln!(
            f,
            "  Tokens:     {} (in: {}, out: {})",
            self.total_tokens(),
            self.input_tokens,
            self.output_tokens
        )?;
        if let Some(d) = self.duration {
            writeln!(f, "  Duration:   {:.2}s", d.as_secs_f64())?;
            if let Some(rate) = self.tokens_per_second() {
                writeln!(f, "  Rate:       {rate:.1} tok/s")?;
            }
        }
        writeln!(f, "  Tool calls: {}", self.tool_calls)?;
        writeln!(f, "  Errors:     {}", self.errors)?;
        Ok(())
    }
}

/// Create a metrics handler that updates the given collector.
pub fn metrics_handler(
    collector: Arc<MetricsCollector>,
) -> impl Fn(&dyn MemoryStep, &CallbackContext) + Send + Sync + 'static {
    move |step, _ctx| {
        if let Some(task) = step.as_any().downcast_ref::<TaskStep>() {
            let _ = task;
            collector.tasks.fetch_add(1, Ordering::SeqCst);
        } else if let Some(action) = step.as_any().downcast_ref::<ActionStep>() {
            collector.steps.fetch_add(1, Ordering::SeqCst);

            if let Some(tool_calls) = &action.tool_calls {
                collector
                    .tool_calls
                    .fetch_add(tool_calls.len(), Ordering::SeqCst);
            }

            if action.error.is_some() {
                collector.errors.fetch_add(1, Ordering::SeqCst);
            }

            if let Some(usage) = &action.token_usage {
                collector
                    .input_tokens
                    .fetch_add(u64::from(usage.input_tokens), Ordering::SeqCst);
                collector
                    .output_tokens
                    .fetch_add(u64::from(usage.output_tokens), Ordering::SeqCst);
            }
        } else if let Some(_final) = step.as_any().downcast_ref::<FinalAnswerStep>() {
            collector.final_answers.fetch_add(1, Ordering::SeqCst);
        }
    }
}

/// Create a tracing handler that emits spans for each step.
///
/// Integrates with the `tracing` ecosystem for observability.
pub fn tracing_handler() -> impl Fn(&dyn MemoryStep, &CallbackContext) + Send + Sync + 'static {
    |step, ctx| {
        let step_type = std::any::type_name_of_val(step)
            .rsplit("::")
            .next()
            .unwrap_or("Unknown");

        let span = tracing::info_span!(
            "callback",
            step_type = step_type,
            step_number = ctx.step_number,
            agent = ctx.agent_name.as_deref(),
        );

        let _guard = span.enter();

        if let Some(action) = step.as_any().downcast_ref::<ActionStep>()
            && let Some(usage) = &action.token_usage
        {
            tracing::info!(
                input_tokens = usage.input_tokens,
                output_tokens = usage.output_tokens,
                "Token usage recorded"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Timing;

    #[test]
    fn test_metrics_collector() {
        let collector = Arc::new(MetricsCollector::new());
        let handler = metrics_handler(Arc::clone(&collector));

        let ctx = CallbackContext::new(1, 10);

        // Task step
        let task = TaskStep::new("test");
        handler(&task, &ctx);

        // Action step with tool call
        let action = ActionStep {
            step_number: 1,
            timing: Timing::start_now(),
            tool_calls: Some(vec![crate::memory::ToolCall::new(
                "1",
                "test_tool",
                serde_json::json!({}),
            )]),
            ..Default::default()
        };
        handler(&action, &ctx);

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.tasks, 1);
        assert_eq!(snapshot.steps, 1);
        assert_eq!(snapshot.tool_calls, 1);
    }

    #[test]
    fn test_logging_config() {
        let config = LoggingConfig::default();
        assert!(config.log_actions);

        let minimal = LoggingConfig::minimal();
        assert!(!minimal.log_actions);
        assert!(minimal.log_final_answers);

        let verbose = LoggingConfig::verbose().with_prefix("MyAgent");
        assert!(verbose.detailed);
        assert_eq!(verbose.prefix, Some("MyAgent".to_string()));
    }
}
