//! Run options for agent execution.
//!
//! This module provides a unified options struct for configuring agent runs,
//! following the smolagents pattern of a single run() method with options.

use std::collections::HashMap;

use serde_json::Value;

use crate::multimodal::AgentImage;

/// Options for running an agent.
///
/// This provides a builder-style API for configuring agent execution,
/// consolidating all the various run options into a single struct.
///
/// # Example
///
/// ```rust,ignore
/// use machi::prelude::*;
///
/// // Simple run
/// let result = agent.run(RunOptions::new("What is 2+2?")).await?;
///
/// // Run with images
/// let result = agent.run(
///     RunOptions::new("Describe this image")
///         .images(vec![image])
/// ).await?;
///
/// // Run with full options
/// let result = agent.run(
///     RunOptions::new("Complex task")
///         .images(vec![img1, img2])
///         .context("user_id", json!("123"))
///         .detailed(true)
/// ).await?;
/// ```
#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    /// The task to perform.
    pub(crate) task: String,
    /// Images for vision models.
    pub(crate) images: Vec<AgentImage>,
    /// Additional context variables.
    pub(crate) context: HashMap<String, Value>,
    /// Whether to return detailed RunResult instead of just the answer.
    pub(crate) detailed: bool,
    /// Whether to reset the agent before running.
    pub(crate) reset: bool,
}

impl RunOptions {
    /// Create new run options with the given task.
    #[must_use]
    pub fn new(task: impl Into<String>) -> Self {
        Self {
            task: task.into(),
            images: Vec::new(),
            context: HashMap::new(),
            detailed: false,
            reset: true,
        }
    }

    /// Add images for vision models.
    #[must_use]
    pub fn images(mut self, images: Vec<AgentImage>) -> Self {
        self.images = images;
        self
    }

    /// Add a single image.
    #[must_use]
    pub fn image(mut self, image: AgentImage) -> Self {
        self.images.push(image);
        self
    }

    /// Add context variables.
    #[must_use]
    pub fn with_context(mut self, context: HashMap<String, Value>) -> Self {
        self.context = context;
        self
    }

    /// Add a single context variable.
    #[must_use]
    pub fn context(mut self, key: impl Into<String>, value: Value) -> Self {
        self.context.insert(key.into(), value);
        self
    }

    /// Request detailed RunResult output.
    #[must_use]
    pub const fn detailed(mut self) -> Self {
        self.detailed = true;
        self
    }

    /// Set whether to reset the agent before running (default: true).
    #[must_use]
    pub const fn reset(mut self, reset: bool) -> Self {
        self.reset = reset;
        self
    }

    /// Don't reset the agent, continue from previous state.
    #[must_use]
    pub const fn no_reset(mut self) -> Self {
        self.reset = false;
        self
    }
}

impl<S: Into<String>> From<S> for RunOptions {
    fn from(task: S) -> Self {
        Self::new(task)
    }
}
