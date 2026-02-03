//! Built-in tools for agents.
//!
//! This module provides a collection of commonly used tools that agents
//! can use out of the box.

mod final_answer;
mod user_input;
mod visit_webpage;
mod web_search;

pub use final_answer::{FinalAnswerArgs, FinalAnswerTool};
pub use user_input::UserInputTool;
pub use visit_webpage::VisitWebpageTool;
pub use web_search::{DuckDuckGoSearchTool, WebSearchTool};

use crate::tool::BoxedTool;

/// Get a map of all default tools.
pub fn default_tools() -> Vec<BoxedTool> {
    vec![Box::new(FinalAnswerTool)]
}

/// Tool names that are available by default.
pub const DEFAULT_TOOL_NAMES: &[&str] =
    &["final_answer", "web_search", "visit_webpage", "user_input"];
