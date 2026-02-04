//! Built-in tools for agents.
//!
//! This module provides a collection of commonly used tools that agents
//! can use out of the box, following the smolagents architecture pattern.
//!
//! # Tool Categories
//!
//! - **Essential**: `FinalAnswerTool` - Required for all agents
//! - **Web**: `WebSearchTool`, `VisitWebpageTool` - Web browsing capabilities
//! - **Interactive**: `UserInputTool` - Human-in-the-loop support
//!
//! # Example
//!
//! ```rust,ignore
//! use machi::tools::{base_tools, create_tool};
//!
//! // Get common base tools
//! let tools = base_tools();
//!
//! // Create a specific tool by name
//! let search = create_tool("web_search").unwrap();
//! ```

mod final_answer;
mod user_input;
mod visit_webpage;
mod web_search;

pub use final_answer::{FinalAnswerArgs, FinalAnswerTool};
pub use user_input::{UserInputArgs, UserInputTool};
pub use visit_webpage::{VisitWebpageArgs, VisitWebpageTool};
pub use web_search::{
    DuckDuckGoSearchTool, SearchEngine, SearchResult, WebSearchArgs, WebSearchTool,
};

use crate::tool::BoxedTool;

/// Tool names that are available as built-in tools.
pub const BUILTIN_TOOL_NAMES: &[&str] = &[
    "final_answer",
    "web_search",
    "visit_webpage",
    "user_input",
    "duckduckgo_search",
];

/// Create a tool by name.
///
/// Returns `None` if the tool name is not recognized.
///
/// # Supported Tools
///
/// - `"final_answer"` - Tool for providing final answers
/// - `"web_search"` - Web search using configurable backend
/// - `"duckduckgo_search"` - Web search using DuckDuckGo
/// - `"visit_webpage"` - Visit and read webpage content
/// - `"user_input"` - Request input from user
#[must_use]
pub fn create_tool(name: &str) -> Option<BoxedTool> {
    match name {
        "final_answer" => Some(Box::new(FinalAnswerTool)),
        "web_search" => Some(Box::new(WebSearchTool::default())),
        "duckduckgo_search" => Some(Box::new(DuckDuckGoSearchTool::new())),
        "visit_webpage" => Some(Box::new(VisitWebpageTool::default())),
        "user_input" => Some(Box::new(UserInputTool)),
        _ => None,
    }
}

/// Create multiple tools by name.
///
/// Skips unrecognized tool names silently.
#[must_use]
pub fn create_tools(names: &[&str]) -> Vec<BoxedTool> {
    names.iter().filter_map(|name| create_tool(name)).collect()
}

/// Get the default tools for agents.
///
/// Returns only the `FinalAnswerTool` as it's the essential tool for
/// concluding agent tasks. Use `base_tools()` for a more complete set.
#[must_use]
pub fn default_tools() -> Vec<BoxedTool> {
    vec![Box::new(FinalAnswerTool)]
}

/// Get base tools commonly used by agents.
///
/// Similar to smolagents' `add_base_tools=True` option, this provides:
/// - `FinalAnswerTool` - for providing final answers
/// - `VisitWebpageTool` - for reading webpage content
///
/// Does NOT include `WebSearchTool` by default as it may require API keys.
#[must_use]
pub fn base_tools() -> Vec<BoxedTool> {
    vec![
        Box::new(FinalAnswerTool),
        Box::new(VisitWebpageTool::default()),
    ]
}

/// Get all available built-in tools.
///
/// Returns a vector containing all built-in tools:
/// - `FinalAnswerTool` - for providing final answers
/// - `WebSearchTool` - for web searches
/// - `VisitWebpageTool` - for visiting webpages
/// - `UserInputTool` - for interactive user input
#[must_use]
pub fn all_tools() -> Vec<BoxedTool> {
    vec![
        Box::new(FinalAnswerTool),
        Box::new(WebSearchTool::default()),
        Box::new(VisitWebpageTool::default()),
        Box::new(UserInputTool),
    ]
}

/// Check if a tool name is a known built-in tool.
#[must_use]
pub fn is_builtin_tool(name: &str) -> bool {
    BUILTIN_TOOL_NAMES.contains(&name)
}
