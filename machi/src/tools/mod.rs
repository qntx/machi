//! Built-in tools for agents.
//!
//! This module provides a collection of commonly used tools that agents can use
//! out of the box. All tools implement the [`Tool`](crate::tool::Tool) trait
//! and can be added to an [`Agent`](crate::agent::Agent) via its builder API.
//!
//! # Tool Categories
//!
//! - **Filesystem**: [`ReadFileTool`], [`WriteFileTool`], [`EditFileTool`],
//!   [`ListDirTool`] — file I/O operations
//! - **Shell**: [`ExecTool`] — command-line execution with timeout support
//! - **Web**: [`WebSearchTool`] — web search with pluggable providers
//!   ([`TavilyProvider`], [`SearxngProvider`], [`BraveProvider`],
//!   [`DuckDuckGoProvider`], [`BingProvider`])
//!
//! # Examples
//!
//! ```rust
//! use machi::tools::{ReadFileTool, ExecTool, create_tool, fs_tools};
//!
//! // Create individual tools
//! let read = ReadFileTool::new();
//! let exec = ExecTool::new();
//!
//! // Create a tool by name
//! let tool = create_tool("read_file");
//! assert!(tool.is_some());
//!
//! // Get all filesystem tools at once
//! let tools = fs_tools();
//! assert_eq!(tools.len(), 4);
//! ```

mod fs;
mod shell;
mod web_search;

pub use fs::{
    EditFileArgs, EditFileTool, ListDirArgs, ListDirTool, ReadFileArgs, ReadFileTool,
    WriteFileArgs, WriteFileTool,
};
pub use shell::{ExecArgs, ExecResult, ExecTool};
pub use web_search::{
    BingProvider, BoxedSearchProvider, BraveProvider, DuckDuckGoProvider, SearchProvider,
    SearchResult, SearxngProvider, TavilyProvider, WebSearchArgs, WebSearchTool,
};

use crate::tool::BoxedTool;

/// Names of all built-in tools provided by this module.
pub const BUILTIN_TOOL_NAMES: &[&str] = &[
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "exec",
    "web_search",
];

/// Create a built-in tool by name.
///
/// Returns `None` if the name is not recognized.
///
/// # Supported names
///
/// | Name | Tool |
/// |------|------|
/// | `"read_file"` | [`ReadFileTool`] |
/// | `"write_file"` | [`WriteFileTool`] |
/// | `"edit_file"` | [`EditFileTool`] |
/// | `"list_dir"` | [`ListDirTool`] |
/// | `"exec"` | [`ExecTool`] |
///
/// **Note:** `"web_search"` is *not* included here because it requires a
/// [`SearchProvider`] configuration.  Use [`WebSearchTool::tavily`],
/// [`WebSearchTool::searxng`], or [`WebSearchTool::brave`] directly.
#[must_use]
pub fn create_tool(name: &str) -> Option<BoxedTool> {
    match name {
        "read_file" => Some(Box::new(ReadFileTool::new())),
        "write_file" => Some(Box::new(WriteFileTool::new())),
        "edit_file" => Some(Box::new(EditFileTool::new())),
        "list_dir" => Some(Box::new(ListDirTool::new())),
        "exec" => Some(Box::new(ExecTool::new())),
        _ => None,
    }
}

/// Create multiple built-in tools by name.
///
/// Unrecognized names are silently skipped.
#[must_use]
#[allow(clippy::module_name_repetitions)]
pub fn create_tools(names: &[&str]) -> Vec<BoxedTool> {
    names.iter().filter_map(|name| create_tool(name)).collect()
}

/// Check if a tool name corresponds to a known built-in tool.
#[must_use]
pub fn is_builtin_tool(name: &str) -> bool {
    BUILTIN_TOOL_NAMES.contains(&name)
}

/// Get the default filesystem tools.
///
/// Returns: [`ReadFileTool`], [`WriteFileTool`], [`EditFileTool`],
/// [`ListDirTool`].
#[must_use]
#[allow(clippy::module_name_repetitions)]
pub fn fs_tools() -> Vec<BoxedTool> {
    vec![
        Box::new(ReadFileTool::new()),
        Box::new(WriteFileTool::new()),
        Box::new(EditFileTool::new()),
        Box::new(ListDirTool::new()),
    ]
}

/// Get all built-in tools.
///
/// Returns every tool listed in [`BUILTIN_TOOL_NAMES`].
#[must_use]
#[allow(clippy::module_name_repetitions)]
pub fn all_tools() -> Vec<BoxedTool> {
    vec![
        Box::new(ReadFileTool::new()),
        Box::new(WriteFileTool::new()),
        Box::new(EditFileTool::new()),
        Box::new(ListDirTool::new()),
        Box::new(ExecTool::new()),
    ]
}
