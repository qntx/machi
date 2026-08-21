//! Shared jail-root resolution for toolkit tools.

use std::path::PathBuf;

use ovo_tools::{ToolCallContext, ToolError};

/// Resolve jail root from tool config or call context.
///
/// # Errors
///
/// Returns invalid-args when neither jail nor context cwd is set.
pub fn resolve_root(
    jail_root: Option<&PathBuf>,
    ctx: &ToolCallContext,
    tool_name: &str,
) -> Result<PathBuf, ToolError> {
    if let Some(root) = jail_root {
        return Ok(root.clone());
    }
    ctx.cwd.clone().ok_or_else(|| {
        ovo_tools::error::codes::invalid_args(format!(
            "{tool_name} requires jail_root or ToolCallContext.cwd"
        ))
    })
}

/// Map path jail errors to denied tool errors.
#[must_use]
pub fn jail_denied(err: impl std::fmt::Display) -> ToolError {
    ovo_tools::error::codes::denied(format!("path jail: {err}"))
}
