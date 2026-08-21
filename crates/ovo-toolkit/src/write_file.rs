//! Cwd-jailed file write tool.

use std::path::PathBuf;

use async_trait::async_trait;
use ovo_tools::stream::ToolStream;
use ovo_tools::{
    DynTool, ToolCallContext, ToolError, ToolMetadata, ToolProgress, ToolResult, with_progress,
};
use serde_json::{Value, json};
use tokio::fs;

use crate::jail::{jail_denied, resolve_root};
use crate::path_util::resolve_jailed;

/// Default max write size.
pub const DEFAULT_MAX_BYTES: usize = 1024 * 1024;

/// Write a text file under a jail root (creates parent dirs).
#[derive(Debug, Clone)]
pub struct WriteFileTool {
    /// Jail root. When unset, uses `ToolCallContext::cwd`.
    pub jail_root: Option<PathBuf>,
    /// Max content bytes accepted.
    pub max_bytes: usize,
}

impl Default for WriteFileTool {
    fn default() -> Self {
        Self {
            jail_root: None,
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }
}

impl WriteFileTool {
    /// Explicit jail root.
    #[must_use]
    pub fn with_jail(root: impl Into<PathBuf>) -> Self {
        Self {
            jail_root: Some(root.into()),
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }
}

#[async_trait]
impl DynTool for WriteFileTool {
    fn name(&self) -> &'static str {
        "write_file"
    }

    fn description(&self) -> &'static str {
        "Write UTF-8 text to a file under the workspace jail. Creates parent directories. \
         Args: path, content."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Relative path under cwd" },
                "content": { "type": "string", "description": "File contents" }
            },
            "required": ["path", "content"],
            "additionalProperties": false
        })
    }

    fn metadata(&self) -> ToolMetadata {
        ToolMetadata::exclusive_write()
    }

    async fn call(&self, ctx: ToolCallContext, arguments: Value) -> Result<ToolResult, ToolError> {
        let stream = self.execute(ctx, arguments).await;
        ovo_tools::drain_terminal(stream).await
    }

    async fn execute(&self, ctx: ToolCallContext, arguments: Value) -> ToolStream {
        let path = arguments
            .get("path")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned);
        let content = arguments
            .get("content")
            .and_then(Value::as_str)
            .map(str::to_owned);

        let root = match resolve_root(self.jail_root.as_ref(), &ctx, "write_file") {
            Ok(r) => r,
            Err(e) => return ovo_tools::terminal_only(Err(e)),
        };
        let Some(path) = path else {
            return ovo_tools::terminal_only(Err(ovo_tools::error::codes::invalid_args(
                "write_file requires non-empty path",
            )));
        };
        let Some(content) = content else {
            return ovo_tools::terminal_only(Err(ovo_tools::error::codes::invalid_args(
                "write_file requires content string",
            )));
        };
        let max_bytes = self.max_bytes;
        with_progress(
            vec![ToolProgress::text(format!("writing {path}"))],
            move || async move {
                if content.len() > max_bytes {
                    return Err(ovo_tools::error::codes::invalid_args(format!(
                        "content exceeds max_bytes ({max_bytes})"
                    )));
                }
                let resolved = resolve_jailed(&root, &path).map_err(jail_denied)?;
                if let Some(parent) = resolved.parent() {
                    fs::create_dir_all(parent).await.map_err(|e| {
                        ovo_tools::error::codes::execution(format!(
                            "create_dir_all {}: {e}",
                            parent.display()
                        ))
                    })?;
                }
                fs::write(&resolved, content.as_bytes())
                    .await
                    .map_err(|e| {
                        ovo_tools::error::codes::execution(format!(
                            "write {}: {e}",
                            resolved.display()
                        ))
                    })?;
                Ok(ToolResult {
                    content: format!("wrote {} bytes to {path}", content.len()),
                    structured: Some(json!({
                        "path": path,
                        "resolved": resolved.display().to_string(),
                        "bytes": content.len(),
                    })),
                    is_error: false,
                })
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[tokio::test]
    async fn writes_and_reads_back() {
        let dir = tempdir().expect("temp");
        let tool = WriteFileTool::with_jail(dir.path());
        let ctx = ToolCallContext {
            cwd: Some(dir.path().to_path_buf()),
            ..ToolCallContext::default()
        };
        let r = tool
            .call(ctx, json!({"path": "nested/a.txt", "content": "hello"}))
            .await
            .expect("write");
        assert!(r.content.contains("wrote"), "{}", r.content);
        let body = std::fs::read_to_string(dir.path().join("nested/a.txt")).expect("read");
        assert_eq!(body, "hello");
    }

    #[tokio::test]
    async fn denies_escape() {
        let dir = tempdir().expect("temp");
        let tool = WriteFileTool::with_jail(dir.path());
        let err = tool
            .call(
                ToolCallContext::default(),
                json!({"path": "../x", "content": "no"}),
            )
            .await
            .expect_err("escape");
        assert_eq!(err.code(), ovo_types::ErrorCode::ToolDenied);
    }
}
