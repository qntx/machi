//! Cwd-jailed file read tool.

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

/// Default max bytes returned to the model.
pub const DEFAULT_MAX_BYTES: usize = 256 * 1024;

/// Read a UTF-8 (lossy) text file under a jail root.
#[derive(Debug, Clone)]
pub struct ReadFileTool {
    /// Jail root. When unset, uses `ToolCallContext::cwd` at call time.
    pub jail_root: Option<PathBuf>,
    /// Max bytes to return (truncate with a notice).
    pub max_bytes: usize,
}

impl Default for ReadFileTool {
    fn default() -> Self {
        Self {
            jail_root: None,
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }
}

impl ReadFileTool {
    /// Create with an explicit jail root.
    #[must_use]
    pub fn with_jail(root: impl Into<PathBuf>) -> Self {
        Self {
            jail_root: Some(root.into()),
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }

    /// Override max bytes.
    #[must_use]
    pub const fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes;
        self
    }
}

#[async_trait]
impl DynTool for ReadFileTool {
    fn name(&self) -> &'static str {
        "read_file"
    }

    fn description(&self) -> &'static str {
        "Read a text file under the workspace jail. Args: path (relative to cwd)."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path under the workspace root"
                }
            },
            "required": ["path"],
            "additionalProperties": false
        })
    }

    fn metadata(&self) -> ToolMetadata {
        ToolMetadata::read_only()
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

        let root = match resolve_root(self.jail_root.as_ref(), &ctx, "read_file") {
            Ok(r) => r,
            Err(e) => return ovo_tools::terminal_only(Err(e)),
        };
        let Some(path) = path else {
            return ovo_tools::terminal_only(Err(ovo_tools::error::codes::invalid_args(
                "read_file requires non-empty path",
            )));
        };

        let max_bytes = self.max_bytes;
        with_progress(
            vec![ToolProgress::text(format!("reading {path}"))],
            move || async move {
                let resolved = resolve_jailed(&root, &path).map_err(jail_denied)?;
                let bytes = fs::read(&resolved).await.map_err(|e| {
                    ovo_tools::error::codes::execution(format!(
                        "failed to read {}: {e}",
                        resolved.display()
                    ))
                })?;
                let truncated = bytes.len() > max_bytes;
                let slice = if truncated {
                    bytes.get(..max_bytes).unwrap_or(&bytes)
                } else {
                    bytes.as_slice()
                };
                let mut text = String::from_utf8_lossy(slice).into_owned();
                if truncated {
                    use std::fmt::Write as _;
                    let _ = write!(
                        text,
                        "\n\n[truncated: showing first {max_bytes} of {} bytes]",
                        bytes.len()
                    );
                }
                Ok(ToolResult {
                    content: text,
                    structured: Some(json!({
                        "path": path,
                        "resolved": resolved.display().to_string(),
                        "bytes": bytes.len(),
                        "truncated": truncated,
                    })),
                    is_error: false,
                })
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ovo_tools::{DispatchRequest, ToolDispatch, ToolRegistry};
    use ovo_types::{ToolCall, ToolCallId};
    use tempfile::tempdir;

    use super::*;

    #[tokio::test]
    async fn reads_jailed_file() {
        let dir = tempdir().expect("temp");
        let file = dir.path().join("hello.txt");
        std::fs::write(&file, "hello world").expect("write");

        let tool = Arc::new(ReadFileTool::with_jail(dir.path()));
        let registry = ToolRegistry::from_tools(vec![tool]);
        let ctx = ToolCallContext {
            cwd: Some(dir.path().to_path_buf()),
            ..ToolCallContext::default()
        };
        let dispatch = ToolDispatch::default();
        let id = ToolCallId::new("c1").expect("id");
        let outs = dispatch
            .execute_batch(
                &registry,
                ctx,
                vec![DispatchRequest {
                    call: ToolCall {
                        id,
                        name: "read_file".into(),
                        arguments: json!({"path": "hello.txt"}),
                    },
                }],
            )
            .await;
        let out = outs.into_iter().next().expect("one");
        let result = out.result.expect("ok");
        assert!(result.content.contains("hello world"), "{}", result.content);
    }

    #[tokio::test]
    async fn denies_escape() {
        let dir = tempdir().expect("temp");
        let tool = ReadFileTool::with_jail(dir.path());
        let ctx = ToolCallContext {
            cwd: Some(dir.path().to_path_buf()),
            ..ToolCallContext::default()
        };
        let err = tool
            .call(ctx, json!({"path": "../secret"}))
            .await
            .expect_err("escape");
        assert_eq!(err.code(), ovo_types::ErrorCode::ToolDenied);
    }
}
