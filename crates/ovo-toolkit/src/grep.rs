//! Cwd-jailed recursive text search.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use ovo_tools::stream::ToolStream;
use ovo_tools::{
    DynTool, ToolCallContext, ToolError, ToolMetadata, ToolProgress, ToolResult, with_progress,
};
use serde_json::{Value, json};
use tokio::fs;

use crate::jail::{jail_denied, resolve_root};
use crate::path_util::resolve_jailed;

/// Default max matches returned.
pub const DEFAULT_MAX_MATCHES: usize = 50;
/// Default max file size scanned.
pub const DEFAULT_MAX_FILE_BYTES: u64 = 512 * 1024;

/// Grep-like search under the jail root.
#[derive(Debug, Clone)]
pub struct GrepTool {
    /// Jail root.
    pub jail_root: Option<PathBuf>,
    /// Max matches.
    pub max_matches: usize,
    /// Skip files larger than this.
    pub max_file_bytes: u64,
}

impl Default for GrepTool {
    fn default() -> Self {
        Self {
            jail_root: None,
            max_matches: DEFAULT_MAX_MATCHES,
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
        }
    }
}

impl GrepTool {
    /// Explicit jail.
    #[must_use]
    pub fn with_jail(root: impl Into<PathBuf>) -> Self {
        Self {
            jail_root: Some(root.into()),
            ..Self::default()
        }
    }
}

#[async_trait]
impl DynTool for GrepTool {
    fn name(&self) -> &'static str {
        "grep"
    }

    fn description(&self) -> &'static str {
        "Search for a substring in text files under the workspace jail. \
         Args: pattern, path (optional directory/file, default \".\")."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Substring to find (literal)" },
                "path": {
                    "type": "string",
                    "description": "Relative path (file or directory). Default: ."
                }
            },
            "required": ["pattern"],
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
        let pattern = arguments
            .get("pattern")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .filter(|s| !s.is_empty());
        let path = arguments
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or(".")
            .to_owned();

        let root = match resolve_root(self.jail_root.as_ref(), &ctx, "grep") {
            Ok(r) => r,
            Err(e) => return ovo_tools::terminal_only(Err(e)),
        };
        let Some(pattern) = pattern else {
            return ovo_tools::terminal_only(Err(ovo_tools::error::codes::invalid_args(
                "grep requires non-empty pattern",
            )));
        };
        let max_matches = self.max_matches;
        let max_file_bytes = self.max_file_bytes;
        with_progress(
            vec![ToolProgress::text(format!("grep {pattern:?} in {path}"))],
            move || async move {
                let start = resolve_jailed(&root, &path).map_err(jail_denied)?;
                let mut matches = Vec::new();
                search_path(
                    &start,
                    &root,
                    &pattern,
                    max_matches,
                    max_file_bytes,
                    &mut matches,
                )
                .await?;
                let truncated = matches.len() >= max_matches;
                let content = if matches.is_empty() {
                    "no matches".to_owned()
                } else {
                    matches.join("\n")
                };
                Ok(ToolResult {
                    content,
                    structured: Some(json!({
                        "pattern": pattern,
                        "path": path,
                        "match_count": matches.len(),
                        "truncated": truncated,
                        "matches": matches,
                    })),
                    is_error: false,
                })
            },
        )
    }
}

async fn search_path(
    path: &Path,
    root: &Path,
    pattern: &str,
    max_matches: usize,
    max_file_bytes: u64,
    out: &mut Vec<String>,
) -> Result<(), ToolError> {
    if out.len() >= max_matches {
        return Ok(());
    }
    let meta = fs::metadata(path)
        .await
        .map_err(|e| ovo_tools::error::codes::execution(format!("stat {}: {e}", path.display())))?;
    if meta.is_file() {
        if meta.len() > max_file_bytes {
            return Ok(());
        }
        let bytes = fs::read(path).await.map_err(|e| {
            ovo_tools::error::codes::execution(format!("read {}: {e}", path.display()))
        })?;
        // Skip likely-binary files.
        if bytes.iter().take(1024).any(|&b| b == 0) {
            return Ok(());
        }
        let text = String::from_utf8_lossy(&bytes);
        let rel = path.strip_prefix(root).unwrap_or(path);
        for (i, line) in text.lines().enumerate() {
            if line.contains(pattern) {
                out.push(format!("{}:{}:{line}", rel.display(), i + 1));
                if out.len() >= max_matches {
                    return Ok(());
                }
            }
        }
        return Ok(());
    }
    if meta.is_dir() {
        let mut rd = fs::read_dir(path).await.map_err(|e| {
            ovo_tools::error::codes::execution(format!("read_dir {}: {e}", path.display()))
        })?;
        while let Some(entry) = rd
            .next_entry()
            .await
            .map_err(|e| ovo_tools::error::codes::execution(format!("read_dir next: {e}")))?
        {
            let name = entry.file_name();
            if name == ".git" || name == "target" || name == "node_modules" {
                continue;
            }
            Box::pin(search_path(
                &entry.path(),
                root,
                pattern,
                max_matches,
                max_file_bytes,
                out,
            ))
            .await?;
            if out.len() >= max_matches {
                break;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[tokio::test]
    async fn finds_literal() {
        let dir = tempdir().expect("temp");
        std::fs::write(dir.path().join("a.rs"), "fn foo() {}\nfn bar() {}\n").expect("w");
        let tool = GrepTool::with_jail(dir.path());
        let r = tool
            .call(
                ToolCallContext {
                    cwd: Some(dir.path().to_path_buf()),
                    ..ToolCallContext::default()
                },
                json!({"pattern": "foo"}),
            )
            .await
            .expect("grep");
        assert!(r.content.contains("foo"), "{}", r.content);
    }
}
