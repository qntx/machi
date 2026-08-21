//! Cwd-jailed recursive path listing with simple glob patterns.

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

/// Default max paths returned.
pub const DEFAULT_MAX_RESULTS: usize = 200;

/// List files under the jail matching a simple glob (`*` / `?` / `**`).
#[derive(Debug, Clone)]
pub struct GlobTool {
    /// Jail root.
    pub jail_root: Option<PathBuf>,
    /// Max results.
    pub max_results: usize,
}

impl Default for GlobTool {
    fn default() -> Self {
        Self {
            jail_root: None,
            max_results: DEFAULT_MAX_RESULTS,
        }
    }
}

impl GlobTool {
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
impl DynTool for GlobTool {
    fn name(&self) -> &'static str {
        "glob"
    }

    fn description(&self) -> &'static str {
        "List files under the workspace jail. Args: pattern (e.g. \"**/*.rs\"), \
         path (optional root, default \".\")."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern relative to path (** and * supported)"
                },
                "path": {
                    "type": "string",
                    "description": "Directory under jail (default .)"
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
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned);
        let path = arguments
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or(".")
            .to_owned();

        let root = match resolve_root(self.jail_root.as_ref(), &ctx, "glob") {
            Ok(r) => r,
            Err(e) => return ovo_tools::terminal_only(Err(e)),
        };
        let Some(pattern) = pattern else {
            return ovo_tools::terminal_only(Err(ovo_tools::error::codes::invalid_args(
                "glob requires non-empty pattern",
            )));
        };
        let max = self.max_results;
        with_progress(
            vec![ToolProgress::text(format!("glob {pattern} under {path}"))],
            move || async move {
                let start = resolve_jailed(&root, &path).map_err(jail_denied)?;
                let mut hits = Vec::new();
                walk_match(&start, &root, &pattern, max, &mut hits).await?;
                let truncated = hits.len() >= max;
                let content = if hits.is_empty() {
                    "no matches".to_owned()
                } else {
                    hits.join("\n")
                };
                Ok(ToolResult {
                    content,
                    structured: Some(json!({
                        "pattern": pattern,
                        "path": path,
                        "count": hits.len(),
                        "truncated": truncated,
                        "paths": hits,
                    })),
                    is_error: false,
                })
            },
        )
    }
}

async fn walk_match(
    dir: &Path,
    root: &Path,
    pattern: &str,
    max: usize,
    out: &mut Vec<String>,
) -> Result<(), ToolError> {
    if out.len() >= max {
        return Ok(());
    }
    let meta = fs::metadata(dir)
        .await
        .map_err(|e| ovo_tools::error::codes::execution(format!("stat {}: {e}", dir.display())))?;
    if meta.is_file() {
        push_if_match(dir, root, pattern, out);
        return Ok(());
    }
    if !meta.is_dir() {
        return Ok(());
    }
    let mut rd = fs::read_dir(dir).await.map_err(|e| {
        ovo_tools::error::codes::execution(format!("read_dir {}: {e}", dir.display()))
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
        let p = entry.path();
        let ft = entry
            .file_type()
            .await
            .map_err(|e| ovo_tools::error::codes::execution(format!("file_type: {e}")))?;
        if ft.is_dir() {
            Box::pin(walk_match(&p, root, pattern, max, out)).await?;
        } else if ft.is_file() {
            push_if_match(&p, root, pattern, out);
        }
        if out.len() >= max {
            break;
        }
    }
    Ok(())
}

fn push_if_match(path: &Path, root: &Path, pattern: &str, out: &mut Vec<String>) {
    let rel = path.strip_prefix(root).map_or_else(
        |_| path.to_string_lossy().into_owned(),
        |p| p.display().to_string(),
    );
    let rel_norm = rel.replace('\\', "/");
    if glob_match(pattern, &rel_norm) {
        out.push(rel_norm);
    }
}

/// Minimal glob: `*` (segment), `**` (any depth), `?` (one char). Case-sensitive.
#[must_use]
pub fn glob_match(pattern: &str, path: &str) -> bool {
    match_glob(pattern.as_bytes(), path.as_bytes())
}

fn match_glob(pat: &[u8], text: &[u8]) -> bool {
    match (pat.first(), text.first()) {
        (None, None) => true,
        (None, Some(_)) => false,
        (Some(b'*'), _) if pat.get(1) == Some(&b'*') => {
            let rest = pat.get(2..).unwrap_or(&[]);
            let rest = if rest.first() == Some(&b'/') {
                rest.get(1..).unwrap_or(&[])
            } else {
                rest
            };
            if rest.is_empty() {
                return true;
            }
            (0..=text.len()).any(|i| text.get(i..).is_some_and(|suffix| match_glob(rest, suffix)))
        }
        (Some(b'*'), _) => {
            let rest = pat.get(1..).unwrap_or(&[]);
            let mut i = 0usize;
            loop {
                if text.get(i..).is_some_and(|suffix| match_glob(rest, suffix)) {
                    return true;
                }
                if text.get(i) == Some(&b'/') || i >= text.len() {
                    return false;
                }
                i = i.saturating_add(1);
            }
        }
        (Some(b'?'), Some(c)) if *c != b'/' => {
            match_glob(pat.get(1..).unwrap_or(&[]), text.get(1..).unwrap_or(&[]))
        }
        (Some(b'?'), _) => false,
        (Some(pc), Some(tc)) if pc == tc => {
            match_glob(pat.get(1..).unwrap_or(&[]), text.get(1..).unwrap_or(&[]))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn glob_semantics() {
        assert!(glob_match("**/*.rs", "src/lib.rs"));
        assert!(glob_match("*.rs", "lib.rs"));
        assert!(!glob_match("*.rs", "src/lib.rs"));
        assert!(glob_match("src/???.rs", "src/lib.rs"));
    }

    #[tokio::test]
    async fn lists_files() {
        let dir = tempdir().expect("tmp");
        std::fs::create_dir_all(dir.path().join("src")).expect("mkdir");
        std::fs::write(dir.path().join("src/a.rs"), "a").expect("w");
        std::fs::write(dir.path().join("src/b.txt"), "b").expect("w");
        let tool = GlobTool::with_jail(dir.path());
        let r = tool
            .call(
                ToolCallContext {
                    cwd: Some(dir.path().to_path_buf()),
                    ..ToolCallContext::default()
                },
                json!({"pattern": "**/*.rs"}),
            )
            .await
            .expect("glob");
        assert!(r.content.contains("src/a.rs"), "{}", r.content);
        assert!(!r.content.contains("b.txt"), "{}", r.content);
    }
}
