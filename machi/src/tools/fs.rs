//! File system tools for agents.
//!
//! Provides tools for reading, writing, editing files and listing directories.
//! All I/O operations use `tokio::fs` for non-blocking execution.

use std::fmt::Write as _;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use tokio::fs;

use crate::error::ToolError;
use crate::tool::Tool;

/// Tool for reading file contents with optional line-range selection.
///
/// Supports reading entire files or specific line ranges (1-indexed, inclusive).
/// Has a configurable maximum file size to prevent excessive memory usage.
///
/// # Examples
///
/// ```rust
/// use machi::tools::ReadFileTool;
///
/// let tool = ReadFileTool::new().with_max_size(2 * 1024 * 1024); // 2 MB
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ReadFileTool {
    /// Maximum file size in bytes. Default: 1 `MiB`.
    max_size: usize,
}

impl Default for ReadFileTool {
    fn default() -> Self {
        Self {
            max_size: 1024 * 1024, // 1 MiB
        }
    }
}

impl ReadFileTool {
    /// Create a new read-file tool with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the maximum readable file size (in bytes).
    #[must_use]
    pub const fn with_max_size(mut self, max_size: usize) -> Self {
        self.max_size = max_size;
        self
    }
}

/// Arguments for [`ReadFileTool`].
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct ReadFileArgs {
    /// Path to the file to read.
    pub path: String,
    /// Optional start line (1-indexed, inclusive).
    pub start_line: Option<usize>,
    /// Optional end line (1-indexed, inclusive).
    pub end_line: Option<usize>,
}

#[async_trait]
impl Tool for ReadFileTool {
    const NAME: &'static str = "read_file";
    type Args = ReadFileArgs;
    type Output = String;
    type Error = ToolError;

    fn description(&self) -> String {
        "Read the contents of a file. Supports optional line range selection (1-indexed, inclusive)."
            .to_owned()
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Start line number (1-indexed, inclusive). Optional."
                },
                "end_line": {
                    "type": "integer",
                    "description": "End line number (1-indexed, inclusive). Optional."
                }
            },
            "required": ["path"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let path = Path::new(&args.path);

        // Existence check
        let metadata = fs::metadata(path)
            .await
            .map_err(|e| ToolError::Execution(format!("Cannot access '{}': {e}", args.path)))?;

        if !metadata.is_file() {
            return Err(ToolError::Execution(format!(
                "Not a regular file: {}",
                args.path
            )));
        }

        // Size guard
        if metadata.len() > self.max_size as u64 {
            return Err(ToolError::Execution(format!(
                "File too large: {} bytes (max: {} bytes)",
                metadata.len(),
                self.max_size
            )));
        }

        let content = fs::read_to_string(path)
            .await
            .map_err(|e| ToolError::Execution(format!("Failed to read '{}': {e}", args.path)))?;

        // Apply optional line range
        Ok(extract_lines(&content, args.start_line, args.end_line))
    }
}

/// Extract a line range from `content`.
///
/// Both `start` and `end` are 1-indexed and inclusive.
fn extract_lines(content: &str, start: Option<usize>, end: Option<usize>) -> String {
    if start.is_none() && end.is_none() {
        return content.to_owned();
    }

    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    let start_idx = start.map_or(0, |s| s.saturating_sub(1));
    let end_idx = end.map_or(total, |e| e.min(total));

    if start_idx >= total {
        return String::new();
    }

    lines[start_idx..end_idx].join("\n")
}

/// Tool for writing content to a file.
///
/// Supports creating new files, overwriting existing files, and appending.
/// Parent directories are created automatically when `create_dirs` is true.
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct WriteFileTool;

impl WriteFileTool {
    /// Create a new write-file tool.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

/// Arguments for [`WriteFileTool`].
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct WriteFileArgs {
    /// Path to the file to write.
    pub path: String,
    /// Content to write.
    pub content: String,
    /// Append instead of overwriting. Default: `false`.
    #[serde(default)]
    pub append: bool,
    /// Create parent directories if missing. Default: `true`.
    #[serde(default = "ret_true")]
    pub create_dirs: bool,
}

/// Default value for `create_dirs` (returns `true`).
const fn ret_true() -> bool {
    true
}

#[async_trait]
impl Tool for WriteFileTool {
    const NAME: &'static str = "write_file";
    type Args = WriteFileArgs;
    type Output = String;
    type Error = ToolError;

    fn description(&self) -> String {
        "Write content to a file. Can create new files or overwrite/append to existing ones."
            .to_owned()
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                },
                "append": {
                    "type": "boolean",
                    "description": "Append to existing content instead of overwriting. Default: false"
                },
                "create_dirs": {
                    "type": "boolean",
                    "description": "Create parent directories if they don't exist. Default: true"
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let path = Path::new(&args.path);

        // Ensure parent directories exist
        if args.create_dirs
            && let Some(parent) = path.parent()
        {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| ToolError::Execution(format!("Failed to create directories: {e}")))?;
        }

        if args.append {
            // Read-then-append to preserve existing content
            let existing = if path.is_file() {
                fs::read_to_string(path).await.map_err(|e| {
                    ToolError::Execution(format!("Failed to read existing file: {e}"))
                })?
            } else {
                String::new()
            };

            let merged = format!("{existing}{}", args.content);
            fs::write(path, &merged)
                .await
                .map_err(|e| ToolError::Execution(format!("Failed to write file: {e}")))?;

            Ok(format!(
                "Appended {} bytes to '{}'",
                args.content.len(),
                args.path
            ))
        } else {
            fs::write(path, &args.content)
                .await
                .map_err(|e| ToolError::Execution(format!("Failed to write file: {e}")))?;

            Ok(format!(
                "Wrote {} bytes to '{}'",
                args.content.len(),
                args.path
            ))
        }
    }
}

/// Tool for editing files via find-and-replace.
///
/// Locates `old_text` inside the target file and replaces it with `new_text`.
/// Supports replacing only the first occurrence or all occurrences.
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct EditFileTool;

impl EditFileTool {
    /// Create a new edit-file tool.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

/// Arguments for [`EditFileTool`].
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct EditFileArgs {
    /// Path to the file to edit.
    pub path: String,
    /// Text to find.
    pub old_text: String,
    /// Replacement text.
    pub new_text: String,
    /// Replace all occurrences. Default: `false` (first only).
    #[serde(default)]
    pub replace_all: bool,
}

#[async_trait]
impl Tool for EditFileTool {
    const NAME: &'static str = "edit_file";
    type Args = EditFileArgs;
    type Output = String;
    type Error = ToolError;

    fn description(&self) -> String {
        "Edit a file by replacing text. Finds old_text and replaces it with new_text.".to_owned()
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "old_text": {
                    "type": "string",
                    "description": "Text to find and replace"
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text"
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences. Default: false (first only)"
                }
            },
            "required": ["path", "old_text", "new_text"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let path = Path::new(&args.path);

        let content = fs::read_to_string(path)
            .await
            .map_err(|e| ToolError::Execution(format!("Failed to read '{}': {e}", args.path)))?;

        if !content.contains(&args.old_text) {
            // Show a truncated preview so the LLM can diagnose mismatches.
            let preview: String = args.old_text.chars().take(80).collect();
            return Err(ToolError::Execution(format!(
                "Text not found in '{}': '{preview}'",
                args.path
            )));
        }

        let (new_content, count) = if args.replace_all {
            let count = content.matches(&args.old_text).count();
            (content.replace(&args.old_text, &args.new_text), count)
        } else {
            (content.replacen(&args.old_text, &args.new_text, 1), 1)
        };

        fs::write(path, &new_content)
            .await
            .map_err(|e| ToolError::Execution(format!("Failed to write '{}': {e}", args.path)))?;

        Ok(format!("Replaced {count} occurrence(s) in '{}'", args.path))
    }
}

/// Tool for listing directory contents.
///
/// Supports recursive listing with configurable depth and optional display of
/// hidden files.
///
/// # Examples
///
/// ```rust
/// use machi::tools::ListDirTool;
///
/// let tool = ListDirTool::new().with_max_depth(3);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct ListDirTool {
    /// Default recursion depth limit. `0` means current directory only.
    max_depth: usize,
}

impl ListDirTool {
    /// Create a new list-directory tool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the default maximum recursion depth.
    #[must_use]
    pub const fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }
}

/// Arguments for [`ListDirTool`].
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct ListDirArgs {
    /// Path to the directory to list.
    pub path: String,
    /// Include hidden files (names starting with `.`). Default: `false`.
    #[serde(default)]
    pub show_hidden: bool,
    /// Recursion depth override. `0` = current directory only.
    pub depth: Option<usize>,
}

/// Internal representation of a single directory entry.
#[derive(Debug)]
#[allow(clippy::missing_docs_in_private_items)]
struct DirEntry {
    name: String,
    is_dir: bool,
    size: Option<u64>,
}

#[async_trait]
impl Tool for ListDirTool {
    const NAME: &'static str = "list_dir";
    type Args = ListDirArgs;
    type Output = String;
    type Error = ToolError;

    fn description(&self) -> String {
        "List contents of a directory. Shows files and subdirectories with sizes.".to_owned()
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list"
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files (starting with '.'). Default: false"
                },
                "depth": {
                    "type": "integer",
                    "description": "Recursion depth. 0 = current directory only. Default: 0"
                }
            },
            "required": ["path"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let path = Path::new(&args.path);

        let metadata = fs::metadata(path)
            .await
            .map_err(|e| ToolError::Execution(format!("Cannot access '{}': {e}", args.path)))?;

        if !metadata.is_dir() {
            return Err(ToolError::Execution(format!(
                "Not a directory: {}",
                args.path
            )));
        }

        let max_depth = args.depth.unwrap_or(self.max_depth);
        let entries = collect_entries(path, args.show_hidden, 0, max_depth).await?;

        if entries.is_empty() {
            return Ok("(empty directory)".to_owned());
        }

        let mut output = String::new();
        for entry in &entries {
            let kind = if entry.is_dir { "[DIR]" } else { "[FILE]" };
            let size_str = entry
                .size
                .map(|s| format!(" ({s} bytes)"))
                .unwrap_or_default();
            let _ = writeln!(output, "{kind} {}{size_str}", entry.name);
        }

        // Trim trailing newline
        if output.ends_with('\n') {
            output.pop();
        }

        Ok(output)
    }
}

/// Recursively collect directory entries up to `max_depth`.
///
/// Uses `Box::pin` for recursive async to satisfy the compiler.
fn collect_entries(
    dir: &Path,
    show_hidden: bool,
    current_depth: usize,
    max_depth: usize,
) -> Pin<Box<dyn Future<Output = Result<Vec<DirEntry>, ToolError>> + Send + '_>> {
    Box::pin(async move {
        let mut entries = Vec::new();
        let mut read_dir = fs::read_dir(dir)
            .await
            .map_err(|e| ToolError::Execution(format!("Failed to read directory: {e}")))?;

        while let Some(entry) = read_dir
            .next_entry()
            .await
            .map_err(|e| ToolError::Execution(format!("Failed to read entry: {e}")))?
        {
            let file_name = entry.file_name().to_string_lossy().into_owned();

            // Skip hidden entries unless requested
            if !show_hidden && file_name.starts_with('.') {
                continue;
            }

            let meta = entry
                .metadata()
                .await
                .map_err(|e| ToolError::Execution(format!("Failed to read metadata: {e}")))?;

            let is_dir = meta.is_dir();
            let size = if is_dir { None } else { Some(meta.len()) };
            let indent = "  ".repeat(current_depth);

            entries.push(DirEntry {
                name: format!("{indent}{file_name}"),
                is_dir,
                size,
            });

            // Recurse into subdirectories
            if is_dir && current_depth < max_depth {
                let sub = collect_entries(&entry.path(), show_hidden, current_depth + 1, max_depth)
                    .await?;
                entries.extend(sub);
            }
        }

        // Sort: directories first, then alphabetically
        entries.sort_by(|a, b| match (a.is_dir, b.is_dir) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.name.cmp(&b.name),
        });

        Ok(entries)
    })
}
