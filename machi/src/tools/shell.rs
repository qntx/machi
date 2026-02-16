//! Shell command execution tool for agents.
//!
//! Provides the ability to execute shell commands with configurable timeout,
//! working directory, and output-size limits. Uses platform-appropriate shells
//! (`sh -c` on Unix, `cmd /C` on Windows).

use std::fmt;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::process::Command;
use tokio::time::timeout;

use crate::error::ToolError;
use crate::tool::Tool;

/// Tool for executing shell commands.
///
/// # Safety considerations
///
/// This tool runs arbitrary commands on the host system. Pair it with
/// [`ToolExecutionPolicy::RequireConfirmation`](crate::tool::ToolExecutionPolicy)
/// to enable human-in-the-loop approval before execution.
///
/// # Examples
///
/// ```rust
/// use machi::tools::ExecTool;
///
/// let tool = ExecTool::new()
///     .with_working_dir("/tmp")
///     .with_timeout(30)
///     .with_max_output(64 * 1024);
/// ```
#[derive(Debug, Clone)]
pub struct ExecTool {
    /// Default working directory (used when the caller does not specify one).
    working_dir: Option<String>,
    /// Command timeout in seconds. Default: 60.
    timeout_secs: u64,
    /// Maximum captured output size in bytes. Default: 100 `KiB`.
    max_output_size: usize,
}

impl Default for ExecTool {
    fn default() -> Self {
        Self {
            working_dir: None,
            timeout_secs: 60,
            max_output_size: 100 * 1024, // 100 KiB
        }
    }
}

impl ExecTool {
    /// Create a new exec tool with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the default working directory.
    #[must_use]
    pub fn with_working_dir(mut self, dir: impl Into<String>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    /// Set the command timeout in seconds.
    #[must_use]
    pub const fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Set the maximum captured output size in bytes.
    #[must_use]
    pub const fn with_max_output(mut self, size: usize) -> Self {
        self.max_output_size = size;
        self
    }
}

/// Arguments for [`ExecTool`].
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct ExecArgs {
    /// The shell command to execute.
    pub command: String,
    /// Working directory for the command. Optional.
    pub cwd: Option<String>,
    /// Timeout in seconds (overrides the tool default). Optional.
    pub timeout: Option<u64>,
}

/// Structured result of a command execution.
#[derive(Debug, Clone, Serialize)]
pub struct ExecResult {
    /// Process exit code (`None` if the process was killed or never started).
    pub exit_code: Option<i32>,
    /// Captured standard output.
    pub stdout: String,
    /// Captured standard error.
    pub stderr: String,
    /// Whether the command timed out.
    pub timed_out: bool,
}

impl fmt::Display for ExecResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.timed_out {
            writeln!(f, "[TIMEOUT]")?;
        }
        if let Some(code) = self.exit_code {
            writeln!(f, "[Exit code: {code}]")?;
        }
        if !self.stdout.is_empty() {
            writeln!(f, "[stdout]\n{}", self.stdout)?;
        }
        if !self.stderr.is_empty() {
            writeln!(f, "[stderr]\n{}", self.stderr)?;
        }
        if self.stdout.is_empty() && self.stderr.is_empty() {
            write!(f, "(no output)")?;
        }
        Ok(())
    }
}

#[async_trait]
impl Tool for ExecTool {
    const NAME: &'static str = "exec";
    type Args = ExecArgs;
    type Output = String;
    type Error = ToolError;

    fn description(&self) -> String {
        "Execute a shell command and return its output. Supports timeout and working directory."
            .to_owned()
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for the command. Optional."
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds. Optional (default: 60)."
                }
            },
            "required": ["command"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let timeout_secs = args.timeout.unwrap_or(self.timeout_secs);
        let effective_cwd = args.cwd.as_ref().or(self.working_dir.as_ref());

        let mut cmd = build_platform_command(&args.command);

        if let Some(dir) = effective_cwd {
            cmd.current_dir(dir);
        }

        // Capture both stdout and stderr
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        // `kill_on_drop` ensures the child is cleaned up if this future is
        // cancelled or if we drop early on timeout.
        cmd.kill_on_drop(true);

        let mut child = cmd
            .spawn()
            .map_err(|e| ToolError::execution(format!("Failed to spawn command: {e}")))?;

        // Take the stdio handles *before* waiting so ownership stays with us.
        let child_stdout = child.stdout.take();
        let child_stderr = child.stderr.take();

        // Run with timeout
        let result = timeout(Duration::from_secs(timeout_secs), async {
            let status = child
                .wait()
                .await
                .map_err(|e| ToolError::execution(format!("Command execution failed: {e}")))?;

            let mut stdout_buf = Vec::new();
            let mut stderr_buf = Vec::new();

            if let Some(mut out) = child_stdout {
                tokio::io::AsyncReadExt::read_to_end(&mut out, &mut stdout_buf)
                    .await
                    .ok();
            }
            if let Some(mut err) = child_stderr {
                tokio::io::AsyncReadExt::read_to_end(&mut err, &mut stderr_buf)
                    .await
                    .ok();
            }

            Ok::<_, ToolError>((status, stdout_buf, stderr_buf))
        })
        .await;

        let max = self.max_output_size;

        match result {
            Ok(Ok((status, stdout_buf, stderr_buf))) => {
                let stdout = truncate_output(&String::from_utf8_lossy(&stdout_buf), max);
                let stderr = truncate_output(&String::from_utf8_lossy(&stderr_buf), max);

                Ok(ExecResult {
                    exit_code: status.code(),
                    stdout,
                    stderr,
                    timed_out: false,
                }
                .to_string())
            }
            Ok(Err(e)) => Err(e),
            Err(_) => {
                // Timeout — `kill_on_drop` will clean up the child.
                Ok(ExecResult {
                    exit_code: None,
                    stdout: String::new(),
                    stderr: format!("Command timed out after {timeout_secs} seconds"),
                    timed_out: true,
                }
                .to_string())
            }
        }
    }
}

/// Build a platform-appropriate `Command` that executes `cmd_str` via the
/// system shell.
fn build_platform_command(cmd_str: &str) -> Command {
    #[cfg(target_family = "windows")]
    {
        let mut cmd = Command::new("cmd");
        cmd.args(["/C", cmd_str]);
        cmd
    }

    #[cfg(not(target_family = "windows"))]
    {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", cmd_str]);
        cmd
    }
}

/// Truncate `output` to at most `max_bytes`, appending an indicator if
/// truncation occurred.
fn truncate_output(output: &str, max_bytes: usize) -> String {
    if output.len() <= max_bytes {
        return output.to_owned();
    }

    // Find a valid UTF-8 boundary at or before `max_bytes`.
    let boundary = output
        .char_indices()
        .map(|(i, _)| i)
        .take_while(|&i| i <= max_bytes)
        .last()
        .unwrap_or(0);

    format!(
        "{}\n... [truncated, {} bytes total]",
        &output[..boundary],
        output.len()
    )
}
