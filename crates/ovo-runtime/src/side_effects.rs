//! In-process workflow side effects: scratch, templates, optional git diff.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};

use ovo_workflow::HostError;

/// Shared mutable store for a single workflow run.
#[derive(Debug, Default)]
pub struct WorkflowSideEffects {
    scratch: Mutex<HashMap<String, String>>,
    templates: Mutex<HashMap<String, String>>,
    /// When set, `git_diff_since` runs `git -C <cwd> diff <commit>`.
    git_cwd: Mutex<Option<PathBuf>>,
}

impl WorkflowSideEffects {
    /// Empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Shared handle.
    #[must_use]
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::new())
    }

    /// Enable git operations rooted at `cwd` (must be a git work tree).
    pub fn set_git_cwd(&self, cwd: impl Into<PathBuf>) {
        *self
            .git_cwd
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(cwd.into());
    }

    /// Register a named template body. Placeholders use `{{key}}`.
    pub fn register_template(&self, name: impl Into<String>, body: impl Into<String>) {
        self.templates
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(name.into(), body.into());
    }

    /// Write scratch content; returns virtual path `scratch/{name}`.
    ///
    /// # Errors
    ///
    /// Empty name or oversized content.
    pub fn write_scratch(&self, name: &str, content: String) -> Result<String, HostError> {
        let name = name.trim();
        if name.is_empty() {
            return Err(HostError::Failed("scratch name must be non-empty".into()));
        }
        if name.contains("..") || name.contains('/') || name.contains('\\') {
            return Err(HostError::Failed(
                "scratch name must be a single path segment".into(),
            ));
        }
        if content.len() > 1_048_576 {
            return Err(HostError::Failed(
                "scratch content exceeds 1 MiB limit".into(),
            ));
        }
        self.scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(name.to_owned(), content);
        Ok(format!("scratch/{name}"))
    }

    /// Read scratch content.
    ///
    /// # Errors
    ///
    /// Missing file.
    pub fn read_scratch(&self, name: &str) -> Result<String, HostError> {
        let name = name.trim().trim_start_matches("scratch/");
        self.scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(name)
            .cloned()
            .ok_or_else(|| HostError::Failed(format!("scratch not found: {name}")))
    }

    /// Render a registered template with string vars from a JSON object.
    ///
    /// # Errors
    ///
    /// Missing template or non-object vars.
    pub fn render_template(
        &self,
        name: &str,
        vars: &serde_json::Value,
    ) -> Result<String, HostError> {
        let body = self
            .templates
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(name)
            .cloned()
            .ok_or_else(|| HostError::Failed(format!("template not found: {name}")))?;
        let Some(obj) = vars.as_object() else {
            return Err(HostError::Failed(
                "render_template vars must be a JSON object".into(),
            ));
        };
        let mut out = body;
        for (k, v) in obj {
            let needle = format!("{{{{{k}}}}}");
            let replacement = match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            out = out.replace(&needle, &replacement);
        }
        Ok(out)
    }

    /// Number of scratch entries (tests).
    #[must_use]
    pub fn scratch_len(&self) -> usize {
        self.scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Run `git diff <commit>` (commit → working tree) in the configured cwd.
    ///
    /// **Optional capability:** requires [`Self::set_git_cwd`]. Prefer leaving
    /// git disabled unless the host product needs repository context.
    ///
    /// # Errors
    ///
    /// Missing cwd, invalid commit string, or git failure.
    pub fn git_diff_since(&self, commit: &str) -> Result<String, HostError> {
        let commit = commit.trim();
        if commit.is_empty() {
            return Err(HostError::Failed("git commit must be non-empty".into()));
        }
        // Single argv; reject whitespace / shell metacharacters.
        if commit
            .chars()
            .any(|c| c.is_whitespace() || ";&|$`()".contains(c))
        {
            return Err(HostError::Failed(
                "git commit contains invalid characters".into(),
            ));
        }
        let cwd = self
            .git_cwd
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .ok_or_else(|| {
                HostError::Unsupported(
                    "git_diff_since requires WorkflowSideEffects::set_git_cwd".into(),
                )
            })?;

        let output = Command::new("git")
            .arg("-C")
            .arg(&cwd)
            .arg("diff")
            .arg("--no-ext-diff")
            .arg(commit)
            .output()
            .map_err(|e| HostError::Failed(format!("git diff: {e}")))?;

        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            return Err(HostError::Failed(format!(
                "git diff failed ({}): {}",
                output.status,
                err.trim()
            )));
        }
        let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
        truncate_diff(&mut text);
        Ok(text)
    }
}

const GIT_DIFF_MAX_BYTES: usize = 512 * 1024;

fn truncate_diff(text: &mut String) {
    if text.len() > GIT_DIFF_MAX_BYTES {
        text.truncate(GIT_DIFF_MAX_BYTES);
        text.push_str("\n…[truncated]");
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn scratch_round_trip() {
        let s = WorkflowSideEffects::new();
        let path = s.write_scratch("a.txt", "hello".into()).expect("write");
        assert_eq!(path, "scratch/a.txt");
        assert_eq!(s.read_scratch("a.txt").expect("read"), "hello");
        assert_eq!(s.read_scratch("scratch/a.txt").expect("read2"), "hello");
    }

    #[test]
    fn template_render() {
        let s = WorkflowSideEffects::new();
        s.register_template("greet", "hi {{name}}");
        let out = s
            .render_template("greet", &json!({"name": "ovo"}))
            .expect("render");
        assert_eq!(out, "hi ovo");
    }

    #[test]
    fn rejects_path_escape_name() {
        let s = WorkflowSideEffects::new();
        assert!(s.write_scratch("../x", "no".into()).is_err());
    }

    #[test]
    fn git_diff_requires_cwd() {
        let s = WorkflowSideEffects::new();
        let err = s.git_diff_since("HEAD").expect_err("cwd");
        assert!(matches!(err, HostError::Unsupported(_)));
    }

    #[test]
    fn git_diff_rejects_bad_commit() {
        let s = WorkflowSideEffects::new();
        s.set_git_cwd("/tmp");
        assert!(s.git_diff_since("HEAD; rm -rf /").is_err());
    }
}
