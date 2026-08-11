//! Optional system-prompt assembly (project `AGENTS.md`, host preambles).

use std::fs;
use std::path::{Path, PathBuf};

use machi_types::{ErrorCode, MachiError};

use crate::definition::AgentDefinition;

/// Default project instruction file relative to a workspace root.
pub const PROJECT_AGENTS_MD: &str = "AGENTS.md";

/// Assembles the final system prompt for an agent definition.
pub trait PromptAssembler: Send + Sync {
    /// Produce the system prompt string for `definition`.
    ///
    /// # Errors
    ///
    /// I/O or validation failures.
    fn assemble(&self, definition: &AgentDefinition) -> Result<String, MachiError>;
}

/// Identity assembler: uses only the definition’s resolved instructions.
#[derive(Debug, Default, Clone, Copy)]
pub struct IdentityAssembler;

impl PromptAssembler for IdentityAssembler {
    fn assemble(&self, definition: &AgentDefinition) -> Result<String, MachiError> {
        Ok(definition.instructions.resolve())
    }
}

/// Prepend optional project preamble (e.g. `AGENTS.md`) to definition instructions.
#[derive(Debug, Clone)]
pub struct ProjectPromptAssembler {
    preamble: Option<String>,
    separator: String,
}

impl Default for ProjectPromptAssembler {
    fn default() -> Self {
        Self {
            preamble: None,
            separator: "\n\n".into(),
        }
    }
}

impl ProjectPromptAssembler {
    /// Empty preamble (equivalent to [`IdentityAssembler`] for assembly content).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Load `AGENTS.md` from `cwd` when present (missing file is not an error).
    ///
    /// # Errors
    ///
    /// Read failures when the file exists but cannot be read.
    pub fn from_project(cwd: impl AsRef<Path>) -> Result<Self, MachiError> {
        let path = cwd.as_ref().join(PROJECT_AGENTS_MD);
        Self::from_path(path)
    }

    /// Load preamble from an explicit path when present.
    ///
    /// # Errors
    ///
    /// Read failures when the path exists.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, MachiError> {
        let path = path.as_ref();
        if !path.is_file() {
            return Ok(Self::new());
        }
        let raw = fs::read_to_string(path).map_err(|e| {
            MachiError::new(
                ErrorCode::AgentBuild,
                format!("read {}: {e}", path.display()),
            )
        })?;
        Ok(Self::with_preamble(raw))
    }

    /// Use an explicit preamble string (trimmed; empty → none).
    #[must_use]
    pub fn with_preamble(text: impl Into<String>) -> Self {
        let t = text.into();
        let preamble = {
            let trimmed = t.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_owned())
            }
        };
        Self {
            preamble,
            separator: "\n\n".into(),
        }
    }

    /// Override separator between preamble and agent body.
    #[must_use]
    pub fn with_separator(mut self, sep: impl Into<String>) -> Self {
        self.separator = sep.into();
        self
    }

    /// Whether a non-empty preamble is configured.
    #[must_use]
    pub const fn has_preamble(&self) -> bool {
        self.preamble.is_some()
    }

    /// Preamble text when set.
    #[must_use]
    pub fn preamble(&self) -> Option<&str> {
        self.preamble.as_deref()
    }
}

impl PromptAssembler for ProjectPromptAssembler {
    fn assemble(&self, definition: &AgentDefinition) -> Result<String, MachiError> {
        let body = definition.instructions.resolve();
        match &self.preamble {
            Some(pre) if !body.is_empty() => Ok(format!("{pre}{}{body}", self.separator)),
            Some(pre) => Ok(pre.clone()),
            None => Ok(body),
        }
    }
}

/// Resolve AGENTS.md path under a project root (does not read).
#[must_use]
pub fn agents_md_path(cwd: impl AsRef<Path>) -> PathBuf {
    cwd.as_ref().join(PROJECT_AGENTS_MD)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::tempdir;

    use super::*;
    use crate::definition::Instructions;

    fn sample_def() -> AgentDefinition {
        let mut d = AgentDefinition::new("a");
        d.instructions = Instructions::Static("Be brief.".into());
        d.model = "mock".into();
        d.max_steps = 4;
        d
    }

    #[test]
    fn identity_returns_body() {
        let p = IdentityAssembler.assemble(&sample_def()).expect("ok");
        assert_eq!(p, "Be brief.");
    }

    #[test]
    fn project_preamble_prepends() {
        let asm = ProjectPromptAssembler::with_preamble("Project rules.");
        let p = asm.assemble(&sample_def()).expect("ok");
        assert!(p.starts_with("Project rules."));
        assert!(p.contains("Be brief."));
    }

    #[test]
    fn from_project_reads_agents_md() {
        let dir = tempdir().expect("tmp");
        let path = dir.path().join(PROJECT_AGENTS_MD);
        let mut f = fs::File::create(&path).expect("create");
        write!(f, "# Rules\n\nNo force-push.\n").expect("write");
        let asm = ProjectPromptAssembler::from_project(dir.path()).expect("load");
        assert!(asm.has_preamble());
        let p = asm.assemble(&sample_def()).expect("ok");
        assert!(p.contains("No force-push."));
        assert!(p.contains("Be brief."));
    }

    #[test]
    fn missing_agents_md_is_ok() {
        let dir = tempdir().expect("tmp");
        let asm = ProjectPromptAssembler::from_project(dir.path()).expect("load");
        assert!(!asm.has_preamble());
    }
}
