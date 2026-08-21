//! Cwd-jail path resolution.

use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

/// Path jail failures.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PathJailError {
    /// Path escapes the configured root after normalization / realpath.
    #[error("path escapes jail root: {0}")]
    EscapesJail(String),
    /// Empty path.
    #[error("path must be non-empty")]
    Empty,
    /// Absolute path not under jail (when absolute inputs are rejected).
    #[error("absolute path not under jail: {0}")]
    AbsoluteOutside(String),
    /// Filesystem error while resolving the jail or candidate.
    #[error("path resolution failed: {0}")]
    Io(String),
}

/// Resolve `user_path` under `jail_root`, rejecting `..` and symlink escapes.
///
/// 1. Lexical normalize and require a prefix of the jail root (cheap gate).
/// 2. When the jail root exists on disk, resolve the deepest existing ancestor
///    of the candidate via `canonicalize` and require it stay under the
///    canonical jail root (blocks in-jail symlinks that point outside).
///
/// # Errors
///
/// Returns [`PathJailError`] when the path is empty, escapes the jail, or
/// filesystem resolution fails for an existing path that must be checked.
pub fn resolve_jailed(jail_root: &Path, user_path: &str) -> Result<PathBuf, PathJailError> {
    let user_path = user_path.trim();
    if user_path.is_empty() {
        return Err(PathJailError::Empty);
    }

    let candidate = if Path::new(user_path).is_absolute() {
        PathBuf::from(user_path)
    } else {
        jail_root.join(user_path)
    };

    let normalized = normalize_lexically(&candidate);
    let root_lex = normalize_lexically(jail_root);

    if !path_under_prefix(&normalized, &root_lex) {
        return Err(PathJailError::EscapesJail(user_path.to_owned()));
    }

    // Jail root not on disk yet: lexical gate only (callers create roots).
    let Ok(root_real) = std::fs::canonicalize(jail_root) else {
        return Ok(normalized);
    };

    let (existing, rest) = deepest_existing_prefix(&normalized);
    let existing_real = if existing.as_os_str().is_empty() {
        root_real.clone()
    } else {
        std::fs::canonicalize(&existing)
            .map_err(|e| PathJailError::Io(format!("{}: {e}", existing.display())))?
    };

    if !path_under_prefix(&existing_real, &root_real) {
        return Err(PathJailError::EscapesJail(user_path.to_owned()));
    }

    let mut out = existing_real;
    for part in rest {
        out.push(part);
    }
    // Non-existing suffix is pure Normal components after normalize_lexically;
    // re-check lexical containment under the real root.
    if !path_under_prefix(&normalize_lexically(&out), &normalize_lexically(&root_real)) {
        return Err(PathJailError::EscapesJail(user_path.to_owned()));
    }
    Ok(out)
}

/// True when `path` is `prefix` or a strict descendant (component-wise).
fn path_under_prefix(path: &Path, prefix: &Path) -> bool {
    path.starts_with(prefix)
}

/// Deepest existing path prefix and the remaining components (outermost first).
fn deepest_existing_prefix(path: &Path) -> (PathBuf, Vec<OsString>) {
    let mut rest = Vec::new();
    let mut cur = path.to_path_buf();
    loop {
        if cur.as_os_str().is_empty() {
            rest.reverse();
            return (PathBuf::new(), rest);
        }
        if cur.exists() {
            rest.reverse();
            return (cur, rest);
        }
        let Some(name) = cur.file_name().map(std::ffi::OsStr::to_os_string) else {
            rest.reverse();
            return (PathBuf::new(), rest);
        };
        rest.push(name);
        match cur.parent() {
            Some(parent) if parent != cur.as_path() => cur = parent.to_path_buf(),
            _ => {
                rest.reverse();
                return (PathBuf::new(), rest);
            }
        }
    }
}

/// Lexical normalization without filesystem access (no symlink resolution).
fn normalize_lexically(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        push_component(&mut out, comp);
    }
    out
}

fn push_component(out: &mut PathBuf, comp: Component<'_>) {
    match comp {
        Component::Prefix(p) => out.push(p.as_os_str()),
        Component::RootDir => out.push(comp.as_os_str()),
        Component::CurDir => {}
        Component::ParentDir => {
            if !out.pop() && out.as_os_str().is_empty() {
                out.push("..");
            }
        }
        Component::Normal(c) => out.push(c),
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, reason = "unit tests")]
mod tests {
    use super::*;

    #[test]
    fn relative_ok() {
        let root = PathBuf::from("/workspace");
        let p = resolve_jailed(&root, "src/main.rs").expect("ok");
        assert_eq!(p, PathBuf::from("/workspace/src/main.rs"));
    }

    #[test]
    fn rejects_parent_escape() {
        let root = PathBuf::from("/workspace");
        let err = resolve_jailed(&root, "../etc/passwd").expect_err("escape");
        assert!(matches!(err, PathJailError::EscapesJail(_)));
    }

    #[test]
    fn rejects_nested_escape() {
        let root = PathBuf::from("/workspace");
        let err = resolve_jailed(&root, "a/../../outside").expect_err("escape");
        assert!(matches!(err, PathJailError::EscapesJail(_)));
    }

    #[test]
    #[cfg(unix)]
    fn rejects_symlink_escape() {
        let dir = tempfile::tempdir().expect("tmp");
        let jail = dir.path().join("jail");
        std::fs::create_dir_all(&jail).expect("jail");
        let outside = dir.path().join("secret.txt");
        std::fs::write(&outside, b"secret").expect("outside");
        let link = jail.join("link");
        std::os::unix::fs::symlink(&outside, &link).expect("symlink");

        let err = resolve_jailed(&jail, "link").expect_err("symlink out");
        assert!(matches!(err, PathJailError::EscapesJail(_)), "got {err:?}");
    }

    #[test]
    #[cfg(unix)]
    fn accepts_in_jail_real_file() {
        let dir = tempfile::tempdir().expect("tmp");
        let jail = dir.path().join("jail");
        std::fs::create_dir_all(jail.join("src")).expect("dirs");
        let file = jail.join("src/a.rs");
        std::fs::write(&file, b"fn main() {}").expect("write");

        let p = resolve_jailed(&jail, "src/a.rs").expect("ok");
        assert_eq!(p, file.canonicalize().expect("canon"));
    }
}
