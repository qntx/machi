//! Multi-level agent discovery with shadowing rules (W5.1).
//!
//! Precedence (later wins except user↛builtin):
//! 1. Builtin catalogue
//! 2. User `~/.machi/agents` — **skipped** when the name collides with a builtin
//! 3. Project `.machi/agents` walking cwd → filesystem root (nearer overrides farther)
//!
//! Disabled definitions (`enabled: false`) are omitted (invisible == not callable).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use machi_tools::CapabilityMode;
use machi_types::{ErrorCode, MachiError};

use crate::builtin::builtin_definitions;
use crate::definition::{AgentDefinition, AgentSource, Instructions, ToolPolicy};

/// Default relative directory under a project root.
pub const PROJECT_AGENTS_DIR: &str = ".machi/agents";

/// Relative directory under the user home.
pub const USER_AGENTS_DIR: &str = ".machi/agents";

/// Parse a definition file: YAML frontmatter between `---` fences, body = instructions.
///
/// Frontmatter keys: `name`, `description`, `model`, `max_steps`, `enabled`,
/// `capability`, `allowed_tools` / `tools` (comma-separated), `denied_tools`.
///
/// # Errors
///
/// Returns parse / validation errors.
pub fn parse_definition_markdown(raw: &str) -> Result<AgentDefinition, MachiError> {
    let (front, body) = split_frontmatter(raw)?;
    let meta = parse_simple_yaml_map(&front)?;
    let name = meta
        .get("name")
        .cloned()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            MachiError::new(
                ErrorCode::AgentInvalidDefinition,
                "frontmatter missing name",
            )
        })?;
    let description = meta.get("description").cloned().unwrap_or_default();
    let model = meta
        .get("model")
        .cloned()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "default".into());
    let max_steps = meta
        .get("max_steps")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(32);
    let enabled = meta
        .get("enabled")
        .map(|s| {
            !matches!(
                s.to_ascii_lowercase().as_str(),
                "false" | "0" | "no" | "off"
            )
        })
        .unwrap_or(true);
    let capability = meta
        .get("capability")
        .or_else(|| meta.get("capability_mode"))
        .and_then(|s| CapabilityMode::parse(s));
    let tools = parse_tool_policy(&meta);
    let def = AgentDefinition {
        name,
        description,
        instructions: Instructions::Static(body.trim().to_owned()),
        model,
        tools,
        output_schema: None,
        completion: None,
        max_steps,
        enabled,
        capability,
        source: None,
    };
    def.validate()?;
    Ok(def)
}

fn parse_tool_policy(meta: &BTreeMap<String, String>) -> ToolPolicy {
    if let Some(raw) = meta.get("allowed_tools").or_else(|| meta.get("tools")) {
        let list = split_csv(raw);
        if !list.is_empty() {
            return ToolPolicy::Allowlist(list);
        }
    }
    if let Some(raw) = meta.get("denied_tools") {
        let list = split_csv(raw);
        if !list.is_empty() {
            return ToolPolicy::Denylist(list);
        }
    }
    ToolPolicy::InheritAll
}

fn split_csv(raw: &str) -> Vec<String> {
    raw.split([',', ' '])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Load a single file.
///
/// # Errors
///
/// I/O or parse failures.
pub fn load_file(path: impl AsRef<Path>) -> Result<AgentDefinition, MachiError> {
    let path = path.as_ref();
    let raw = fs::read_to_string(path).map_err(|e| {
        MachiError::new(
            ErrorCode::AgentBuild,
            format!("read agent file {}: {e}", path.display()),
        )
    })?;
    parse_definition_markdown(&raw)
}

/// Discover `*.md` definitions under `root` (non-recursive).
///
/// # Errors
///
/// Directory read failures; individual bad files are skipped unless `strict`.
pub fn discover_in_dir(
    root: impl AsRef<Path>,
    strict: bool,
) -> Result<Vec<AgentDefinition>, MachiError> {
    let root = root.as_ref();
    let rd = fs::read_dir(root).map_err(|e| {
        MachiError::new(
            ErrorCode::AgentBuild,
            format!("read agents dir {}: {e}", root.display()),
        )
    })?;
    let mut out = Vec::new();
    for entry in rd {
        let entry = entry
            .map_err(|e| MachiError::new(ErrorCode::AgentBuild, format!("read_dir entry: {e}")))?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        match load_file(&path) {
            Ok(def) => out.push(def),
            Err(e) if strict => return Err(e),
            Err(_) => {}
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Discover under `{cwd}/.machi/agents` when the directory exists.
///
/// # Errors
///
/// Propagates discovery errors; missing directory yields empty list.
pub fn discover_project(cwd: impl AsRef<Path>) -> Result<Vec<AgentDefinition>, MachiError> {
    let dir = cwd.as_ref().join(PROJECT_AGENTS_DIR);
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    discover_in_dir(dir, false)
}

/// User home agents directory (`$HOME/.machi/agents`), when resolvable.
#[must_use]
pub fn user_agents_dir() -> Option<PathBuf> {
    home_dir().map(|h| h.join(USER_AGENTS_DIR))
}

/// Discover user-level agents when the directory exists.
///
/// # Errors
///
/// Directory read failures.
pub fn discover_user() -> Result<Vec<AgentDefinition>, MachiError> {
    let Some(dir) = user_agents_dir() else {
        return Ok(Vec::new());
    };
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    discover_in_dir(dir, false)
}

/// Walk `cwd` → filesystem root collecting existing `.machi/agents` dirs
/// (nearest first).
#[must_use]
pub fn project_agent_dirs(cwd: impl AsRef<Path>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut cur = Some(cwd.as_ref().to_path_buf());
    let mut seen = BTreeSet::new();
    while let Some(dir) = cur {
        let agents = dir.join(PROJECT_AGENTS_DIR);
        if agents.is_dir() {
            let canon = agents.clone();
            if seen.insert(canon.clone()) {
                out.push(canon);
            }
        }
        let parent = dir.parent().map(Path::to_path_buf);
        if parent.as_ref() == Some(&dir) {
            break;
        }
        cur = parent;
    }
    out
}

/// Full multi-level resolve with shadowing (W5.1).
///
/// # Errors
///
/// I/O failures from strict project/user discovery (non-strict soft-skips bad files).
pub fn resolve_agents(cwd: impl AsRef<Path>) -> Result<Vec<AgentDefinition>, MachiError> {
    let cwd = cwd.as_ref();
    let mut map: BTreeMap<String, AgentDefinition> = BTreeMap::new();

    // 1. Builtins
    let mut builtin_names = BTreeSet::new();
    for mut def in builtin_definitions() {
        if !def.enabled {
            continue;
        }
        builtin_names.insert(def.name.clone());
        def.source = Some(AgentSource::Builtin);
        map.insert(def.name.clone(), def);
    }

    // 2. User — skip names that collide with builtin
    for mut def in discover_user()? {
        if !def.enabled {
            continue;
        }
        if builtin_names.contains(&def.name) {
            continue;
        }
        def.source = Some(AgentSource::User);
        map.insert(def.name.clone(), def);
    }

    // 3. Project dirs: farthest → nearest so nearer overrides
    let mut dirs = project_agent_dirs(cwd);
    dirs.reverse();
    for dir in dirs {
        for mut def in discover_in_dir(&dir, false)? {
            if !def.enabled {
                continue;
            }
            def.source = Some(AgentSource::Project);
            map.insert(def.name.clone(), def);
        }
    }

    Ok(map.into_values().collect())
}

/// Find by name in a directory (file stem or frontmatter name).
///
/// # Errors
///
/// Not found or I/O.
pub fn by_name_in_dir(root: impl AsRef<Path>, name: &str) -> Result<AgentDefinition, MachiError> {
    let root = root.as_ref();
    let direct = root.join(format!("{name}.md"));
    if direct.is_file() {
        return load_file(direct);
    }
    for def in discover_in_dir(root, false)? {
        if def.name == name {
            return Ok(def);
        }
    }
    Err(MachiError::new(
        ErrorCode::AgentNotFound,
        format!("agent not found: {name}"),
    ))
}

/// Resolve `name` via multi-level discovery (cwd defaults to current dir).
///
/// # Errors
///
/// Not found after searching all layers.
pub fn by_name_resolved(name: &str, cwd: impl AsRef<Path>) -> Result<AgentDefinition, MachiError> {
    resolve_agents(cwd)?
        .into_iter()
        .find(|d| d.name == name)
        .ok_or_else(|| {
            MachiError::new(ErrorCode::AgentNotFound, format!("agent not found: {name}"))
        })
}

/// Convenience: `{cwd}/.machi/agents` then optional extra roots (legacy).
///
/// Prefer [`by_name_resolved`] for full precedence.
///
/// # Errors
///
/// Not found after searching all roots.
pub fn by_name(name: &str, roots: &[PathBuf]) -> Result<AgentDefinition, MachiError> {
    for root in roots {
        let dir = if root.ends_with(PROJECT_AGENTS_DIR) {
            root.clone()
        } else {
            root.join(PROJECT_AGENTS_DIR)
        };
        if dir.is_dir()
            && let Ok(def) = by_name_in_dir(&dir, name)
        {
            return Ok(def);
        }
    }
    Err(MachiError::new(
        ErrorCode::AgentNotFound,
        format!("agent not found: {name}"),
    ))
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

fn split_frontmatter(raw: &str) -> Result<(String, String), MachiError> {
    let text = raw.trim_start_matches('\u{feff}');
    let Some(rest) = text.strip_prefix("---") else {
        return Err(MachiError::new(
            ErrorCode::AgentInvalidDefinition,
            "agent markdown must start with --- frontmatter",
        ));
    };
    let rest = rest.strip_prefix('\n').unwrap_or(rest);
    let Some((front, body)) = rest.split_once("\n---") else {
        return Err(MachiError::new(
            ErrorCode::AgentInvalidDefinition,
            "agent markdown missing closing --- frontmatter",
        ));
    };
    let body = body.strip_prefix('\n').unwrap_or(body);
    Ok((front.to_owned(), body.to_owned()))
}

/// Minimal `key: value` YAML map (string values only; no nested objects).
fn parse_simple_yaml_map(front: &str) -> Result<BTreeMap<String, String>, MachiError> {
    let mut map = BTreeMap::new();
    for line in front.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        let key = k.trim().to_owned();
        if key.is_empty() {
            continue;
        }
        let mut val = v.trim().to_owned();
        if (val.starts_with('"') && val.ends_with('"'))
            || (val.starts_with('\'') && val.ends_with('\''))
        {
            val = val
                .get(1..val.len().saturating_sub(1))
                .unwrap_or("")
                .to_owned();
        }
        map.insert(key, val);
    }
    Ok(map)
}

#[cfg(test)]
#[allow(clippy::expect_used, reason = "unit tests")]
mod tests {
    use std::io::Write;

    use tempfile::tempdir;

    use super::*;
    use crate::builtin::{EXPLORE, GENERAL_PURPOSE};

    #[test]
    fn parses_frontmatter() {
        let raw = "---\n\
name: reviewer\n\
description: Reviews code\n\
model: mock\n\
max_steps: 8\n\
allowed_tools: calc, read_file\n\
capability: read_only\n\
---\n\
\n\
You review diffs carefully.\n";
        let def = parse_definition_markdown(raw).expect("parse");
        assert_eq!(def.name, "reviewer");
        assert_eq!(def.model, "mock");
        assert_eq!(def.max_steps, 8);
        assert!(def.instructions.resolve().contains("review diffs"));
        assert_eq!(
            def.tools,
            ToolPolicy::Allowlist(vec!["calc".into(), "read_file".into()])
        );
        assert_eq!(def.capability, Some(CapabilityMode::ReadOnly));
    }

    #[test]
    fn parses_disabled() {
        let raw = "---\nname: x\nmodel: m\nenabled: false\n---\n\nHi.\n";
        let def = parse_definition_markdown(raw).expect("parse");
        assert!(!def.enabled);
    }

    #[test]
    fn discover_dir() {
        let dir = tempdir().expect("tmp");
        let path = dir.path().join("helper.md");
        let mut f = fs::File::create(&path).expect("create");
        write!(f, "---\nname: helper\nmodel: m\n---\n\nHelp.\n").expect("write");
        let defs = discover_in_dir(dir.path(), true).expect("discover");
        assert_eq!(defs.len(), 1);
        assert_eq!(defs.first().map(|d| d.name.as_str()), Some("helper"));
    }

    #[test]
    fn resolve_project_overrides_builtin() {
        let root = tempdir().expect("tmp");
        let agents = root.path().join(PROJECT_AGENTS_DIR);
        fs::create_dir_all(&agents).expect("mkdir");
        let path = agents.join("general-purpose.md");
        let mut f = fs::File::create(&path).expect("create");
        write!(
            f,
            "---\nname: general-purpose\nmodel: custom\n---\n\nProject override.\n"
        )
        .expect("write");
        let defs = resolve_agents(root.path()).expect("resolve");
        let gp = defs.iter().find(|d| d.name == GENERAL_PURPOSE).expect("gp");
        assert_eq!(gp.model, "custom");
        assert_eq!(gp.source, Some(AgentSource::Project));
        assert!(gp.instructions.resolve().contains("Project override"));
        // explore still builtin
        let ex = defs.iter().find(|d| d.name == EXPLORE).expect("explore");
        assert_eq!(ex.source, Some(AgentSource::Builtin));
    }

    #[test]
    fn resolve_skips_disabled_project() {
        let root = tempdir().expect("tmp");
        let agents = root.path().join(PROJECT_AGENTS_DIR);
        fs::create_dir_all(&agents).expect("mkdir");
        let path = agents.join("ghost.md");
        let mut f = fs::File::create(&path).expect("create");
        write!(
            f,
            "---\nname: ghost\nmodel: m\nenabled: false\n---\n\nNope.\n"
        )
        .expect("write");
        let defs = resolve_agents(root.path()).expect("resolve");
        assert!(defs.iter().all(|d| d.name != "ghost"));
    }

    #[test]
    fn nearer_project_overrides_farther() {
        let root = tempdir().expect("tmp");
        let far = root.path().join(PROJECT_AGENTS_DIR);
        fs::create_dir_all(&far).expect("mkdir far");
        write!(
            fs::File::create(far.join("worker.md")).expect("f"),
            "---\nname: worker\nmodel: far\n---\n\nFar.\n"
        )
        .expect("w");
        let near_cwd = root.path().join("pkg");
        let near = near_cwd.join(PROJECT_AGENTS_DIR);
        fs::create_dir_all(&near).expect("mkdir near");
        write!(
            fs::File::create(near.join("worker.md")).expect("f"),
            "---\nname: worker\nmodel: near\n---\n\nNear.\n"
        )
        .expect("w");
        let defs = resolve_agents(&near_cwd).expect("resolve");
        let w = defs.iter().find(|d| d.name == "worker").expect("worker");
        assert_eq!(w.model, "near");
    }
}
