//! Workflow metadata extraction from scripts.

use serde::{Deserialize, Serialize};

/// Maximum `meta.name` length in bytes.
pub const META_NAME_MAX: usize = 64;
/// Maximum `meta.description` length in bytes.
pub const META_DESCRIPTION_MAX: usize = 1024;
/// Maximum optional `when_to_use` length in bytes.
pub const META_WHEN_TO_USE_MAX: usize = 2048;
/// Maximum number of phases.
pub const META_PHASES_MAX: usize = 64;
/// Maximum phase title length in bytes.
pub const META_PHASE_TITLE_MAX: usize = 128;

/// Phase descriptor for UIs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseMeta {
    /// Title.
    pub title: String,
    /// Optional detail.
    #[serde(default)]
    pub detail: Option<String>,
}

/// Workflow catalog metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkflowMeta {
    /// Slug name (kebab-case, ≤64 bytes).
    pub name: String,
    /// Description (≤1024 bytes).
    pub description: String,
    /// Optional phases (≤64, unique titles ≤128).
    #[serde(default)]
    pub phases: Vec<PhaseMeta>,
    /// Optional when-to-use hint (≤2048 bytes).
    #[serde(default)]
    pub when_to_use: Option<String>,
}

/// Meta extraction errors.
#[derive(Debug, thiserror::Error)]
pub enum MetaError {
    /// Script failed to parse/run for meta probe.
    #[error("meta extract failed: {0}")]
    Failed(String),
    /// Missing required field.
    #[error("meta missing field: {0}")]
    Missing(&'static str),
    /// Field failed validation.
    #[error("meta invalid {field}: {reason}")]
    Invalid {
        /// Field path.
        field: &'static str,
        /// Reason.
        reason: String,
    },
}

/// Extract `meta` map by evaluating the script with dummy host functions noop.
///
/// Scripts **must** start with `let meta = #{ name: "...", description: "..." };`
/// as the first statement (after optional whitespace/comments).
///
/// # Errors
///
/// Returns [`MetaError`] on parse/eval/missing fields/validation.
pub fn extract_meta(script: &str) -> Result<WorkflowMeta, MetaError> {
    require_leading_meta_stmt(script)?;

    let mut engine = rhai::Engine::new();
    engine.set_max_operations(100_000);
    engine.set_max_expr_depths(128, 64);
    engine.set_max_string_size(META_WHEN_TO_USE_MAX.saturating_mul(2));
    engine.disable_symbol("eval");
    // Stub host fns so meta-only scripts that reference them later still compile.
    engine.register_fn("agent", |_p: &str| rhai::Map::new());
    engine.register_fn("phase", |_t: &str| {});
    engine.register_fn(
        "complete",
        |_v: rhai::Dynamic| -> Result<(), Box<rhai::EvalAltResult>> { Err("complete".into()) },
    );
    engine.register_fn(
        "pause",
        |_k: &str, _m: &str| -> Result<(), Box<rhai::EvalAltResult>> { Err("pause".into()) },
    );
    engine.register_fn(
        "await_user",
        |_k: &str, _m: &str| -> Result<(), Box<rhai::EvalAltResult>> { Err("await_user".into()) },
    );
    engine.register_fn("log", |_m: &str| {});
    engine.register_fn("print", |_m: &str| {});
    engine.register_fn("debug", |_m: &str| {});
    engine.register_fn("telemetry_event", |_n: &str, _f: rhai::Map| {});
    engine.register_fn("write_scratch_file", |_n: &str, _c: &str| String::new());
    engine.register_fn("read_scratch_file", |_n: &str| String::new());
    engine.register_fn("render_template", |_n: &str, _v: rhai::Dynamic| {
        String::new()
    });
    engine.register_fn("git_diff_since", |_c: &str| String::new());
    engine.register_fn("parallel", |_a: rhai::Array| rhai::Array::new());
    engine.register_fn("budget", rhai::Map::new);
    engine.register_fn("json_encode", |_v: rhai::Dynamic| String::from("null"));
    engine.register_fn("fingerprint", |_t: &str| "0".repeat(32));

    let mut scope = rhai::Scope::new();
    scope.push_dynamic("args", rhai::Dynamic::UNIT);
    // Probe evaluation may fail after `meta` is bound (e.g. complete()); ignore.
    drop(engine.eval_with_scope::<rhai::Dynamic>(&mut scope, script));

    let meta_map = scope
        .get_value::<rhai::Map>("meta")
        .ok_or(MetaError::Missing("meta"))?;
    let value = rhai::serde::from_dynamic::<serde_json::Value>(&meta_map.into())
        .map_err(|e| MetaError::Failed(e.to_string()))?;
    let meta: WorkflowMeta =
        serde_json::from_value(value).map_err(|e| MetaError::Failed(e.to_string()))?;
    validate_meta(&meta)?;
    Ok(meta)
}

fn require_leading_meta_stmt(script: &str) -> Result<(), MetaError> {
    let stripped = strip_leading_ws_and_comments(script);
    let lower = stripped.to_ascii_lowercase();
    // Accept `let meta = #{` or `let meta=#{`
    let ok = lower.starts_with("let meta")
        && stripped
            .get("let meta".len()..)
            .is_some_and(|rest| rest.trim_start().starts_with('='));
    if !ok {
        return Err(MetaError::Invalid {
            field: "script",
            reason: "first statement must be `let meta = #{ ... };`".into(),
        });
    }
    Ok(())
}

fn strip_leading_ws_and_comments(script: &str) -> &str {
    let mut s = script.trim_start();
    loop {
        if s.starts_with("//") {
            s = s.split_once('\n').map_or("", |(_, rest)| rest.trim_start());
            continue;
        }
        if s.starts_with("/*") {
            s = s.split_once("*/").map_or("", |(_, rest)| rest.trim_start());
            continue;
        }
        break;
    }
    s
}

fn validate_meta(meta: &WorkflowMeta) -> Result<(), MetaError> {
    let name = meta.name.trim();
    if name.is_empty() {
        return Err(MetaError::Missing("meta.name"));
    }
    if name.len() > META_NAME_MAX {
        return Err(MetaError::Invalid {
            field: "meta.name",
            reason: format!("exceeds {META_NAME_MAX} bytes"),
        });
    }
    if !is_kebab_case(name) {
        return Err(MetaError::Invalid {
            field: "meta.name",
            reason: "must be kebab-case ([a-z0-9]+(-[a-z0-9]+)*)".into(),
        });
    }

    let description = meta.description.trim();
    if description.is_empty() {
        return Err(MetaError::Missing("meta.description"));
    }
    if description.len() > META_DESCRIPTION_MAX {
        return Err(MetaError::Invalid {
            field: "meta.description",
            reason: format!("exceeds {META_DESCRIPTION_MAX} bytes"),
        });
    }

    if let Some(w) = &meta.when_to_use
        && w.len() > META_WHEN_TO_USE_MAX
    {
        return Err(MetaError::Invalid {
            field: "meta.when_to_use",
            reason: format!("exceeds {META_WHEN_TO_USE_MAX} bytes"),
        });
    }

    if meta.phases.len() > META_PHASES_MAX {
        return Err(MetaError::Invalid {
            field: "meta.phases",
            reason: format!("at most {META_PHASES_MAX} phases"),
        });
    }

    let mut seen = std::collections::BTreeSet::new();
    for (i, phase) in meta.phases.iter().enumerate() {
        let title = phase.title.trim();
        if title.is_empty() {
            return Err(MetaError::Invalid {
                field: "meta.phases.title",
                reason: format!("phase {i} title is empty"),
            });
        }
        if title.len() > META_PHASE_TITLE_MAX {
            return Err(MetaError::Invalid {
                field: "meta.phases.title",
                reason: format!("phase {i} title exceeds {META_PHASE_TITLE_MAX} bytes"),
            });
        }
        if !seen.insert(title.to_owned()) {
            return Err(MetaError::Invalid {
                field: "meta.phases.title",
                reason: format!("duplicate phase title `{title}`"),
            });
        }
    }
    Ok(())
}

fn is_kebab_case(s: &str) -> bool {
    if s.is_empty() || s.starts_with('-') || s.ends_with('-') {
        return false;
    }
    let mut prev_dash = false;
    for c in s.chars() {
        match c {
            'a'..='z' | '0'..='9' => prev_dash = false,
            '-' if !prev_dash => prev_dash = true,
            _ => return false,
        }
    }
    true
}

#[cfg(test)]
#[allow(clippy::expect_used, reason = "unit tests")]
mod tests {
    use super::*;

    #[test]
    fn extracts_meta() {
        let script = r#"
            let meta = #{ name: "fanout", description: "test workflow", phases: [] };
            complete(#{});
        "#;
        let meta = extract_meta(script).expect("meta");
        assert_eq!(meta.name, "fanout");
    }

    #[test]
    fn rejects_non_kebab_name() {
        let script = r#"
            let meta = #{ name: "Bad_Name", description: "x" };
            complete(1);
        "#;
        let err = extract_meta(script).expect_err("kebab");
        assert!(matches!(
            err,
            MetaError::Invalid {
                field: "meta.name",
                ..
            }
        ));
    }

    #[test]
    fn rejects_missing_leading_meta() {
        let script = r#"
            phase("x");
            let meta = #{ name: "fanout", description: "x" };
            complete(1);
        "#;
        let err = extract_meta(script).expect_err("leading");
        assert!(matches!(
            err,
            MetaError::Invalid {
                field: "script",
                ..
            }
        ));
    }

    #[test]
    fn rejects_duplicate_phase_titles() {
        let script = r#"
            let meta = #{
                name: "t",
                description: "d",
                phases: [ #{ title: "a" }, #{ title: "a" } ]
            };
            complete(1);
        "#;
        let err = extract_meta(script).expect_err("dup");
        assert!(matches!(
            err,
            MetaError::Invalid {
                field: "meta.phases.title",
                ..
            }
        ));
    }
}
