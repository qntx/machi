//! Tool sources: static lists and merge for multi-origin tool sets.

use std::collections::HashMap;
use std::sync::Arc;

use crate::registry::ToolRegistry;
use crate::tool::SharedTool;

/// Provides tools that can be merged into a [`ToolRegistry`].
///
/// Merge is last-wins on tool name.
pub trait ToolSource: Send + Sync {
    /// Stable source id for logs (`static`, `mcp:server`, …).
    fn name(&self) -> &str;

    /// Tools contributed by this source (order not significant after merge).
    fn tools(&self) -> Vec<SharedTool>;
}

/// Fixed list of tools (primary host-registered set).
#[derive(Clone)]
pub struct StaticToolSource {
    name: String,
    tools: Vec<SharedTool>,
}

impl std::fmt::Debug for StaticToolSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StaticToolSource")
            .field("name", &self.name)
            .field("tools", &self.tools.len())
            .finish()
    }
}

impl StaticToolSource {
    /// Named static source.
    #[must_use]
    pub fn new(name: impl Into<String>, tools: Vec<SharedTool>) -> Self {
        Self {
            name: name.into(),
            tools,
        }
    }
}

impl ToolSource for StaticToolSource {
    fn name(&self) -> &str {
        &self.name
    }

    fn tools(&self) -> Vec<SharedTool> {
        self.tools.clone()
    }
}

/// Merge multiple sources into one registry.
///
/// **Last source wins** on tool name collision (deterministic, documented).
#[must_use]
pub fn merge_tool_sources<'a>(
    sources: impl IntoIterator<Item = &'a dyn ToolSource>,
) -> ToolRegistry {
    let mut map: HashMap<String, SharedTool> = HashMap::new();
    for source in sources {
        for tool in source.tools() {
            map.insert(tool.name().to_owned(), tool);
        }
    }
    ToolRegistry::from_tools(map.into_values().collect())
}

/// Arc-wrapped dynamic source list helper.
#[must_use]
pub fn merge_arc_sources(sources: &[Arc<dyn ToolSource>]) -> ToolRegistry {
    let refs: Vec<&dyn ToolSource> = sources.iter().map(AsRef::as_ref).collect();
    merge_tool_sources(refs)
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use serde_json::{Value, json};

    use super::*;
    use crate::calc::CalcTool;
    use crate::tool::DynTool;

    struct NamedTool {
        n: &'static str,
    }

    #[async_trait]
    impl DynTool for NamedTool {
        fn name(&self) -> &str {
            self.n
        }
        fn description(&self) -> &str {
            "t"
        }
        fn parameters(&self) -> Value {
            json!({"type":"object","properties":{}})
        }
        async fn call(
            &self,
            _ctx: crate::context::ToolCallContext,
            _arguments: Value,
        ) -> Result<crate::tool::ToolResult, crate::error::ToolError> {
            Ok(crate::tool::ToolResult::text(self.n))
        }
    }

    #[test]
    fn last_source_wins_on_name() {
        let a = StaticToolSource::new("a", vec![Arc::new(NamedTool { n: "dup" })]);
        let b = StaticToolSource::new(
            "b",
            vec![Arc::new(NamedTool { n: "dup" }), Arc::new(CalcTool)],
        );
        let sources: [&dyn ToolSource; 2] = [&a, &b];
        let reg = merge_tool_sources(sources);
        assert_eq!(reg.len(), 2);
        assert_eq!(reg.get("dup").expect("dup").name(), "dup");
        assert!(reg.get(CalcTool.name()).is_some());
        assert_eq!(reg.names().len(), 2);
    }

    #[test]
    fn empty_merge() {
        let reg = merge_tool_sources([]);
        assert!(reg.is_empty());
    }
}
