//! Token usage ledger for sessions, prompts, and models.

use std::collections::BTreeMap;

use ovo_types::Usage;
use serde::{Deserialize, Serialize};

/// Accumulated usage with incomplete markers (fail-closed reporting).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageLedger {
    /// Main-loop / session totals.
    pub main: Usage,
    /// Nested agent fold-in totals.
    pub subagents: Usage,
    /// Per-prompt (turn) usage, indexed by prompt ordinal.
    #[serde(default)]
    pub per_prompt: Vec<Usage>,
    /// Per-model breakdown (stable map for serde).
    #[serde(default)]
    pub per_model: BTreeMap<String, Usage>,
    /// Compaction events recorded at message indices.
    #[serde(default)]
    pub compaction_at: Vec<CompactionRecord>,
    /// True when some usage could not be attributed.
    pub incomplete: bool,
}

/// One compaction bookkeeping entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompactionRecord {
    /// Message index in the session at compaction time (post-compact length).
    pub at_message_index: usize,
    /// Strategy name.
    pub strategy: String,
}

impl UsageLedger {
    /// Empty ledger.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a main-loop sample/turn usage.
    pub fn record_main(&mut self, usage: Usage) {
        self.main += usage;
    }

    /// Fold nested agent usage.
    pub fn record_subagent(&mut self, usage: Usage) {
        self.subagents += usage;
    }

    /// Record usage for the current prompt (creates the slot if needed).
    pub fn record_prompt(&mut self, prompt_index: usize, usage: Usage) {
        if prompt_index >= self.per_prompt.len() {
            self.per_prompt
                .resize(prompt_index.saturating_add(1), Usage::zero());
        }
        if let Some(slot) = self.per_prompt.get_mut(prompt_index) {
            *slot += usage;
        }
    }

    /// Record usage attributed to a model id.
    pub fn record_model(&mut self, model: impl Into<String>, usage: Usage) {
        let key = model.into();
        let entry = self.per_model.entry(key).or_insert_with(Usage::zero);
        *entry += usage;
    }

    /// Record that compaction ran when the message list had `len` messages.
    pub fn record_compaction_at(&mut self, at_message_index: usize, strategy: impl Into<String>) {
        self.compaction_at.push(CompactionRecord {
            at_message_index,
            strategy: strategy.into(),
        });
    }

    /// Mark ledger incomplete (cancel mid-flight, missing child drain).
    pub fn mark_incomplete(&mut self) {
        self.incomplete = true;
    }

    /// Combined totals.
    #[must_use]
    pub fn total(&self) -> Usage {
        self.main + self.subagents
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_and_per_model() {
        let mut l = UsageLedger::new();
        l.record_main(Usage::new(3, 2));
        l.record_subagent(Usage::new(1, 1));
        l.record_model("gpt", Usage::new(2, 0));
        l.record_prompt(0, Usage::new(1, 1));
        l.record_compaction_at(4, "max_messages");
        assert_eq!(l.total().total_tokens, 7);
        assert_eq!(l.per_model.get("gpt").map(|u| u.input_tokens), Some(2));
        assert_eq!(l.compaction_at.len(), 1);
    }
}
