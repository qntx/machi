//! Light compaction: prune tool results and strip images.

use ovo_types::{ContentPart, Message, OvoError, Role};

use crate::select::{apply_range, select_compaction_range, tool_pair_invariant_holds};
use crate::strategy::{CompactionOutcome, CompactionStrategy};

/// Replace bulky tool result bodies with a short placeholder, keeping pairing.
#[derive(Debug, Clone, Copy)]
pub struct PruneToolResults {
    /// Max characters kept from each tool result body.
    pub max_chars: usize,
}

impl Default for PruneToolResults {
    fn default() -> Self {
        Self { max_chars: 200 }
    }
}

impl CompactionStrategy for PruneToolResults {
    fn name(&self) -> &'static str {
        "prune_tool_results"
    }

    fn should_compact(&self, messages: &[Message], _token_estimate: u64) -> bool {
        messages
            .iter()
            .any(|m| m.role == Role::Tool && m.text().chars().count() > self.max_chars)
    }

    fn compact(&self, messages: Vec<Message>) -> Result<CompactionOutcome, OvoError> {
        let max = self.max_chars;
        let mut changed = false;
        let out: Vec<Message> = messages
            .into_iter()
            .map(|mut m| {
                if m.role != Role::Tool {
                    return m;
                }
                let t = m.text();
                if t.chars().count() <= max {
                    return m;
                }
                let head: String = t.chars().take(max).collect();
                m.content = Some(format!("{head}…[pruned]"));
                m.parts.clear();
                changed = true;
                m
            })
            .collect();
        debug_assert!(
            tool_pair_invariant_holds(&out),
            "prune_tool_results must preserve tool-pair invariant"
        );
        Ok(CompactionOutcome {
            messages: out,
            changed,
            strategy: self.name(),
        })
    }
}

/// Drop image parts from multimodal messages (text retained).
#[derive(Debug, Clone, Copy, Default)]
pub struct StripImages;

impl CompactionStrategy for StripImages {
    fn name(&self) -> &'static str {
        "strip_images"
    }

    fn should_compact(&self, messages: &[Message], _token_estimate: u64) -> bool {
        messages.iter().any(|m| {
            m.parts
                .iter()
                .any(|p| matches!(p, ContentPart::Image { .. }))
        })
    }

    fn compact(&self, messages: Vec<Message>) -> Result<CompactionOutcome, OvoError> {
        let mut changed = false;
        let out: Vec<Message> = messages
            .into_iter()
            .map(|mut m| {
                let before = m.parts.len();
                m.parts.retain(|p| !matches!(p, ContentPart::Image { .. }));
                if m.parts.len() != before {
                    changed = true;
                }
                m
            })
            .collect();
        Ok(CompactionOutcome {
            messages: out,
            changed,
            strategy: self.name(),
        })
    }
}

/// Drop oldest messages via [`select_compaction_range`] (tool-safe).
#[derive(Debug, Clone, Copy)]
pub struct DropPrefix {
    /// Messages to keep at the tail (including system when present).
    pub keep_tail: usize,
}

impl DropPrefix {
    /// Construct.
    #[must_use]
    pub const fn new(keep_tail: usize) -> Self {
        Self { keep_tail }
    }
}

impl CompactionStrategy for DropPrefix {
    fn name(&self) -> &'static str {
        "drop_prefix"
    }

    fn should_compact(&self, messages: &[Message], _token_estimate: u64) -> bool {
        messages.len() > self.keep_tail && self.keep_tail > 0
    }

    fn compact(&self, messages: Vec<Message>) -> Result<CompactionOutcome, OvoError> {
        let Some(range) = select_compaction_range(&messages, self.keep_tail) else {
            return Ok(CompactionOutcome {
                messages,
                changed: false,
                strategy: self.name(),
            });
        };
        let out = apply_range(messages, range, None);
        debug_assert!(
            tool_pair_invariant_holds(&out),
            "drop_prefix must preserve tool-pair invariant"
        );
        Ok(CompactionOutcome {
            messages: out,
            changed: true,
            strategy: self.name(),
        })
    }
}

/// Summarizing compaction: replace dropped prefix with a single summary message.
///
/// The `summarize` callback is synchronous so strategies stay pure-sync; hosts
/// that call an LLM inject a precomputed summary or a blocking adapter.
#[derive(Clone)]
pub struct SummarizingCompaction {
    /// Tail size to retain.
    pub keep_tail: usize,
    /// Build a summary for the compacted-away prefix (excluding system).
    pub summarize: std::sync::Arc<dyn Fn(Vec<Message>) -> Result<String, OvoError> + Send + Sync>,
}

impl std::fmt::Debug for SummarizingCompaction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SummarizingCompaction")
            .field("keep_tail", &self.keep_tail)
            .finish_non_exhaustive()
    }
}

impl CompactionStrategy for SummarizingCompaction {
    fn name(&self) -> &'static str {
        "summarizing"
    }

    fn should_compact(&self, messages: &[Message], _token_estimate: u64) -> bool {
        messages.len() > self.keep_tail && self.keep_tail > 0
    }

    fn compact(&self, messages: Vec<Message>) -> Result<CompactionOutcome, OvoError> {
        let Some(range) = select_compaction_range(&messages, self.keep_tail) else {
            return Ok(CompactionOutcome {
                messages,
                changed: false,
                strategy: self.name(),
            });
        };
        let split = range.split_idx.min(messages.len());
        let prefix: Vec<Message> = messages
            .get(..split)
            .unwrap_or(&[])
            .iter()
            .filter(|m| m.role != Role::System)
            .cloned()
            .collect();
        let summary_text = (self.summarize)(prefix)?;
        let summary = Message::user(format!("[conversation summary]\n{summary_text}"));
        let out = apply_range(messages, range, Some(summary));
        debug_assert!(
            tool_pair_invariant_holds(&out),
            "summarizing compaction must preserve tool-pair invariant"
        );
        Ok(CompactionOutcome {
            messages: out,
            changed: true,
            strategy: self.name(),
        })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, reason = "unit tests")]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn prune_shortens_tool_bodies() {
        let s = PruneToolResults { max_chars: 5 };
        let call = ovo_types::ToolCall {
            id: ovo_types::ToolCallId::new("c").expect("id"),
            name: "t".into(),
            arguments: serde_json::json!({}),
        };
        let out = s
            .compact(vec![
                Message::user("u"),
                Message::assistant_tools(vec![call]),
                Message::tool_result(
                    ovo_types::ToolCallId::new("c").expect("id"),
                    "t",
                    "1234567890",
                ),
            ])
            .expect("ok");
        assert!(out.changed, "tool body should be truncated");
        let tool_msg = out.messages.get(2).expect("tool message");
        assert!(
            tool_msg.text().contains("pruned"),
            "expected pruned marker, got {:?}",
            tool_msg.text()
        );
        assert!(
            tool_pair_invariant_holds(&out.messages),
            "tool-pair invariant after prune"
        );
    }

    #[test]
    fn summarizing_inserts_summary() {
        let s = SummarizingCompaction {
            keep_tail: 2,
            summarize: Arc::new(|msgs| Ok(format!("n={}", msgs.len()))),
        };
        let out = s
            .compact(vec![
                Message::system("s"),
                Message::user("1"),
                Message::user("2"),
                Message::user("3"),
                Message::user("4"),
            ])
            .expect("ok");
        assert!(out.changed, "summarizing should drop prefix");
        assert!(
            out.messages
                .iter()
                .any(|m| m.text().contains("conversation summary")),
            "summary message missing from {:?}",
            out.messages.iter().map(Message::text).collect::<Vec<_>>()
        );
    }
}
