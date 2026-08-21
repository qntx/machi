//! Compaction range selection with tool-pair invariant.
//!
//!
//! A split index must never land inside an assistant tool-call run followed by
//! its tool results. Snapping moves the split past any orphaned tool messages.

use ovo_types::{Message, Role};

/// Plan for compacting `messages[0..split_idx]` and keeping the tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionRange {
    /// First index of the kept tail (`messages[split_idx..]`).
    pub split_idx: usize,
}

/// Whether `split_idx` is a safe boundary (not mid tool-result run).
#[must_use]
pub fn is_safe_split(messages: &[Message], split_idx: usize) -> bool {
    if split_idx >= messages.len() {
        return true;
    }
    let Some(msg) = messages.get(split_idx) else {
        return true;
    };
    // Split must not start on a Tool message (orphans results from their assistant).
    msg.role != Role::Tool
}

/// Snap `split_idx` forward until the kept prefix no longer ends mid tool-run
/// and the tail does not begin on a `Tool` role.
///
/// Returns `None` when snapping would keep nothing (entire list unsafe/consumed).
/// `split_idx == messages.len()` is always safe (drop whole list / keep only system via
/// [`apply_range`]).
#[must_use]
pub fn snap_split_forward(messages: &[Message], mut split_idx: usize) -> Option<usize> {
    let n = messages.len();
    if n == 0 {
        return None;
    }
    if split_idx == 0 {
        return None;
    }
    if split_idx > n {
        split_idx = n;
    }
    while split_idx < n {
        if is_safe_split(messages, split_idx) {
            // Safe if first kept is not Tool (orphan results without assistant).
            return Some(split_idx);
        }
        split_idx = split_idx.saturating_add(1);
    }
    // End-of-list is a safe split (compact away entire non-system body).
    Some(n)
}

/// Choose a split that keeps at most `keep_tail` messages from the end,
/// after preserving a leading system message, then snap for tool-pair safety.
///
/// Returns `None` when no compaction is needed or no safe split exists.
/// `split_idx == messages.len()` is allowed (keep system only / empty tail).
#[must_use]
pub fn select_compaction_range(messages: &[Message], keep_tail: usize) -> Option<CompactionRange> {
    if messages.is_empty() || keep_tail == 0 {
        return None;
    }
    if messages.len() <= keep_tail {
        return None;
    }

    let has_system = messages.first().is_some_and(|m| m.role == Role::System);
    let rest_len = messages.len().saturating_sub(usize::from(has_system));
    let keep_rest = keep_tail
        .saturating_sub(usize::from(has_system))
        .min(rest_len);
    // Index into full list: after system + drop oldest rest.
    let drop_rest = rest_len.saturating_sub(keep_rest);
    let split_idx = usize::from(has_system).saturating_add(drop_rest);

    if split_idx == 0 {
        return None;
    }

    let split_idx = snap_split_forward(messages, split_idx)?;
    if split_idx == 0 {
        return None;
    }
    Some(CompactionRange { split_idx })
}

/// Apply a range: drop prefix, keep tail; optionally insert a summary as the
/// first post-system message.
#[must_use]
pub fn apply_range(
    messages: Vec<Message>,
    range: CompactionRange,
    summary: Option<Message>,
) -> Vec<Message> {
    let n = messages.len();
    let split = range.split_idx.min(n);
    let (head, tail) = messages.split_at(split);
    let mut out = Vec::with_capacity(tail.len().saturating_add(2));
    if let Some(first) = head.first()
        && first.role == Role::System
    {
        out.push(first.clone());
    }
    if let Some(sum) = summary {
        out.push(sum);
    }
    out.extend(tail.iter().cloned());
    out
}

/// Invariant check for tests / fuzz: no kept Tool without a prior assistant
/// tool-call message still in the list (or system/user only prefix is ok if no tools).
#[must_use]
pub fn tool_pair_invariant_holds(messages: &[Message]) -> bool {
    let mut open_tool_calls: usize = 0;
    for m in messages {
        match m.role {
            Role::Assistant if !m.tool_calls.is_empty() => {
                open_tool_calls = open_tool_calls.saturating_add(m.tool_calls.len());
            }
            Role::Tool => {
                if open_tool_calls == 0 {
                    return false;
                }
                open_tool_calls = open_tool_calls.saturating_sub(1);
            }
            _ => {}
        }
    }
    true
}

#[cfg(test)]
#[allow(clippy::expect_used, reason = "unit tests")]
mod tests {
    use ovo_types::{ToolCall, ToolCallId};
    use serde_json::json;

    use super::*;

    fn tool_call(name: &str) -> ToolCall {
        ToolCall {
            id: ToolCallId::new("c1").expect("id"),
            name: name.into(),
            arguments: json!({}),
        }
    }

    fn assistant_tools() -> Message {
        Message::assistant_tools(vec![tool_call("x")])
    }

    fn tool_result() -> Message {
        Message::tool_result(ToolCallId::new("c1").expect("id"), "x", "ok")
    }

    #[test]
    fn snap_avoids_starting_on_tool() {
        let msgs = vec![
            Message::user("u1"),
            assistant_tools(),
            tool_result(),
            Message::user("u2"),
        ];
        // Naïve split at 2 lands on tool_result.
        assert!(!is_safe_split(&msgs, 2));
        let snapped = snap_split_forward(&msgs, 2).expect("snap");
        assert_eq!(snapped, 3);
        assert!(is_safe_split(&msgs, snapped));
    }

    #[test]
    fn select_range_preserves_invariant() {
        let mut msgs = vec![Message::system("s")];
        for i in 0..10 {
            msgs.push(Message::user(format!("u{i}")));
            msgs.push(assistant_tools());
            msgs.push(tool_result());
        }
        let range = select_compaction_range(&msgs, 6).expect("range");
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out), "{out:?}");
    }

    #[test]
    fn fuzz_random_splits_after_snap() {
        let msgs = vec![
            Message::system("s"),
            Message::user("a"),
            assistant_tools(),
            tool_result(),
            Message::user("b"),
            assistant_tools(),
            tool_result(),
            Message::user("c"),
        ];
        for split in 1..msgs.len() {
            if let Some(s) = snap_split_forward(&msgs, split) {
                let out = apply_range(msgs.clone(), CompactionRange { split_idx: s }, None);
                assert!(
                    tool_pair_invariant_holds(&out),
                    "split {split} -> {s}: {out:?}"
                );
            }
        }
    }
}

include!("select_matrix.rs");
