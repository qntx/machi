//! Strict conversation append invariants.

use ovo_types::{Message, Role, ToolCallId};

/// Failures when appending would break conversation integrity.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StrictAppendError {
    /// Tool result without a matching open tool call.
    #[error("tool result id {0} has no matching open tool call")]
    DanglingToolResult(ToolCallId),
    /// Duplicate tool result for the same call id.
    #[error("duplicate tool result for call id {0}")]
    DuplicateToolResult(ToolCallId),
}

/// Verify tool-call / tool-result pairing across the full history.
///
/// # Errors
///
/// Returns the first pairing violation found.
pub fn check_tool_pairing(messages: &[Message]) -> Result<(), StrictAppendError> {
    let mut open: Vec<ToolCallId> = Vec::new();
    for msg in messages {
        match msg.role {
            Role::Assistant => {
                for call in &msg.tool_calls {
                    open.push(call.id.clone());
                }
            }
            Role::Tool => {
                let Some(id) = &msg.tool_call_id else {
                    continue;
                };
                if let Some(pos) = open.iter().position(|o| o == id) {
                    open.remove(pos);
                } else {
                    return Err(StrictAppendError::DanglingToolResult(id.clone()));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Whether appending `next` to `history` preserves pairing for the new message.
///
/// # Errors
///
/// Returns pairing errors for the extended history.
pub fn check_append(history: &[Message], next: &Message) -> Result<(), StrictAppendError> {
    let mut extended = history.to_vec();
    extended.push(next.clone());
    // Only enforce dangling results (duplicate results). Open calls without
    // results are allowed mid-turn.
    check_tool_pairing(&extended)?;
    // Detect duplicate results: two tool messages with same id.
    let mut seen = std::collections::HashSet::new();
    for msg in &extended {
        if msg.role == Role::Tool
            && let Some(id) = &msg.tool_call_id
            && !seen.insert(id.clone())
        {
            return Err(StrictAppendError::DuplicateToolResult(id.clone()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ovo_types::{ToolCall, ToolCallId};
    use serde_json::json;

    use super::*;

    #[test]
    fn pairs_ok() {
        let id = ToolCallId::new("c1").expect("id");
        let msgs = vec![
            Message::user("hi"),
            Message::assistant_tools(vec![ToolCall {
                id: id.clone(),
                name: "t".into(),
                arguments: json!({}),
            }]),
            Message::tool_result(id, "t", "ok"),
        ];
        check_tool_pairing(&msgs).expect("pair");
    }

    #[test]
    fn dangling_result_fails() {
        let id = ToolCallId::new("missing").expect("id");
        let msgs = vec![Message::tool_result(id, "t", "x")];
        assert!(check_tool_pairing(&msgs).is_err());
    }
}
