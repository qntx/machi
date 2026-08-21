//! Conversation state abstractions.

use ovo_compaction::max_messages::compact_max_messages;
use ovo_protocol::{MESSAGE_FRAME_TOKENS, estimate_image_tokens, estimate_text_tokens};
use ovo_types::{ContentPart, Message};

/// Mutable conversation backing a turn or session.
pub trait ConversationState: Send {
    /// Immutable view of messages.
    fn messages(&self) -> &[Message];
    /// Append a message.
    fn append(&mut self, message: Message);
    /// Replace the entire message list (compaction / restore).
    fn replace(&mut self, messages: Vec<Message>);
    /// Token estimate for compaction triggers (aligned with preflight).
    ///
    /// Includes framing, multimodal parts, and tool-call argument JSON.
    fn token_estimate(&self) -> u64 {
        u64::from(estimate_messages_tokens(self.messages()))
    }
}

/// Shared estimator used by [`ConversationState::token_estimate`] and turn preflight.
#[must_use]
pub fn estimate_messages_tokens(messages: &[Message]) -> u32 {
    messages
        .iter()
        .fold(0u32, |acc, m| acc.saturating_add(estimate_one_message(m)))
}

fn estimate_one_message(m: &Message) -> u32 {
    let mut n = MESSAGE_FRAME_TOKENS;
    if m.parts.is_empty() {
        n = n.saturating_add(estimate_text_tokens(&m.text()));
    } else {
        for part in &m.parts {
            n = n.saturating_add(match part {
                ContentPart::Text { text } => estimate_text_tokens(text),
                ContentPart::Image { .. } => estimate_image_tokens(),
                _ => 0,
            });
        }
    }
    for call in &m.tool_calls {
        n = n.saturating_add(estimate_text_tokens(&call.name));
        n = n.saturating_add(estimate_text_tokens(&call.arguments.to_string()));
    }
    n
}

/// In-memory conversation state.
#[derive(Debug, Clone, Default)]
pub struct VecConversationState {
    messages: Vec<Message>,
}

impl VecConversationState {
    /// Empty state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Seed with messages.
    #[must_use]
    pub fn from_messages(messages: Vec<Message>) -> Self {
        Self { messages }
    }

    /// Drop oldest non-system messages until `max_messages` remains.
    ///
    /// Delegates to [`ovo_compaction::max_messages::compact_max_messages`].
    pub fn compact_max_messages(&mut self, max_messages: usize) {
        self.messages = compact_max_messages(std::mem::take(&mut self.messages), max_messages);
    }
}

impl ConversationState for VecConversationState {
    fn messages(&self) -> &[Message] {
        &self.messages
    }

    fn append(&mut self, message: Message) {
        self.messages.push(message);
    }

    fn replace(&mut self, messages: Vec<Message>) {
        self.messages = messages;
    }
}

#[cfg(test)]
mod tests {
    use ovo_types::Message;

    use super::*;

    #[test]
    fn max_messages_keeps_system_and_tail() {
        let mut state = VecConversationState::from_messages(vec![
            Message::system("sys"),
            Message::user("1"),
            Message::user("2"),
            Message::user("3"),
            Message::user("4"),
        ]);
        state.compact_max_messages(3);
        let msgs = state.messages();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs.first().map(Message::text).as_deref(), Some("sys"));
        assert_eq!(msgs.get(1).map(Message::text).as_deref(), Some("3"));
        assert_eq!(msgs.get(2).map(Message::text).as_deref(), Some("4"));
    }
}
