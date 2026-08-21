//! Stop gates evaluated when the model returns a non-tool final message.

use ovo_agent::{Agent, CompletionRequirement};
use ovo_types::Message;

use crate::state::ConversationState;

/// Result of evaluating stop gates after a final assistant message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateDecision {
    /// Turn may complete with the current assistant message.
    Complete,
    /// Inject a user/system reminder and force another sample.
    Continue {
        /// Reminder text appended to the conversation.
        reminder: String,
    },
    /// Requirement unmet after budgeted retries — turn must fail closed.
    Fail {
        /// Human-readable reason (surfaced as [`ovo_types::ErrorCode::RuntimeGate`]).
        reason: String,
    },
}

/// Extensible stop-gate. Gates run in order; first non-[`GateDecision::Complete`] wins.
pub trait StopGate: Send + Sync {
    /// Evaluate after a final assistant message is appended.
    fn evaluate(
        &self,
        agent: &Agent,
        state: &dyn ConversationState,
        retries_used: u32,
    ) -> GateDecision;
}

/// Require a named tool to have been called in this conversation.
#[derive(Debug, Clone)]
pub struct CompletionToolGate {
    /// Requirement from the agent definition (or override).
    pub requirement: CompletionRequirement,
}

impl StopGate for CompletionToolGate {
    fn evaluate(
        &self,
        _agent: &Agent,
        state: &dyn ConversationState,
        retries_used: u32,
    ) -> GateDecision {
        completion_gate(&self.requirement, state, retries_used).unwrap_or(GateDecision::Complete)
    }
}

/// Composite of ordered gates (first Continue wins).
#[derive(Default)]
pub struct GateChain {
    gates: Vec<Box<dyn StopGate>>,
}

impl std::fmt::Debug for GateChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GateChain")
            .field("gates", &self.gates.len())
            .finish()
    }
}

impl GateChain {
    /// Empty chain → always complete.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a gate.
    #[must_use]
    pub fn push(mut self, gate: impl StopGate + 'static) -> Self {
        self.gates.push(Box::new(gate));
        self
    }

    /// Build default chain from agent definition (completion requirement if any).
    #[must_use]
    pub fn from_agent(agent: &Agent) -> Self {
        let mut chain = Self::new();
        if let Some(req) = agent.definition().completion.clone() {
            chain = chain.push(CompletionToolGate { requirement: req });
        }
        chain
    }

    /// Evaluate all gates.
    #[must_use]
    pub fn evaluate(
        &self,
        agent: &Agent,
        state: &dyn ConversationState,
        retries_used: u32,
    ) -> GateDecision {
        for gate in &self.gates {
            match gate.evaluate(agent, state, retries_used) {
                GateDecision::Complete => {}
                other => return other,
            }
        }
        GateDecision::Complete
    }
}

/// Evaluate configured gates for the agent against conversation history.
///
/// Convenience wrapper around [`GateChain::from_agent`].
#[must_use]
pub fn evaluate_stop_gates(
    agent: &Agent,
    state: &dyn ConversationState,
    completion_retries_used: u32,
) -> GateDecision {
    GateChain::from_agent(agent).evaluate(agent, state, completion_retries_used)
}

fn completion_gate(
    req: &CompletionRequirement,
    state: &dyn ConversationState,
    retries_used: u32,
) -> Option<GateDecision> {
    if tool_was_called(state, &req.tool) {
        return None;
    }
    if retries_used >= req.max_retries {
        return Some(GateDecision::Fail {
            reason: format!(
                "required tool '{}' not called after {} reminder(s)",
                req.tool, req.max_retries
            ),
        });
    }
    Some(GateDecision::Continue {
        reminder: req.reminder.clone(),
    })
}

fn tool_was_called(state: &dyn ConversationState, tool_name: &str) -> bool {
    state
        .messages()
        .iter()
        .any(|m| message_calls_tool(m, tool_name))
}

fn message_calls_tool(message: &Message, tool_name: &str) -> bool {
    message.tool_calls.iter().any(|c| c.name == tool_name)
}

#[cfg(test)]
mod tests {
    use ovo_agent::{AgentBuilder, CompletionRequirement};
    use ovo_types::{Message, ToolCall, ToolCallId};
    use serde_json::json;

    use super::*;
    use crate::state::VecConversationState;

    #[test]
    fn requires_completion_tool() {
        let agent = AgentBuilder::named("a")
            .model("mock")
            .completion(CompletionRequirement {
                tool: "submit".into(),
                reminder: "call submit".into(),
                max_retries: 2,
            })
            .build()
            .expect("agent");
        let state = VecConversationState::from_messages(vec![Message::user("hi")]);
        let d = evaluate_stop_gates(&agent, &state, 0);
        assert_eq!(
            d,
            GateDecision::Continue {
                reminder: "call submit".into()
            }
        );
    }

    #[test]
    fn passes_when_tool_called() {
        let agent = AgentBuilder::named("a")
            .model("mock")
            .completion(CompletionRequirement {
                tool: "submit".into(),
                reminder: "call submit".into(),
                max_retries: 2,
            })
            .build()
            .expect("agent");
        let id = ToolCallId::new("t1").expect("id");
        let mut state = VecConversationState::new();
        state.append(Message::assistant_tools(vec![ToolCall {
            id,
            name: "submit".into(),
            arguments: json!({}),
        }]));
        let d = evaluate_stop_gates(&agent, &state, 0);
        assert_eq!(d, GateDecision::Complete);
    }

    #[test]
    fn chain_from_agent() {
        let agent = AgentBuilder::named("a")
            .model("mock")
            .completion(CompletionRequirement {
                tool: "done".into(),
                reminder: "use done".into(),
                max_retries: 1,
            })
            .build()
            .expect("agent");
        let chain = GateChain::from_agent(&agent);
        let state = VecConversationState::new();
        assert!(matches!(
            chain.evaluate(&agent, &state, 0),
            GateDecision::Continue { .. }
        ));
    }

    #[test]
    fn max_retries_then_fail_without_tool() {
        let agent = AgentBuilder::named("a")
            .model("mock")
            .completion(CompletionRequirement {
                tool: "submit".into(),
                reminder: "call submit".into(),
                max_retries: 2,
            })
            .build()
            .expect("agent");
        let state = VecConversationState::from_messages(vec![Message::user("hi")]);
        assert!(matches!(
            evaluate_stop_gates(&agent, &state, 0),
            GateDecision::Continue { .. }
        ));
        assert!(matches!(
            evaluate_stop_gates(&agent, &state, 1),
            GateDecision::Continue { .. }
        ));
        // Exhausted: fail-closed (required tool never called).
        assert!(
            matches!(
                evaluate_stop_gates(&agent, &state, 2),
                GateDecision::Fail { ref reason } if reason.contains("submit")
            ),
            "{:?}",
            evaluate_stop_gates(&agent, &state, 2)
        );
    }

    #[test]
    fn custom_gate_in_chain_first_continue_wins() {
        struct AlwaysContinue;
        impl StopGate for AlwaysContinue {
            fn evaluate(
                &self,
                _agent: &Agent,
                _state: &dyn ConversationState,
                _retries_used: u32,
            ) -> GateDecision {
                GateDecision::Continue {
                    reminder: "again".into(),
                }
            }
        }

        let agent = AgentBuilder::named("a").model("mock").build().expect("a");
        let chain = GateChain::new().push(AlwaysContinue);
        let state = VecConversationState::new();
        assert_eq!(
            chain.evaluate(&agent, &state, 0),
            GateDecision::Continue {
                reminder: "again".into()
            }
        );
    }
}
