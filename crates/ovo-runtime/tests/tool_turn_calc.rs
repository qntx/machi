//! Full turn: model `tool_call` → `CalcTool` → final answer via shipped runtime.
#![allow(
    unused_crate_dependencies,
    reason = "integration binary links runtime deps"
)]

#[cfg(test)]
mod calc_turn {
    #![allow(
        clippy::expect_used,
        clippy::unwrap_used,
        clippy::indexing_slicing,
        reason = "integration test harness"
    )]

    use std::sync::Arc;

    use ovo_agent::AgentBuilder;
    use ovo_llm::MockSampler;
    use ovo_runtime::{
        ConversationState, TurnInput, TurnOptions, TurnRuntime, VecConversationState,
    };
    use ovo_tools::CalcTool;
    use ovo_types::{Message, ToolCall, ToolCallId};
    use serde_json::json;

    #[tokio::test]
    async fn calc_tool_round_trip_through_turn_runtime() {
        let sampler = Arc::new(MockSampler::new());
        let call_id = ToolCallId::new("call_calc_1").expect("id");
        sampler.push_tools(Message::assistant_tools(vec![ToolCall {
            id: call_id.clone(),
            name: "calc".into(),
            arguments: json!({"expr": "(2+3)*4"}),
        }]));
        sampler.push_text("The result is 20.");

        let agent = AgentBuilder::named("math")
            .instructions("Use calc for arithmetic.")
            .model("mock")
            .tools(vec![Arc::new(CalcTool)])
            .build()
            .expect("agent");

        let mut state = VecConversationState::new();
        let outcome = TurnRuntime::new()
            .run(
                &agent,
                sampler.as_ref(),
                &mut state,
                TurnInput::Text("What is (2+3)*4?".into()),
                TurnOptions::default(),
            )
            .await
            .expect("turn");

        assert_eq!(outcome.output_text, "The result is 20.");
        assert_eq!(outcome.steps, 2);

        let tool_msg = state
            .messages()
            .iter()
            .find(|m| m.role == ovo_types::Role::Tool)
            .expect("tool message in history");
        assert_eq!(tool_msg.content.as_deref(), Some("20"));
        assert_eq!(
            tool_msg.tool_call_id.as_ref().map(ToolCallId::as_str),
            Some(call_id.as_str())
        );
        assert!(
            state
                .messages()
                .iter()
                .any(|m| m.role == ovo_types::Role::Assistant
                    && m.tool_calls.iter().any(|t| t.name == "calc")),
            "assistant tool_calls must be retained in history"
        );
    }
}
