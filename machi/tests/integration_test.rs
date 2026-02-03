//! Integration tests for machi framework.

#![allow(clippy::unwrap_used, clippy::panic, clippy::clone_on_ref_ptr)]

use async_trait::async_trait;
use machi::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A simple echo tool for testing.
#[derive(Debug, Clone, Copy, Default)]
struct EchoTool;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EchoArgs {
    message: String,
}

#[async_trait]
impl Tool for EchoTool {
    const NAME: &'static str = "echo";
    type Args = EchoArgs;
    type Output = String;
    type Error = ToolError;

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn description(&self) -> String {
        "Echoes back the input message.".to_string()
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to echo"
                }
            },
            "required": ["message"]
        })
    }

    async fn call(&self, args: Self::Args) -> std::result::Result<Self::Output, Self::Error> {
        Ok(args.message)
    }
}

#[test]
fn test_tool_definition() {
    let tool = EchoTool;
    let def = Tool::definition(&tool);

    assert_eq!(def.name, "echo");
    assert!(!def.description.is_empty());
    assert!(def.parameters.is_object());
}

#[test]
fn test_toolbox() {
    let mut toolbox = ToolBox::new();
    toolbox.add(EchoTool);
    toolbox.add(FinalAnswerTool);

    assert_eq!(toolbox.len(), 2);
    assert!(toolbox.contains("echo"));
    assert!(toolbox.contains("final_answer"));
    assert!(!toolbox.contains("nonexistent"));

    let defs = toolbox.definitions();
    assert_eq!(defs.len(), 2);
}

#[tokio::test]
async fn test_tool_call() {
    let toolbox = {
        let mut tb = ToolBox::new();
        tb.add(EchoTool);
        tb
    };

    let args = serde_json::json!({ "message": "Hello, World!" });
    let result = toolbox.call("echo", args).await.unwrap();

    assert_eq!(result, Value::String("Hello, World!".to_string()));
}

#[tokio::test]
async fn test_tool_not_found() {
    let toolbox = ToolBox::new();
    let result = toolbox.call("nonexistent", Value::Null).await;

    assert!(result.is_err());
    if let Err(ToolError::NotFound(name)) = result {
        assert_eq!(name, "nonexistent");
    } else {
        panic!("Expected NotFound error");
    }
}

#[test]
fn test_chat_message_creation() {
    let system_msg = ChatMessage::system("You are a helpful assistant.");
    assert_eq!(system_msg.role, MessageRole::System);
    assert!(system_msg.text_content().is_some());

    let user_msg = ChatMessage::user("Hello!");
    assert_eq!(user_msg.role, MessageRole::User);

    let assistant_msg = ChatMessage::assistant("Hi there!");
    assert_eq!(assistant_msg.role, MessageRole::Assistant);
}

#[test]
fn test_token_usage() {
    let usage1 = TokenUsage::new(100, 50);
    let usage2 = TokenUsage::new(200, 100);

    assert_eq!(usage1.total(), 150);

    let combined = usage1 + usage2;
    assert_eq!(combined.input_tokens, 300);
    assert_eq!(combined.output_tokens, 150);
    assert_eq!(combined.total(), 450);
}

#[test]
fn test_agent_memory() {
    let mut memory = AgentMemory::new("You are a test agent.");

    memory.add_step(TaskStep {
        task: "Test task".to_string(),
        task_images: None,
    });

    let messages = memory.to_messages(false);
    assert_eq!(messages.len(), 2); // System + Task

    memory.reset();
    let messages_after_reset = memory.to_messages(false);
    assert_eq!(messages_after_reset.len(), 1); // Only system
}

#[test]
fn test_callback_manager() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();

    let mut manager = CallbackManager::new();
    manager.add(move |_| {
        counter_clone.fetch_add(1, Ordering::SeqCst);
    });

    manager.emit(&StepEvent::ActionStarting { step_number: 1 });
    assert_eq!(counter.load(Ordering::SeqCst), 1);
}

#[test]
fn test_prompt_templates() {
    let prompts = PromptTemplates::tool_calling_agent();
    assert!(!prompts.system_prompt.is_empty());

    let code_prompts = PromptTemplates::code_agent();
    assert!(!code_prompts.system_prompt.is_empty());
    assert!(code_prompts.system_prompt != prompts.system_prompt);
}
