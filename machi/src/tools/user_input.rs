//! User input tool for interactive agents.

#![allow(clippy::print_stdout, clippy::exhaustive_structs)]

use crate::tool::{Tool, ToolError};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{self, Write};

/// Tool that asks for user input on a specific question.
#[derive(Debug, Clone, Copy, Default)]
pub struct UserInputTool;

/// Arguments for the user input tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInputArgs {
    /// The question to ask the user.
    pub question: String,
}

#[async_trait]
impl Tool for UserInputTool {
    const NAME: &'static str = "user_input";
    type Args = UserInputArgs;
    type Output = String;
    type Error = ToolError;

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn description(&self) -> String {
        "Asks for user's input on a specific question.".to_string()
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user"
                }
            },
            "required": ["question"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        print!("{} => ", args.question);
        io::stdout()
            .flush()
            .map_err(|e| ToolError::ExecutionError(e.to_string()))?;

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .map_err(|e| ToolError::ExecutionError(e.to_string()))?;

        Ok(input.trim().to_string())
    }
}
