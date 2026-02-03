//! Final answer tool for concluding agent tasks.

use crate::tool::{Tool, ToolError};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Built-in tool for providing the final answer to a task.
#[derive(Debug, Clone, Copy, Default)]
pub struct FinalAnswerTool;

/// Arguments for the final answer tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalAnswerArgs {
    /// The final answer to the problem.
    pub answer: Value,
}

#[async_trait]
impl Tool for FinalAnswerTool {
    const NAME: &'static str = "final_answer";
    type Args = FinalAnswerArgs;
    type Output = Value;
    type Error = ToolError;

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn description(&self) -> String {
        "Provides the final answer to the given problem.".to_string()
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "answer": {
                    "description": "The final answer to the problem. Can be any type.",
                }
            },
            "required": ["answer"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.answer)
    }
}
