//! Agent run result example with execution summary.
//!
//! ```bash
//! ollama pull qwen3
//! cargo run --example agent_run_result
//! ```

#![allow(clippy::print_stdout, clippy::print_stderr, clippy::unused_async)]

use machi::prelude::*;

/// Divides two numbers.
#[machi::tool]
async fn divide(a: i64, b: i64) -> std::result::Result<i64, ToolError> {
    if b == 0 {
        return Err(ToolError::ExecutionError("Division by zero".into()));
    }
    Ok(a / b)
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let model = OllamaClient::new().completion_model("qwen3");
    let mut agent = Agent::builder()
        .model(model)
        .tool(Box::new(Divide))
        .max_steps(3)
        .final_answer_checks(FinalAnswerChecks::new().not_null().not_empty())
        .build();

    let result = agent.execute("What is 100 divided by 4?").await;

    println!("{}", result.summary());

    if result.is_success() {
        println!("Output: {:?}", result.output);
    } else {
        println!("State: {}", result.state);
    }

    Ok(())
}
