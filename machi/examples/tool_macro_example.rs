//! Example demonstrating the `#[machi_tool]` macro for creating tools.
//!
//! This example shows how to use the `machi_tool` attribute macro to easily
//! convert functions into tools that can be used with Machi agents.
//!
//! Run with: `cargo run -p machi --example tool_macro_example --features derive`

use machi::tool::Tool;
use machi::tool_macro;

// =============================================================================
// Tool Definitions using #[machi_tool] macro
// =============================================================================

/// A simple addition tool created with the macro.
///
/// The macro automatically generates:
/// - `AddParameters` struct for deserializing arguments
/// - `Add` struct implementing the `Tool` trait
/// - `ADD` static instance
#[tool_macro(
    description = "Add two numbers together",
    params(
        a = "The first number to add",
        b = "The second number to add"
    ),
    required(a, b)
)]
fn add(a: i32, b: i32) -> Result<i32, machi::tool::ToolError> {
    println!("[tool-call] Adding {} + {}", a, b);
    Ok(a + b)
}

/// A subtraction tool.
#[tool_macro(
    description = "Subtract one number from another (a - b)",
    params(
        a = "The number to subtract from",
        b = "The number to subtract"
    ),
    required(a, b)
)]
fn subtract(a: i32, b: i32) -> Result<i32, machi::tool::ToolError> {
    println!("[tool-call] Subtracting {} - {}", a, b);
    Ok(a - b)
}

/// A multiplication tool.
#[tool_macro(
    description = "Multiply two numbers together",
    params(
        a = "The first number",
        b = "The second number"
    ),
    required(a, b)
)]
fn multiply(a: i32, b: i32) -> Result<i32, machi::tool::ToolError> {
    println!("[tool-call] Multiplying {} * {}", a, b);
    Ok(a * b)
}

/// An async tool example - simulates a network request.
#[tool_macro(
    description = "Fetch a greeting message for a given name", 
    params(name = "The name to greet"),
    required(name)
)]
async fn greet(name: String) -> Result<String, machi::tool::ToolError> {
    println!("[tool-call] Generating greeting for: {}", name);
    // Simulate async work
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    Ok(format!("Hello, {}! Welcome to Machi!", name))
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize tracing for debug output
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    // Demonstrate that the macro generated proper Tool implementations
    println!("=== Tool Definitions ===\n");

    // The macro generates static instances with UPPERCASE names
    println!("Add tool name: {}", Tool::name(&ADD));
    println!("Subtract tool name: {}", Tool::name(&SUBTRACT));
    println!("Multiply tool name: {}", Tool::name(&MULTIPLY));
    println!("Greet tool name: {}", Tool::name(&GREET));

    // Show the generated tool definitions
    println!("\n=== Tool Definition JSON ===\n");
    let add_def = Tool::definition(&ADD, String::new()).await;
    println!(
        "Add definition:\n{}",
        serde_json::to_string_pretty(&add_def.parameters)?
    );

    // Direct tool calls (without agent)
    println!("\n=== Direct Tool Calls ===\n");

    let result = Tool::call(&ADD, AddParameters { a: 10, b: 20 }).await?;
    println!("10 + 20 = {}", result);

    let result = Tool::call(&SUBTRACT, SubtractParameters { a: 100, b: 42 }).await?;
    println!("100 - 42 = {}", result);

    let result = Tool::call(&MULTIPLY, MultiplyParameters { a: 7, b: 8 }).await?;
    println!("7 * 8 = {}", result);

    let greeting = Tool::call(
        &GREET,
        GreetParameters {
            name: "Rustacean".to_string(),
        },
    )
    .await?;
    println!("Greeting: {}", greeting);

    println!("\n=== Example Complete ===");
    Ok(())
}
