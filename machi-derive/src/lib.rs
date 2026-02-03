//! Procedural macros for the machi AI agents framework.
//!
//! This crate provides the `#[tool]` attribute macro for easily defining tools
//! that can be used by AI agents.

extern crate proc_macro;

mod tool;

use proc_macro::TokenStream;

/// A procedural macro that transforms a function into a `machi::tool::Tool` implementation.
///
/// This macro generates a struct that implements the `Tool` trait, automatically
/// extracting parameter information and generating JSON schema for tool calling.
///
/// # Examples
///
/// Basic usage:
/// ```rust,ignore
/// use machi_derive::tool;
///
/// #[tool]
/// async fn add(a: i32, b: i32) -> Result<i32, machi::tool::ToolError> {
///     Ok(a + b)
/// }
/// ```
///
/// With description and parameter documentation:
/// ```rust,ignore
/// use machi_derive::tool;
///
/// #[tool(
///     description = "Perform arithmetic operations",
///     params(
///         x = "First operand",
///         y = "Second operand",
///         operation = "Operation to perform: add, subtract, multiply, divide"
///     )
/// )]
/// async fn calculator(x: i32, y: i32, operation: String) -> Result<i32, machi::tool::ToolError> {
///     match operation.as_str() {
///         "add" => Ok(x + y),
///         "subtract" => Ok(x - y),
///         "multiply" => Ok(x * y),
///         "divide" => Ok(x / y),
///         _ => Err(machi::tool::ToolError::ExecutionError("Unknown operation".into())),
///     }
/// }
/// ```
#[proc_macro_attribute]
pub fn tool(args: TokenStream, input: TokenStream) -> TokenStream {
    tool::tool_impl(args, input)
}
