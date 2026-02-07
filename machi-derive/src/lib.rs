//! Procedural macros for the machi AI agents framework.
//!
//! This crate provides derive macros and attribute macros that simplify the process
//! of defining tools for AI agent interactions. The primary export is the [`tool`]
//! attribute macro, which transforms annotated functions into fully-featured tool
//! implementations with automatic JSON schema generation.
//!
//! # Overview
//!
//! The `machi-derive` crate is designed to eliminate boilerplate when creating tools
//! for LLM-powered agents. It follows the design philosophy of:
//!
//! - **Zero-boilerplate definitions**: Write a function, add `#[tool]`, and you're done
//! - **Automatic metadata extraction**: Doc comments become tool descriptions
//! - **Compile-time validation**: Catch errors early with helpful messages
//! - **Type-safe JSON schemas**: Rust types are automatically mapped to JSON Schema
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use machi::prelude::*;
//!
//! /// Calculate the factorial of a non-negative integer.
//! ///
//! /// # Arguments
//! ///
//! /// * `n` - The non-negative integer to calculate factorial for
//! #[tool]
//! async fn factorial(n: u64) -> ToolResult<u64> {
//!     if n <= 1 {
//!         Ok(1)
//!     } else {
//!         Ok((1..=n).product())
//!     }
//! }
//!
//! // The macro generates:
//! // - `FactorialArgs` struct with `n: u64` field
//! // - `Factorial` struct implementing `Tool` trait
//! // - `FACTORIAL` static instance for convenient access
//! ```
//!
//! # Crate Features
//!
//! This crate has no optional features. All functionality is available by default.
//!
//! # Re-exports
//!
//! This crate is typically used through the main `machi` crate, which re-exports
//! the [`tool`] macro in its prelude.

extern crate proc_macro;

mod tool;

use proc_macro::TokenStream;

/// Transforms a function into a [`machi::tool::Tool`] implementation.
///
/// This attribute macro generates all the necessary boilerplate to make a function
/// usable as a tool by AI agents. It creates argument structs, implements the `Tool`
/// trait, and generates JSON schemas for parameter validation.
///
/// # Generated Items
///
/// For a function named `my_tool`, the macro generates:
///
/// | Item | Type | Description |
/// |------|------|-------------|
/// | `MyToolArgs` | `struct` | Serializable struct containing all function parameters |
/// | `MyTool` | `struct` | Zero-sized type implementing the `Tool` trait |
/// | `MY_TOOL` | `static` | Global instance of `MyTool` for convenient access |
///
/// # Attributes
///
/// The macro accepts the following optional attributes:
///
/// ## `description`
///
/// Overrides the tool description extracted from doc comments.
///
/// ```rust,ignore
/// #[tool(description = "Custom tool description")]
/// async fn my_tool() -> ToolResult<()> { Ok(()) }
/// ```
///
/// ## `params(...)`
///
/// Provides descriptions for parameters, overriding doc comment extraction.
///
/// ```rust,ignore
/// #[tool(params(
///     x = "The x-coordinate in pixels",
///     y = "The y-coordinate in pixels"
/// ))]
/// async fn move_cursor(x: i32, y: i32) -> ToolResult<()> { Ok(()) }
/// ```
///
/// ## `required`
///
/// Forces `Option<T>` parameters to be marked as required in the JSON schema.
/// By default, `Option<T>` parameters are optional.
///
/// ```rust,ignore
/// #[tool(required = "callback_url")]
/// async fn fetch_data(url: String, callback_url: Option<String>) -> ToolResult<String> {
///     // callback_url will be marked as required despite being Option<String>
///     Ok("data".into())
/// }
/// ```
///
/// # Type Mapping
///
/// Rust types are automatically mapped to JSON Schema types:
///
/// | Rust Type | JSON Schema Type |
/// |-----------|------------------|
/// | `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, etc. | `integer` |
/// | `f32`, `f64` | `number` |
/// | `String`, `&str` | `string` |
/// | `bool` | `boolean` |
/// | `Vec<T>`, `HashSet<T>`, `BTreeSet<T>` | `array` |
/// | `HashMap<K, V>`, `BTreeMap<K, V>` | `object` |
/// | `Option<T>` | Inner type with `nullable: true` |
/// | `()` | `null` |
/// | Other types | `object` |
///
/// # Requirements
///
/// The annotated function must satisfy the following requirements:
///
/// 1. **Return type**: Must return `Result<T, E>` or `ToolResult<T>`
/// 2. **Parameter names**: Cannot use Rust keywords as parameter names
/// 3. **Async support**: Both sync and async functions are supported
///
/// # Examples
///
/// ## Basic Usage with Doc Comments
///
/// Doc comments are automatically parsed to extract descriptions:
///
/// ```rust,ignore
/// use machi::prelude::*;
///
/// /// Add two numbers together.
/// ///
/// /// This tool performs simple integer addition and returns the sum.
/// ///
/// /// # Arguments
/// ///
/// /// * `a` - The first operand
/// /// * `b` - The second operand
/// ///
/// /// # Returns
/// ///
/// /// The sum of `a` and `b`
/// #[tool]
/// async fn add(a: i32, b: i32) -> ToolResult<i32> {
///     Ok(a + b)
/// }
///
/// // Use the generated static instance
/// let result = ADD.call(AddArgs { a: 2, b: 3 }).await?;
/// assert_eq!(result, 5);
/// ```
///
/// ## Optional Parameters
///
/// `Option<T>` parameters are automatically marked as optional:
///
/// ```rust,ignore
/// use machi::prelude::*;
///
/// /// Greet a user with an optional custom message.
/// ///
/// /// # Arguments
/// ///
/// /// * `name` - The user's name
/// /// * `greeting` - Optional custom greeting (defaults to "Hello")
/// #[tool]
/// async fn greet(name: String, greeting: Option<String>) -> ToolResult<String> {
///     let greeting = greeting.unwrap_or_else(|| "Hello".to_string());
///     Ok(format!("{greeting}, {name}!"))
/// }
/// ```
///
/// ## Complex Types
///
/// The macro handles complex nested types:
///
/// ```rust,ignore
/// use machi::prelude::*;
///
/// /// Process a batch of items.
/// ///
/// /// # Arguments
/// ///
/// /// * `items` - List of items to process
/// /// * `options` - Optional processing options
/// #[tool]
/// async fn process_batch(
///     items: Vec<String>,
///     options: Option<HashMap<String, String>>,
/// ) -> ToolResult<Vec<String>> {
///     // Process items...
///     Ok(items)
/// }
/// ```
///
/// ## Synchronous Functions
///
/// Non-async functions are also supported:
///
/// ```rust,ignore
/// use machi::prelude::*;
///
/// /// Calculate the square of a number.
/// #[tool]
/// fn square(x: f64) -> ToolResult<f64> {
///     Ok(x * x)
/// }
/// ```
///
/// # Compile-Time Errors
///
/// The macro produces helpful compile-time errors for common mistakes:
///
/// - Missing return type or non-Result return type
/// - Using Rust keywords as parameter names
/// - Invalid attribute syntax
///
/// # See Also
///
/// - [`machi::tool::Tool`] - The trait implemented by generated tool structs
/// - [`machi::tool::ToolResult`] - Type alias for tool return types
/// - [`machi::tool::ToolError`] - Error type for tool execution failures
#[proc_macro_attribute]
pub fn tool(args: TokenStream, input: TokenStream) -> TokenStream {
    tool::tool_impl(args, input)
}
