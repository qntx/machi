//! Procedural macros for the Machi AI Agent Framework.
//!
//! This crate provides derive macros and attribute macros to simplify
//! working with the Machi framework:
//!
//! - [`Embed`] - Derive macro for implementing the `Embed` trait
//! - [`machi_tool`] - Attribute macro for converting functions into tools
//! - [`ProviderClient`] - Derive macro for provider client implementations

extern crate proc_macro;

use proc_macro::TokenStream;
use syn::{DeriveInput, ItemFn, parse_macro_input};

mod client;
mod embed;
mod tool;

/// Derive macro for implementing provider client traits.
///
/// This macro generates empty implementations for provider capability traits
/// that are not specified in the `features` attribute.
///
/// # Example
/// ```rust,ignore
/// #[derive(ProviderClient)]
/// #[client(features = ["completion", "embeddings"])]
/// struct MyProvider;
/// ```
#[proc_macro_derive(ProviderClient, attributes(client))]
pub fn derive_provider_client(input: TokenStream) -> TokenStream {
    client::provider_client(input)
}

/// Derive macro for implementing the `Embed` trait.
///
/// This macro allows you to implement the `machi::embedding::Embed` trait by deriving it.
/// Use the `#[embed]` helper attribute to mark fields that should be embedded.
///
/// # Example
/// ```rust,ignore
/// #[derive(Embed)]
/// struct Document {
///     id: String,
///     #[embed]
///     content: String,
/// }
/// ```
///
/// # Custom Embedding Functions
/// You can also use custom embedding functions with `#[embed(embed_with = "path::to::fn")]`:
///
/// ```rust,ignore
/// #[derive(Embed)]
/// struct Document {
///     #[embed(embed_with = "my_custom_embed")]
///     content: String,
/// }
/// ```
#[proc_macro_derive(Embed, attributes(embed))]
pub fn derive_embedding_trait(item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as DeriveInput);

    embed::expand(&mut input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Attribute macro that transforms a function into a `machi::tool::Tool`.
///
/// This macro generates a tool struct that implements the `Tool` trait,
/// allowing the function to be used with Machi agents.
///
/// # Arguments
///
/// - `description` - Optional description of the tool for LLM context
/// - `params(...)` - Optional parameter descriptions for each argument
/// - `required(...)` - List of required parameters
///
/// # Examples
///
/// ## Basic Usage
/// ```rust,ignore
/// use machi_derive::machi_tool;
///
/// #[machi_tool]
/// fn add(a: i32, b: i32) -> Result<i32, machi::tool::ToolError> {
///     Ok(a + b)
/// }
/// ```
///
/// ## With Description
/// ```rust,ignore
/// #[machi_tool(description = "Perform basic arithmetic operations")]
/// fn calculator(x: i32, y: i32, op: String) -> Result<i32, machi::tool::ToolError> {
///     match op.as_str() {
///         "add" => Ok(x + y),
///         "sub" => Ok(x - y),
///         _ => Err(machi::tool::ToolError::ToolCallError("Unknown op".into())),
///     }
/// }
/// ```
///
/// ## With Parameter Descriptions
/// ```rust,ignore
/// #[machi_tool(
///     description = "Process text with various operations",
///     params(
///         text = "The input text to process",
///         operation = "The operation: uppercase, lowercase, or reverse"
///     ),
///     required(text, operation)
/// )]
/// fn process_text(text: String, operation: String) -> Result<String, machi::tool::ToolError> {
///     match operation.as_str() {
///         "uppercase" => Ok(text.to_uppercase()),
///         "lowercase" => Ok(text.to_lowercase()),
///         "reverse" => Ok(text.chars().rev().collect()),
///         _ => Err(machi::tool::ToolError::ToolCallError("Unknown operation".into())),
///     }
/// }
/// ```
///
/// # Generated Code
///
/// For a function `my_tool`, this macro generates:
/// - `MyToolParameters` - A struct for deserializing arguments
/// - `MyTool` - A struct implementing `machi::tool::Tool`
/// - `MY_TOOL` - A static instance of the tool
#[proc_macro_attribute]
pub fn machi_tool(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as tool::ToolMacroArgs);
    let input_fn = parse_macro_input!(input as ItemFn);

    tool::expand_machi_tool(args, input_fn)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
