//! This module contains clients for the different LLM providers that Machi supports.
//!
//! Currently, the following providers are supported:
//! - OpenAI
//! - Anthropic
//! - Ollama
//! - Huggingface
//! - xAI
//!
//! Each provider has its own module, which contains a `Client` implementation that can
//! be used to initialize completion and embedding models and execute requests to those models.
//!
//! The clients also contain methods to easily create higher level AI constructs such as
//! agents and RAG systems, reducing the need for boilerplate.
pub mod anthropic;
pub mod huggingface;
pub mod ollama;
pub mod openai;
pub mod xai;


