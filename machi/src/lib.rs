#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(tail_expr_drop_order)]
//! Machi is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity.
//!
extern crate self as machi;

// Core modules
pub mod core;
pub mod http;
pub mod client;
pub mod completion;
pub mod embedding;

// Agent and tools
pub mod agent;
pub mod tool;

// Storage and extraction
pub mod store;
pub mod extract;
pub mod loader;

// Providers
pub mod providers;

// Multi-modal
pub mod modalities;

// Integration and utilities
pub mod integration;
pub mod prelude;
pub mod telemetry;

#[cfg(feature = "experimental")]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
pub mod evals;

// Re-export commonly used types and traits
pub use completion::message;
pub use embedding::Embed;
pub use core::{OneOrMany, EmptyListError};

// Compatibility re-exports (for backward compatibility)
pub use http as http_client;
pub use embedding as embeddings;
pub use store as vector_store;
pub use extract as extractor;
pub use completion::streaming;
pub use core::json_utils;
pub use core::one_or_many;
pub use core::wasm_compat;
pub use modalities::audio::transcription;

#[cfg(feature = "derive")]
#[cfg_attr(docsrs, doc(cfg(feature = "derive")))]
pub use machi_derive::{Embed, machi_tool as tool_macro};


