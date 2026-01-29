#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(tail_expr_drop_order)]
//! Machi is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity.
//!
extern crate self as machi;

pub mod agent;
pub mod client;
pub mod completion;
pub mod core;
pub mod embedding;
pub mod extract;
pub mod http;
pub mod loader;
pub mod modalities;
pub mod prelude;
pub mod providers;
pub mod store;
pub mod telemetry;
pub mod tool;

#[cfg(feature = "audio")]
pub use modalities::audio::generation as audio_generation;
#[cfg(feature = "image")]
pub use modalities::image::ImageGenerationError;
#[cfg(feature = "image")]
pub use modalities::image::generation as image_generation;

#[cfg(feature = "derive")]
#[cfg_attr(docsrs, doc(cfg(feature = "derive")))]
pub use machi_derive::{Embed, machi_tool as tool_macro};
