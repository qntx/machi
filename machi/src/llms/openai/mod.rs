//! OpenAI API client implementation.
//!
//! This module provides a client for the OpenAI API, supporting:
//! - Chat completions (synchronous and streaming)
//! - Text-to-Speech (TTS)
//! - Speech-to-Text (STT/Whisper)
//! - Text embeddings

mod audio;
mod chat;
mod client;
mod config;
mod embedding;
mod stream;
mod types;

pub use client::OpenAI;
pub use config::OpenAIConfig;
