//! Text-to-Speech example using OpenAI.
//!
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! cargo run --example llm_openai_tts
//! ```

#![allow(clippy::print_stdout)]

use machi::audio::{SpeechRequest, TextToSpeechProvider};
use machi::prelude::*;
use std::fs;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAI::from_env()?;

    let request = SpeechRequest::new(
        "tts-1",
        "Hello! This is a test of OpenAI's text-to-speech API.",
        "nova",
    );

    println!("\nGenerating speech...");
    let response = client.speech(&request).await?;

    let output_path = "output.mp3";
    fs::write(output_path, &response.audio)?;
    println!(
        "Audio saved to: {output_path} ({} bytes)",
        response.audio.len()
    );

    Ok(())
}
