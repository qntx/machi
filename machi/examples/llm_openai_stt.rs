//! Speech-to-Text example using OpenAI.
//!
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! cargo run --example llm_openai_stt
//! ```

#![allow(clippy::print_stdout)]

use machi::audio::{AudioFormat, SpeechToTextProvider, TranscriptionRequest};
use machi::prelude::*;
use std::fs;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAI::from_env()?;

    // Read audio file
    let audio_data = fs::read("machi/examples/data/en-us-natural-speech.mp3")?;

    // Transcribe
    let request = TranscriptionRequest::new("whisper-1", audio_data).format(AudioFormat::Mp3);
    let response = client.transcribe(&request).await?;

    println!("\nTranscription:");
    println!("{}", response.text);

    if let Some(lang) = &response.language {
        println!("\nDetected language: {lang}");
    }
    if let Some(duration) = response.duration {
        println!("Duration: {duration:.2}s");
    }

    Ok(())
}
