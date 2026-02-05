//! Vision example using Ollama with multimodal models.
//!
//! ```bash
//! ollama pull qwen3-vl
//! cargo run --example llm_ollama_vision
//! ```

#![allow(clippy::print_stdout)]

use machi::message::{Content, ContentPart, Message, Role};
use machi::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = Ollama::with_defaults()?;

    // URL images are automatically downloaded and converted to base64
    let user_message = Message::new(
        Role::User,
        Content::Parts(vec![
            ContentPart::text("What is in this image?"),
            ContentPart::image_url("https://picsum.photos/id/237/400/300"),
        ]),
    );

    let request = ChatRequest::new("qwen3-vl").message(user_message);

    let response = client.chat(&request).await?;
    println!("{}", response.text().unwrap_or_default());

    Ok(())
}
