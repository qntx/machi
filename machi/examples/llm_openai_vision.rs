//! Vision example using OpenAI with multimodal models.
//!
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! cargo run --example llm_openai_vision
//! ```

#![allow(clippy::print_stdout)]

use machi::message::{Content, ContentPart, Message, Role};
use machi::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAI::from_env()?;

    let user_message = Message::new(
        Role::User,
        Content::Parts(vec![
            ContentPart::text("What is in this image? Describe it concisely."),
            ContentPart::image_url("https://picsum.photos/id/237/400/300"),
        ]),
    );

    let request = ChatRequest::new("gpt-4o-mini").message(user_message);

    let response = client.chat(&request).await?;
    println!("{}", response.text().unwrap_or_default());

    Ok(())
}
