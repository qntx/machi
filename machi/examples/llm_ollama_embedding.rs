//! Embedding example using Ollama.
//!
//! ```bash
//! ollama pull nomic-embed-text
//! cargo run --example llm_ollama_embedding
//! ```

#![allow(clippy::print_stdout)]

use machi::embedding::{EmbeddingProvider, EmbeddingRequest};
use machi::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = Ollama::with_defaults()?;

    // Single text embedding
    let request = EmbeddingRequest::new("nomic-embed-text", vec!["Hello, world!".to_owned()]);

    let response = client.embed(&request).await?;
    println!("Single embedding:");
    println!("  Dimension: {}", response.embeddings[0].vector.len());
    println!(
        "  First 5 values: {:?}",
        &response.embeddings[0].vector[..5]
    );

    // Batch embeddings for similarity comparison
    let texts = vec![
        "The cat sat on the mat.".to_owned(),
        "A feline rested on the rug.".to_owned(),
        "The stock market crashed today.".to_owned(),
    ];

    let batch_request = EmbeddingRequest::new("nomic-embed-text", texts);
    let batch_response = client.embed(&batch_request).await?;

    println!(
        "\nBatch embeddings ({} texts):",
        batch_response.embeddings.len()
    );

    // Calculate cosine similarity between embeddings
    let sim_0_1 = batch_response.embeddings[0].cosine_similarity(&batch_response.embeddings[1]);
    let sim_0_2 = batch_response.embeddings[0].cosine_similarity(&batch_response.embeddings[2]);

    println!("  Similarity (cat/feline sentences): {sim_0_1:.4}");
    println!("  Similarity (cat/stock sentences):  {sim_0_2:.4}");

    Ok(())
}
