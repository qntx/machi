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

    let request = EmbeddingRequest::new("nomic-embed-text", texts);
    let response = client.embed(&request).await?;

    println!("\nBatch embeddings ({} texts):", response.embeddings.len());

    // Calculate cosine similarity between embeddings
    let sim_0_1 = cosine_similarity(
        &response.embeddings[0].vector,
        &response.embeddings[1].vector,
    );
    let sim_0_2 = cosine_similarity(
        &response.embeddings[0].vector,
        &response.embeddings[2].vector,
    );

    println!("  Similarity (cat/feline sentences): {sim_0_1:.4}");
    println!("  Similarity (cat/stock sentences):  {sim_0_2:.4}");

    if let Some(tokens) = response.total_tokens {
        println!("\nTotal tokens used: {tokens}");
    }

    Ok(())
}

/// Calculate cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}
