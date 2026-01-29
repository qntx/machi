//! Ollama client and provider configuration

use crate::client::{
    self, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder, ProviderClient,
};
use crate::http_client;

use super::completion::CompletionModel;
use super::embedding::EmbeddingModel;

const OLLAMA_API_BASE_URL: &str = "http://localhost:11434";

#[derive(Debug, Default, Clone, Copy)]
pub struct OllamaExt;

#[derive(Debug, Default, Clone, Copy)]
pub struct OllamaBuilder;

impl Provider for OllamaExt {
    type Builder = OllamaBuilder;

    const VERIFY_PATH: &'static str = "api/tags";

    fn build<H>(
        _: &crate::client::ClientBuilder<
            Self::Builder,
            <Self::Builder as crate::client::ProviderBuilder>::ApiKey,
            H,
        >,
    ) -> http_client::Result<Self> {
        Ok(Self)
    }
}

impl<H> Capabilities<H> for OllamaExt {
    type Completion = Capable<CompletionModel<H>>;
    type Transcription = Nothing;
    type Embeddings = Capable<EmbeddingModel<H>>;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;

    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for OllamaExt {}

impl ProviderBuilder for OllamaBuilder {
    type Output = OllamaExt;
    type ApiKey = Nothing;

    const BASE_URL: &'static str = OLLAMA_API_BASE_URL;
}

pub type Client<H = reqwest::Client> = client::Client<OllamaExt, H>;
pub type ClientBuilder<H = reqwest::Client> = client::ClientBuilder<OllamaBuilder, Nothing, H>;

impl ProviderClient for Client {
    type Input = Nothing;

    fn from_env() -> Self {
        let api_base = std::env::var("OLLAMA_API_BASE_URL").expect("OLLAMA_API_BASE_URL not set");

        Self::builder()
            .api_key(Nothing)
            .base_url(&api_base)
            .build()
            .unwrap()
    }

    fn from_val(_: Self::Input) -> Self {
        Self::builder().api_key(Nothing).build().unwrap()
    }
}

// ---------- API Error and Response Structures ----------

#[derive(Debug, serde::Deserialize)]
pub(super) struct ApiErrorResponse {
    pub message: String,
}

#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
pub(super) enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}
