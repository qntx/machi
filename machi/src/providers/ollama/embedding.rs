//! Ollama embedding model implementation

use crate::embeddings::{self, EmbeddingError};
use crate::http_client::{self, HttpClientExt};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::client::Client;

// ---------- Embedding Constants ----------

pub const ALL_MINILM: &str = "all-minilm";
pub const NOMIC_EMBED_TEXT: &str = "nomic-embed-text";

// ---------- Embedding Response ----------

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub model: String,
    pub embeddings: Vec<Vec<f64>>,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
}

impl From<super::client::ApiErrorResponse> for EmbeddingError {
    fn from(err: super::client::ApiErrorResponse) -> Self {
        EmbeddingError::ProviderError(err.message)
    }
}

impl From<super::client::ApiResponse<EmbeddingResponse>> for Result<EmbeddingResponse, EmbeddingError> {
    fn from(value: super::client::ApiResponse<EmbeddingResponse>) -> Self {
        match value {
            super::client::ApiResponse::Ok(response) => Ok(response),
            super::client::ApiResponse::Err(err) => Err(EmbeddingError::ProviderError(err.message)),
        }
    }
}

// ---------- Embedding Model ----------

#[derive(Clone)]
pub struct EmbeddingModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
    ndims: usize,
}

impl<T> EmbeddingModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>, ndims: usize) -> Self {
        Self {
            client,
            model: model.into(),
            ndims,
        }
    }

    pub fn with_model(client: Client<T>, model: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.into(),
            ndims,
        }
    }
}

impl<T> embeddings::EmbeddingModel for EmbeddingModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>, dims: Option<usize>) -> Self {
        Self::new(client.clone(), model, dims.unwrap())
    }

    const MAX_DOCUMENTS: usize = 1024;

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let docs: Vec<String> = documents.into_iter().collect();

        let body = serde_json::to_vec(&json!({
            "model": self.model,
            "input": docs
        }))?;

        let req = self
            .client
            .post("api/embed")?
            .body(body)
            .map_err(|e| EmbeddingError::HttpError(e.into()))?;

        let response = self.client.send(req).await?;

        if !response.status().is_success() {
            let text = http_client::text(response).await?;
            return Err(EmbeddingError::ProviderError(text));
        }

        let bytes: Vec<u8> = response.into_body().await?;

        let api_resp: EmbeddingResponse = serde_json::from_slice(&bytes)?;

        if api_resp.embeddings.len() != docs.len() {
            return Err(EmbeddingError::ResponseError(
                "Number of returned embeddings does not match input".into(),
            ));
        }
        Ok(api_resp
            .embeddings
            .into_iter()
            .zip(docs.into_iter())
            .map(|(vec, document)| embeddings::Embedding { document, vec })
            .collect())
    }
}
