//! Core client traits for provider abstraction.
//!
//! This module defines the fundamental traits for provider clients:
//! - [`ProviderClient`] - Instantiate clients from env or values
//! - [`ApiKey`] - API key abstraction
//! - [`DebugExt`] - Debug extension for providers
//! - [`Provider`] - API provider extension
//! - [`Capability`] - Capability marker trait
//! - [`Capabilities`] - Provider capability markers
//! - [`ProviderBuilder`] - Provider-specific builder

use std::fmt::Debug;

use http::{HeaderName, HeaderValue};

use crate::http::{self, Builder};

use super::{ClientBuilder, Transport};

/// Abstracts over the ability to instantiate a client, either via environment variables or some
/// `Self::Input`
pub trait ProviderClient {
    type Input;

    /// Create a client from the process's environment.
    /// Panics if an environment is improperly configured.
    fn from_env() -> Self;

    fn from_val(input: Self::Input) -> Self;
}

/// A trait for API keys. This determines whether the key is inserted into a [Client]'s default
/// headers (in the `Some` case) or handled by a given provider extension (in the `None` case)
pub trait ApiKey: Sized {
    fn into_header(self) -> Option<http::Result<(HeaderName, HeaderValue)>> {
        None
    }
}

/// Debug extension trait for provider types.
pub trait DebugExt: Debug {
    fn fields(&self) -> impl Iterator<Item = (&'static str, &dyn Debug)> {
        std::iter::empty()
    }
}

/// An API provider extension, this abstracts over extensions which may be use in conjunction with
/// the `Client<Ext, H>` struct to define the behavior of a provider with respect to networking,
/// auth, instantiating models
pub trait Provider: Sized {
    const VERIFY_PATH: &'static str;

    type Builder: ProviderBuilder;

    fn build<H>(
        builder: &ClientBuilder<Self::Builder, <Self::Builder as ProviderBuilder>::ApiKey, H>,
    ) -> http::Result<Self>;

    fn build_uri(&self, base_url: &str, path: &str, _transport: Transport) -> String {
        // Some providers (like Azure) have a blank base URL to allow users to input their own endpoints.
        let base_url = if base_url.is_empty() {
            base_url.to_string()
        } else {
            base_url.to_string() + "/"
        };

        base_url.to_string() + path.trim_start_matches('/')
    }

    fn with_custom(&self, req: Builder) -> http::Result<Builder> {
        Ok(req)
    }
}

/// Marker trait for capability checks.
pub trait Capability {
    const CAPABLE: bool;
}

/// The capabilities of a given provider, i.e. embeddings, audio transcriptions, text completion
pub trait Capabilities<H = reqwest::Client> {
    type Completion: Capability;
    type Embeddings: Capability;
    type Transcription: Capability;
    #[cfg(feature = "image")]
    type ImageGeneration: Capability;
    #[cfg(feature = "audio")]
    type AudioGeneration: Capability;
}

/// An API provider extension *builder*, this abstracts over provider-specific builders which are
/// able to configure and produce a given provider's extension type
///
/// See [Provider]
pub trait ProviderBuilder: Sized {
    type Output: Provider;
    type ApiKey;

    const BASE_URL: &'static str;

    /// This method can be used to customize the fields of `builder` before it is used to create
    /// a client. For example, adding default headers
    fn finish<H>(
        &self,
        builder: ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http::Result<ClientBuilder<Self, Self::ApiKey, H>> {
        Ok(builder)
    }
}
