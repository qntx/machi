//! Core HTTP client traits.
//!
//! This module defines the fundamental traits for HTTP clients:
//! - [`HttpClientExt`] - HTTP client extension trait for sending requests
//! - [`RetryPolicy`] - Retry policy trait for handling connection delays

use std::time::Duration;

use bytes::Bytes;

use super::{Error, LazyBody, MultipartForm, Result, StreamingResponse};
use crate::core::wasm_compat::*;
pub use http::Request;

/// A helper trait to make generic requests (both regular and SSE) possible.
pub trait HttpClientExt: WasmCompatSend + WasmCompatSync {
    /// Send a HTTP request, get a response back (as bytes). Response must be able to be turned back into Bytes.
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<http::Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes>,
        T: WasmCompatSend,
        U: From<Bytes>,
        U: WasmCompatSend + 'static;

    /// Send a HTTP request with a multipart body, get a response back (as bytes). Response must be able to be turned back into Bytes (although usually for the response, you will probably want to specify Bytes anyway).
    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = Result<http::Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes>,
        U: WasmCompatSend + 'static;

    /// Send a HTTP request, get a streamed response back (as a stream of [`bytes::Bytes`].)
    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes>;
}

/// Trait for defining retry policies when receiving errors.
pub trait RetryPolicy {
    /// Submit a new retry delay based on the [`enum@Error`], last retry number and duration, if
    /// available. A policy may also return `None` if it does not want to retry
    fn retry(&self, error: &Error, last_retry: Option<(usize, Duration)>) -> Option<Duration>;

    /// Set a new reconnection time if received from an event
    fn set_reconnection_time(&mut self, duration: Duration);
}
