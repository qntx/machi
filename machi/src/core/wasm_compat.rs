//! WASM compatibility layer.
//!
//! This module provides traits and types for cross-platform compatibility
//! between native and WASM targets.

use bytes::Bytes;
use std::pin::Pin;

use futures::Stream;

#[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
pub trait WasmCompatSend: Send {}
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
pub trait WasmCompatSend {}

#[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
impl<T> WasmCompatSend for T where T: Send {}
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
impl<T> WasmCompatSend for T {}

#[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
pub trait WasmCompatSendStream:
    Stream<Item = Result<Bytes, crate::http::Error>> + Send
{
    type InnerItem: Send;
}

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
pub trait WasmCompatSendStream: Stream<Item = Result<Bytes, crate::http::Error>> {
    type InnerItem;
}

#[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
impl<T> WasmCompatSendStream for T
where
    T: Stream<Item = Result<Bytes, crate::http::Error>> + Send,
{
    type InnerItem = Result<Bytes, crate::http::Error>;
}

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
impl<T> WasmCompatSendStream for T
where
    T: Stream<Item = Result<Bytes, crate::http::Error>>,
{
    type InnerItem = Result<Bytes, crate::http::Error>;
}

#[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
pub trait WasmCompatSync: Sync {}
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
pub trait WasmCompatSync {}

#[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
impl<T> WasmCompatSync for T where T: Sync {}
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
impl<T> WasmCompatSync for T {}

#[cfg(not(target_family = "wasm"))]
pub type WasmBoxedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[cfg(target_family = "wasm")]
pub type WasmBoxedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

#[macro_export]
macro_rules! if_wasm {
    ($($tokens:tt)*) => {
        #[cfg(all(feature = "wasm", target_arch = "wasm32"))]
        $($tokens)*

    };
}

#[macro_export]
macro_rules! if_not_wasm {
    ($($tokens:tt)*) => {
        #[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
        $($tokens)*

    };
}




