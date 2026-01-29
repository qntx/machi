//! Core types and utilities for the Machi framework.
//!
//! This module contains fundamental types and helper functions used throughout the crate:
//! - [`OneOrMany`] - A container that holds one or more items
//! - [`json_utils`] - JSON manipulation utilities
//! - [`wasm_compat`] - WASM compatibility layer

pub mod json_utils;
pub mod one_or_many;
pub mod wasm_compat;

pub use json_utils::{merge, merge_inplace, stringified_json, value_to_json_string};
pub use one_or_many::{EmptyListError, OneOrMany, string_or_one_or_many, string_or_option_one_or_many};
pub use wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSendStream, WasmCompatSync};


