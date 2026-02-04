//! Advanced callback system for agent events.
//!
//! This module provides a type-safe, extensible callback system that allows
//! hooking into various agent lifecycle events with support for:
//!
//! - **Type-based dispatch**: Register callbacks for specific step types
//! - **Priority ordering**: Control callback execution order
//! - **Async support**: Both sync and async callbacks
//! - **Context passing**: Access agent state in callbacks
//!
//! # Example
//!
//! ```rust,ignore
//! use machi::callback::{CallbackRegistry, Priority};
//! use machi::memory::ActionStep;
//!
//! // Sync callbacks
//! let registry = CallbackRegistry::builder()
//!     .on::<ActionStep>(|step, ctx| {
//!         println!("Step {} completed", step.step_number);
//!     })
//!     .on_any(|step, ctx| {
//!         println!("Any step: {:?}", step.to_value());
//!     })
//!     .with_logging()
//!     .build();
//!
//! // Async callbacks
//! let async_registry = AsyncCallbackRegistry::builder()
//!     .on_async::<ActionStep>(|step, ctx| async move {
//!         // Async operation like sending to a channel
//!         println!("Async: Step {} completed", step.step_number);
//!     })
//!     .build();
//! ```

mod async_registry;
mod builtins;
mod context;
mod handlers;
mod registry;

pub use async_registry::{AsyncCallbackRegistry, AsyncCallbackRegistryBuilder};
pub use builtins::{
    LoggingConfig, MetricsCollector, MetricsSnapshot, RunMetrics, logging_handler, metrics_handler,
    tracing_handler,
};
pub use context::{CallbackContext, CallbackContextRef};
pub use handlers::{BoxedCallback, CallbackFn, Priority};
pub use registry::{CallbackRegistry, CallbackRegistryBuilder};
