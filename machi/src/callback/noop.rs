//! No-op implementation of the [`Hooks`] trait.
//!
//! [`NoopHooks`] is a zero-sized type that relies on the trait's default
//! no-op method implementations. Used as the fallback when no hooks are
//! configured.

use async_trait::async_trait;

use super::hooks::Hooks;

/// A no-op implementation of [`Hooks`] that does nothing.
///
/// All methods are inherited from the trait defaults (empty bodies).
/// This is a zero-sized type with no runtime overhead.
///
/// # Example
///
/// ```rust
/// use machi::callback::NoopHooks;
///
/// let hooks = NoopHooks;
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopHooks;

#[async_trait]
impl Hooks for NoopHooks {}
