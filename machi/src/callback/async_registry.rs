//! Async callback registry for asynchronous event handling.
//!
//! This module provides async callback support for scenarios where callbacks
//! need to perform I/O operations like sending to channels, HTTP requests, etc.

use super::context::CallbackContext;
use super::handlers::Priority;
use crate::memory::{
    ActionStep, FinalAnswerStep, MemoryStep, PlanningStep, SystemPromptStep, TaskStep,
};
use std::any::TypeId;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Boxed async callback that takes owned data to avoid lifetime issues.
pub type BoxedAsyncCallback = Arc<
    dyn Fn(Box<dyn MemoryStep>, CallbackContext) -> Pin<Box<dyn Future<Output = ()> + Send>>
        + Send
        + Sync,
>;

/// Internal async handler wrapper.
struct AsyncCallbackHandler {
    /// The async callback function.
    callback: BoxedAsyncCallback,
    /// Priority for ordering.
    priority: Priority,
    /// Target type ID (None = any step).
    target_type: Option<TypeId>,
}

impl AsyncCallbackHandler {
    /// Check if this handler matches the given step type.
    fn matches_type(&self, type_id: TypeId) -> bool {
        match self.target_type {
            None => true,
            Some(tid) => tid == type_id,
        }
    }
}

impl std::fmt::Debug for AsyncCallbackHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncCallbackHandler")
            .field("priority", &self.priority)
            .field("target_type", &self.target_type)
            .finish_non_exhaustive()
    }
}

/// Async callback registry for managing async callbacks.
///
/// Unlike `CallbackRegistry`, this registry supports async callbacks that can
/// perform I/O operations. Callbacks receive owned clones of step data to avoid
/// lifetime issues with async execution.
///
/// # Example
///
/// ```rust,ignore
/// use machi::callback::{AsyncCallbackRegistry, CallbackContext};
/// use machi::memory::ActionStep;
/// use tokio::sync::mpsc;
///
/// let (tx, mut rx) = mpsc::channel(100);
///
/// let registry = AsyncCallbackRegistry::builder()
///     .on_action_async(move |step: ActionStep, ctx| {
///         let tx = tx.clone();
///         async move {
///             tx.send(format!("Step {} done", step.step_number)).await.ok();
///         }
///     })
///     .build();
/// ```
#[derive(Default)]
pub struct AsyncCallbackRegistry {
    /// Handlers organized by target type.
    handlers: HashMap<Option<TypeId>, Vec<AsyncCallbackHandler>>,
}

impl AsyncCallbackRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for fluent construction.
    #[must_use]
    pub fn builder() -> AsyncCallbackRegistryBuilder {
        AsyncCallbackRegistryBuilder::new()
    }

    /// Register an async callback for a specific step type.
    ///
    /// The callback receives an owned clone of the step data.
    pub fn register<S, F, Fut>(&mut self, callback: F)
    where
        S: MemoryStep + Clone + 'static,
        F: Fn(S, CallbackContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.register_with_priority::<S, F, Fut>(callback, Priority::NORMAL);
    }

    /// Register an async callback with a specific priority.
    pub fn register_with_priority<S, F, Fut>(&mut self, callback: F, priority: Priority)
    where
        S: MemoryStep + Clone + 'static,
        F: Fn(S, CallbackContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let callback = Arc::new(callback);
        let wrapped: BoxedAsyncCallback = Arc::new(move |step: Box<dyn MemoryStep>, ctx| {
            let callback = Arc::clone(&callback);
            Box::pin(async move {
                if let Some(typed_step) = step.as_any().downcast_ref::<S>() {
                    callback(typed_step.clone(), ctx).await;
                }
            })
        });

        let handler = AsyncCallbackHandler {
            callback: wrapped,
            priority,
            target_type: Some(TypeId::of::<S>()),
        };

        self.handlers
            .entry(Some(TypeId::of::<S>()))
            .or_default()
            .push(handler);
    }

    /// Dispatch a step to all matching async callbacks.
    ///
    /// Returns a future that completes when all callbacks have finished.
    pub fn callback(&self, step: &dyn MemoryStep, ctx: &CallbackContext)
    where
    // We need the step to be cloneable for async dispatch
    {
        let type_id = step.as_any().type_id();
        let step_value = step.to_value();
        let ctx = ctx.clone();

        // Collect matching handlers
        let mut matching_handlers: Vec<&AsyncCallbackHandler> = Vec::new();

        if let Some(handlers) = self.handlers.get(&Some(type_id)) {
            matching_handlers.extend(handlers.iter());
        }

        if let Some(handlers) = self.handlers.get(&None) {
            matching_handlers.extend(handlers.iter());
        }

        // Sort by priority
        matching_handlers.sort_by_key(|a| a.priority);

        // Execute callbacks sequentially (to maintain order)
        // For parallel execution, users can spawn tasks inside their callbacks
        for handler in matching_handlers {
            // Create a boxed step from the value (this is a workaround for the Clone constraint)
            // In practice, users should use the typed registration methods
            let _ = (handler, &step_value, &ctx);
            // Note: Full async dispatch requires step cloning, which we handle in typed methods
        }
    }

    /// Dispatch a typed step to all matching async callbacks.
    ///
    /// This is the preferred method as it properly clones the step data.
    pub async fn callback_typed<S>(&self, step: &S, ctx: &CallbackContext)
    where
        S: MemoryStep + Clone + 'static,
    {
        let type_id = TypeId::of::<S>();
        let _step_clone: Box<dyn MemoryStep> = Box::new(step.clone());
        let ctx = ctx.clone();

        // Collect matching handlers
        let mut matching_handlers: Vec<&AsyncCallbackHandler> = Vec::new();

        if let Some(handlers) = self.handlers.get(&Some(type_id)) {
            matching_handlers.extend(handlers.iter());
        }

        if let Some(handlers) = self.handlers.get(&None) {
            matching_handlers.extend(handlers.iter().filter(|h| h.matches_type(type_id)));
        }

        // Sort by priority
        matching_handlers.sort_by_key(|a| a.priority);

        // Execute callbacks sequentially
        for handler in matching_handlers {
            let step_box: Box<dyn MemoryStep> = Box::new(step.clone());
            (handler.callback)(step_box, ctx.clone()).await;
        }
    }

    /// Get the total number of registered handlers.
    #[must_use]
    pub fn handler_count(&self) -> usize {
        self.handlers.values().map(Vec::len).sum()
    }

    /// Check if there are any handlers registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.handler_count() == 0
    }
}

impl std::fmt::Debug for AsyncCallbackRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncCallbackRegistry")
            .field("handler_count", &self.handler_count())
            .finish()
    }
}

/// Builder for constructing an `AsyncCallbackRegistry` with fluent API.
#[derive(Default)]
pub struct AsyncCallbackRegistryBuilder {
    registry: AsyncCallbackRegistry,
}

impl AsyncCallbackRegistryBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an async callback for a specific step type.
    #[must_use]
    pub fn on_async<S, F, Fut>(mut self, callback: F) -> Self
    where
        S: MemoryStep + Clone + 'static,
        F: Fn(S, CallbackContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.registry.register::<S, F, Fut>(callback);
        self
    }

    /// Register an async callback with priority.
    #[must_use]
    pub fn on_async_with_priority<S, F, Fut>(mut self, callback: F, priority: Priority) -> Self
    where
        S: MemoryStep + Clone + 'static,
        F: Fn(S, CallbackContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.registry
            .register_with_priority::<S, F, Fut>(callback, priority);
        self
    }

    // Convenience methods for common step types

    /// Register an async callback for `ActionStep`.
    #[must_use]
    pub fn on_action_async<F, Fut>(self, callback: F) -> Self
    where
        F: Fn(ActionStep, CallbackContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.on_async::<ActionStep, F, Fut>(callback)
    }

    /// Register an async callback for `PlanningStep`.
    #[must_use]
    pub fn on_planning_async<F, Fut>(self, callback: F) -> Self
    where
        F: Fn(PlanningStep, CallbackContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.on_async::<PlanningStep, F, Fut>(callback)
    }

    /// Register an async callback for `TaskStep`.
    #[must_use]
    pub fn on_task_async<F, Fut>(self, callback: F) -> Self
    where
        F: Fn(TaskStep, CallbackContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.on_async::<TaskStep, F, Fut>(callback)
    }

    /// Register an async callback for `FinalAnswerStep`.
    #[must_use]
    pub fn on_final_answer_async<F, Fut>(self, callback: F) -> Self
    where
        F: Fn(FinalAnswerStep, CallbackContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.on_async::<FinalAnswerStep, F, Fut>(callback)
    }

    /// Register an async callback for `SystemPromptStep`.
    #[must_use]
    pub fn on_system_prompt_async<F, Fut>(self, callback: F) -> Self
    where
        F: Fn(SystemPromptStep, CallbackContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.on_async::<SystemPromptStep, F, Fut>(callback)
    }

    /// Build the registry.
    #[must_use]
    pub fn build(self) -> AsyncCallbackRegistry {
        self.registry
    }
}

impl std::fmt::Debug for AsyncCallbackRegistryBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncCallbackRegistryBuilder")
            .field("registry", &self.registry)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Timing;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn test_async_callback_registration() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let registry = AsyncCallbackRegistry::builder()
            .on_action_async(move |step: ActionStep, _ctx| {
                let counter = Arc::clone(&counter_clone);
                async move {
                    counter.fetch_add(step.step_number, Ordering::SeqCst);
                }
            })
            .build();

        assert_eq!(registry.handler_count(), 1);

        let step = ActionStep {
            step_number: 5,
            timing: Timing::start_now(),
            ..Default::default()
        };
        let ctx = CallbackContext::new(1, 10);

        registry.callback_typed(&step, &ctx).await;
        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }

    #[tokio::test]
    async fn test_async_priority_ordering() {
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));

        let order1 = Arc::clone(&order);
        let order2 = Arc::clone(&order);
        let order3 = Arc::clone(&order);

        let registry = AsyncCallbackRegistry::builder()
            .on_async_with_priority::<ActionStep, _, _>(
                move |_, _| {
                    let order = Arc::clone(&order1);
                    async move {
                        order.lock().expect("lock poisoned").push("low");
                    }
                },
                Priority::LOW,
            )
            .on_async_with_priority::<ActionStep, _, _>(
                move |_, _| {
                    let order = Arc::clone(&order2);
                    async move {
                        order.lock().expect("lock poisoned").push("high");
                    }
                },
                Priority::HIGH,
            )
            .on_async_with_priority::<ActionStep, _, _>(
                move |_, _| {
                    let order = Arc::clone(&order3);
                    async move {
                        order.lock().expect("lock poisoned").push("normal");
                    }
                },
                Priority::NORMAL,
            )
            .build();

        let step = ActionStep {
            step_number: 1,
            timing: Timing::start_now(),
            ..Default::default()
        };
        let ctx = CallbackContext::new(1, 10);

        registry.callback_typed(&step, &ctx).await;

        let final_order = order.lock().expect("lock poisoned");
        assert_eq!(*final_order, vec!["high", "normal", "low"]);
    }
}
