//! Callback registry for managing and dispatching callbacks.

use super::context::CallbackContext;
use super::handlers::{CallbackHandler, Priority};
use crate::memory::{
    ActionStep, FinalAnswerStep, MemoryStep, PlanningStep, SystemPromptStep, TaskStep,
};
use std::any::TypeId;
use std::collections::HashMap;
use std::sync::Arc;

/// Registry for managing callbacks by step type.
///
/// Supports registering callbacks for specific step types or for all steps,
/// with priority-based execution ordering.
///
/// # Example
///
/// ```rust,ignore
/// use machi::callback::{CallbackRegistry, Priority};
/// use machi::memory::ActionStep;
///
/// let registry = CallbackRegistry::builder()
///     .on::<ActionStep>(|step, ctx| {
///         println!("Action step {} completed", step.step_number);
///     })
///     .on_any(|step, ctx| {
///         println!("Step completed");
///     })
///     .build();
/// ```
#[derive(Default)]
pub struct CallbackRegistry {
    /// Handlers organized by target type.
    /// Key is Some(TypeId) for specific types, None for "any" handlers.
    handlers: HashMap<Option<TypeId>, Vec<CallbackHandler>>,
    /// Cached sorted handlers for fast dispatch.
    sorted_cache: Option<Vec<Arc<CallbackHandler>>>,
}

impl CallbackRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for fluent construction.
    #[must_use]
    pub fn builder() -> CallbackRegistryBuilder {
        CallbackRegistryBuilder::new()
    }

    /// Register a callback for a specific step type.
    pub fn register<S, F>(&mut self, callback: F)
    where
        S: MemoryStep + 'static,
        F: Fn(&S, &CallbackContext) + Send + Sync + 'static,
    {
        self.register_with_priority::<S, F>(callback, Priority::NORMAL);
    }

    /// Register a callback with a specific priority.
    pub fn register_with_priority<S, F>(&mut self, callback: F, priority: Priority)
    where
        S: MemoryStep + 'static,
        F: Fn(&S, &CallbackContext) + Send + Sync + 'static,
    {
        let handler = CallbackHandler::new::<S, F>(callback, priority);
        let type_id = Some(TypeId::of::<S>());

        self.handlers.entry(type_id).or_default().push(handler);

        // Invalidate cache
        self.sorted_cache = None;
    }

    /// Register a callback for all step types.
    pub fn register_any<F>(&mut self, callback: F)
    where
        F: Fn(&dyn MemoryStep, &CallbackContext) + Send + Sync + 'static,
    {
        self.register_any_with_priority(callback, Priority::NORMAL);
    }

    /// Register a callback for all step types with a specific priority.
    pub fn register_any_with_priority<F>(&mut self, callback: F, priority: Priority)
    where
        F: Fn(&dyn MemoryStep, &CallbackContext) + Send + Sync + 'static,
    {
        let handler = CallbackHandler::any(callback, priority);

        self.handlers.entry(None).or_default().push(handler);

        // Invalidate cache
        self.sorted_cache = None;
    }

    /// Dispatch a step to all matching callbacks.
    pub fn callback(&self, step: &dyn MemoryStep, ctx: &CallbackContext) {
        let type_id = step.as_any().type_id();

        // Collect matching handlers with their priorities
        let mut matching_handlers: Vec<&CallbackHandler> = Vec::new();

        // Add type-specific handlers
        if let Some(handlers) = self.handlers.get(&Some(type_id)) {
            matching_handlers.extend(handlers.iter());
        }

        // Add "any" handlers
        if let Some(handlers) = self.handlers.get(&None) {
            matching_handlers.extend(handlers.iter());
        }

        // Sort by priority (higher first)
        matching_handlers.sort_by_key(|a| a.priority);

        // Invoke all matching handlers
        for handler in matching_handlers {
            handler.invoke(step, ctx);
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

    /// Get handler count for a specific step type.
    #[must_use]
    pub fn handler_count_for<S: MemoryStep + 'static>(&self) -> usize {
        self.handlers
            .get(&Some(TypeId::of::<S>()))
            .map_or(0, Vec::len)
    }

    /// Clear all handlers.
    pub fn clear(&mut self) {
        self.handlers.clear();
        self.sorted_cache = None;
    }
}

impl std::fmt::Debug for CallbackRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CallbackRegistry")
            .field("handler_count", &self.handler_count())
            .field("type_count", &self.handlers.len())
            .finish_non_exhaustive()
    }
}

/// Builder for constructing a `CallbackRegistry` with fluent API.
#[derive(Default)]
pub struct CallbackRegistryBuilder {
    registry: CallbackRegistry,
}

impl std::fmt::Debug for CallbackRegistryBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CallbackRegistryBuilder")
            .field("registry", &self.registry)
            .finish()
    }
}

impl CallbackRegistryBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a callback for a specific step type.
    #[must_use]
    pub fn on<S, F>(mut self, callback: F) -> Self
    where
        S: MemoryStep + 'static,
        F: Fn(&S, &CallbackContext) + Send + Sync + 'static,
    {
        self.registry.register::<S, F>(callback);
        self
    }

    /// Register a callback for a specific step type with priority.
    #[must_use]
    pub fn on_with_priority<S, F>(mut self, callback: F, priority: Priority) -> Self
    where
        S: MemoryStep + 'static,
        F: Fn(&S, &CallbackContext) + Send + Sync + 'static,
    {
        self.registry
            .register_with_priority::<S, F>(callback, priority);
        self
    }

    /// Register a callback for all step types.
    #[must_use]
    pub fn on_any<F>(mut self, callback: F) -> Self
    where
        F: Fn(&dyn MemoryStep, &CallbackContext) + Send + Sync + 'static,
    {
        self.registry.register_any(callback);
        self
    }

    /// Register a callback for all step types with priority.
    #[must_use]
    pub fn on_any_with_priority<F>(mut self, callback: F, priority: Priority) -> Self
    where
        F: Fn(&dyn MemoryStep, &CallbackContext) + Send + Sync + 'static,
    {
        self.registry.register_any_with_priority(callback, priority);
        self
    }

    // Convenience methods for common step types

    /// Register a callback for `ActionStep`.
    #[must_use]
    pub fn on_action<F>(self, callback: F) -> Self
    where
        F: Fn(&ActionStep, &CallbackContext) + Send + Sync + 'static,
    {
        self.on::<ActionStep, F>(callback)
    }

    /// Register a callback for `PlanningStep`.
    #[must_use]
    pub fn on_planning<F>(self, callback: F) -> Self
    where
        F: Fn(&PlanningStep, &CallbackContext) + Send + Sync + 'static,
    {
        self.on::<PlanningStep, F>(callback)
    }

    /// Register a callback for `TaskStep`.
    #[must_use]
    pub fn on_task<F>(self, callback: F) -> Self
    where
        F: Fn(&TaskStep, &CallbackContext) + Send + Sync + 'static,
    {
        self.on::<TaskStep, F>(callback)
    }

    /// Register a callback for `FinalAnswerStep`.
    #[must_use]
    pub fn on_final_answer<F>(self, callback: F) -> Self
    where
        F: Fn(&FinalAnswerStep, &CallbackContext) + Send + Sync + 'static,
    {
        self.on::<FinalAnswerStep, F>(callback)
    }

    /// Register a callback for `SystemPromptStep`.
    #[must_use]
    pub fn on_system_prompt<F>(self, callback: F) -> Self
    where
        F: Fn(&SystemPromptStep, &CallbackContext) + Send + Sync + 'static,
    {
        self.on::<SystemPromptStep, F>(callback)
    }

    /// Add a logging handler (convenience method).
    #[must_use]
    pub fn with_logging(self) -> Self {
        self.on_any_with_priority(
            |step, ctx| {
                let step_type = std::any::type_name_of_val(step)
                    .rsplit("::")
                    .next()
                    .unwrap_or("Unknown");
                tracing::debug!(
                    step_type = step_type,
                    step_number = ctx.step_number,
                    "Step callback triggered"
                );
            },
            Priority::LOWEST,
        )
    }

    /// Build the registry.
    #[must_use]
    pub fn build(self) -> CallbackRegistry {
        self.registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Timing;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_type_specific_callback() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let registry = CallbackRegistry::builder()
            .on_action(move |_step, _ctx| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            })
            .build();

        let ctx = CallbackContext::new(1, 10);

        // ActionStep should trigger
        let action = ActionStep {
            step_number: 1,
            timing: Timing::start_now(),
            ..Default::default()
        };
        registry.callback(&action, &ctx);
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // TaskStep should NOT trigger
        let task = TaskStep::new("test task");
        registry.callback(&task, &ctx);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_any_callback() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let registry = CallbackRegistry::builder()
            .on_any(move |_step, _ctx| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            })
            .build();

        let ctx = CallbackContext::new(1, 10);

        // Both should trigger
        let action = ActionStep {
            step_number: 1,
            timing: Timing::start_now(),
            ..Default::default()
        };
        registry.callback(&action, &ctx);

        let task = TaskStep::new("test task");
        registry.callback(&task, &ctx);

        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_priority_ordering() {
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));

        let order1 = Arc::clone(&order);
        let order2 = Arc::clone(&order);
        let order3 = Arc::clone(&order);

        let registry = CallbackRegistry::builder()
            .on_any_with_priority(
                move |_, _| order1.lock().expect("lock poisoned").push("low"),
                Priority::LOW,
            )
            .on_any_with_priority(
                move |_, _| order2.lock().expect("lock poisoned").push("high"),
                Priority::HIGH,
            )
            .on_any_with_priority(
                move |_, _| order3.lock().expect("lock poisoned").push("normal"),
                Priority::NORMAL,
            )
            .build();

        let ctx = CallbackContext::new(1, 10);
        let task = TaskStep::new("test");
        registry.callback(&task, &ctx);

        let final_order = order.lock().expect("lock poisoned");
        assert_eq!(*final_order, vec!["high", "normal", "low"]);
    }

    #[test]
    fn test_builder_convenience_methods() {
        let registry = CallbackRegistry::builder()
            .on_action(|_, _| {})
            .on_planning(|_, _| {})
            .on_task(|_, _| {})
            .on_final_answer(|_, _| {})
            .build();

        assert_eq!(registry.handler_count_for::<ActionStep>(), 1);
        assert_eq!(registry.handler_count_for::<PlanningStep>(), 1);
        assert_eq!(registry.handler_count_for::<TaskStep>(), 1);
        assert_eq!(registry.handler_count_for::<FinalAnswerStep>(), 1);
    }
}
