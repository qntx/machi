//! Runner — the agent execution engine.
//!
//! The [`Runner`] drives an [`Agent`] through its reasoning loop:
//!
//! 1. Build messages from instructions + conversation history
//! 2. Call the LLM with available tools
//! 3. Parse the response into a [`NextStep`]
//! 4. Execute tool calls (including managed agent sub-runs)
//! 5. Append results and loop back to step 2
//!
//! The loop terminates when the LLM produces a final text output, an error
//! occurs, or the maximum step count is exceeded.
//!
//! # Managed Agent Execution
//!
//! When a tool call targets a managed agent, the Runner spawns a recursive
//! child run for the sub-agent. Each sub-agent uses its own provider,
//! enabling heterogeneous multi-agent systems.

use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;

use futures::StreamExt as _;
use futures::stream::Stream;
use serde_json::Value;
use tracing::{Instrument, debug, error, info, info_span, warn};

use crate::callback::{NoopRunHooks, RunContext, RunHooks};
use crate::chat::{ChatRequest, ChatResponse, ToolChoice};
use crate::error::{Error, Result};
use crate::message::Message;
use crate::stream::{StreamAggregator, StreamChunk};
use crate::tool::{
    BoxedTool, ConfirmationHandler, ToolCallResult, ToolConfirmationRequest,
    ToolConfirmationResponse, ToolDefinition, ToolExecutionPolicy,
};
use crate::usage::Usage;

use super::config::Agent;
use super::hook::HookPair;
use super::result::{
    NextStep, RunConfig, RunEvent, RunResult, StepInfo, ToolCallRecord, ToolCallRequest, UserInput,
};

/// Stateless execution engine that drives an [`Agent`] through its reasoning loop.
///
/// `Runner` owns no state — all per-run state lives in local variables within
/// [`Runner::run`]. This makes it safe to call `run` concurrently for different
/// agents or even the same agent with different inputs.
#[derive(Debug, Clone, Copy)]
pub struct Runner;

impl Runner {
    /// Execute an agent run to completion.
    ///
    /// The agent's own [`provider`](Agent::provider) is used for LLM calls.
    /// Each managed sub-agent uses its own provider, enabling heterogeneous
    /// multi-agent systems with different LLMs.
    ///
    /// # Arguments
    ///
    /// * `agent` — the agent to run (must have a provider configured)
    /// * `input` — the user's input (text, multimodal, or raw content parts)
    /// * `config` — run-level configuration (hooks, session, limits)
    ///
    /// # Returns
    ///
    /// A [`RunResult`] containing the final output, usage stats, and step history.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Agent`] if no provider is configured on the agent,
    /// [`Error::MaxSteps`] if the step limit is exceeded, or propagates
    /// LLM / tool errors encountered during execution.
    pub fn run<'a>(
        agent: &'a Agent,
        input: impl Into<UserInput>,
        config: RunConfig,
    ) -> Pin<Box<dyn Future<Output = Result<RunResult>> + Send + 'a>> {
        let input = input.into();
        let span = info_span!(
            "agent",
            agent.name = %agent.name,
            agent.model = %agent.model,
            gen_ai.system = "machi",
            agent.max_steps = agent.max_steps,
            agent.tools = tracing::field::Empty,
            agent.result_steps = tracing::field::Empty,
            error = tracing::field::Empty,
        );
        Box::pin(Self::run_inner(agent, input, config).instrument(span))
    }

    /// Internal async implementation of the agent run loop.
    async fn run_inner(agent: &Agent, input: UserInput, config: RunConfig) -> Result<RunResult> {
        let provider = agent.provider.as_deref().ok_or_else(|| {
            Error::agent(format!(
                "Agent '{}' has no provider configured. Call .provider() before running.",
                agent.name
            ))
        })?;
        let max_steps = config.max_steps.unwrap_or(agent.max_steps);
        let noop = NoopRunHooks;
        let run_hooks: &dyn RunHooks = config.hooks.as_deref().unwrap_or(&noop);
        let hooks = HookPair::new(run_hooks, agent.hooks.as_deref(), &agent.name);

        let mut context = RunContext::new().with_agent_name(&agent.name);
        let mut messages = Vec::new();
        let mut step_history = Vec::new();
        let mut cumulative_usage = Usage::zero();
        let mut auto_approved: HashSet<String> = HashSet::new();

        // Resolve system instructions.
        let system_prompt = agent.resolve_instructions();

        // Build initial messages: system + user input.
        if !system_prompt.is_empty() {
            messages.push(Message::system(&system_prompt));
        }
        let user_message = input.into_message();
        messages.push(user_message.clone());

        // Load session history (inserted between system prompt and user input).
        if let Some(ref session) = config.session {
            let history = session.get_messages(None).await?;
            if !history.is_empty() {
                let insert_pos = messages.len().saturating_sub(1);
                messages.splice(insert_pos..insert_pos, history);
            }
        }

        // Collect tool definitions: regular tools + managed agent tool stubs.
        let all_definitions = Self::collect_all_definitions(agent);

        // Record tool names in the agent span.
        let tool_names: Vec<&str> = all_definitions.iter().map(ToolDefinition::name).collect();
        tracing::Span::current().record("agent.tools", tracing::field::debug(&tool_names));

        hooks.agent_start(&context).await;

        let system_ref = (!system_prompt.is_empty()).then_some(system_prompt.as_str());

        for step in 1..=max_steps {
            context.advance_step();
            debug!(agent = %agent.name, step, "Starting step");

            let request = Self::build_request(agent, &messages, &all_definitions);

            hooks.llm_start(&context, system_ref, &messages).await;

            let response = provider.chat(&request).await.map_err(|e| {
                error!(error = %e, agent = %agent.name, step, "LLM call failed");
                tracing::Span::current().record("error", tracing::field::display(&e));
                e
            })?;

            hooks.llm_end(&context, &response).await;

            // Accumulate usage.
            if let Some(usage) = response.usage {
                cumulative_usage += usage;
                context.add_usage(usage);
            }

            let structured = agent.output_schema.is_some();
            let next_step = Self::classify_response(&response, structured);
            let (next_step, forbidden) = Self::apply_policies(next_step, agent, &auto_approved);

            match next_step {
                NextStep::FinalOutput { ref output } => {
                    // Append assistant message to history.
                    messages.push(response.message.clone());

                    step_history.push(StepInfo {
                        step,
                        response: response.clone(),
                        tool_calls: Vec::new(),
                    });

                    let output_value = output.clone();
                    hooks.agent_end(&context, &output_value).await;

                    // Persist to session if configured.
                    if let Some(ref session) = config.session {
                        let to_save = vec![user_message, response.message.clone()];
                        let _ = session.add_messages(&to_save).await;
                    }

                    tracing::Span::current().record("agent.result_steps", step);
                    info!(
                        agent = %agent.name,
                        steps = step,
                        input_tokens = cumulative_usage.input_tokens,
                        output_tokens = cumulative_usage.output_tokens,
                        "Agent run completed",
                    );

                    return Ok(RunResult {
                        output: output_value,
                        usage: cumulative_usage,
                        steps: step,
                        step_history,
                        agent_name: agent.name.clone(),
                    });
                }

                NextStep::ToolCalls { ref calls } => {
                    messages.push(response.message.clone());
                    Self::append_denied_messages(
                        &forbidden,
                        "forbidden by execution policy",
                        &mut messages,
                    );

                    let tool_records = Self::execute_tool_calls(
                        calls,
                        agent,
                        &context,
                        &hooks,
                        &mut messages,
                        config.max_tool_concurrency,
                    )
                    .await?;

                    step_history.push(StepInfo {
                        step,
                        response,
                        tool_calls: tool_records,
                    });
                }

                NextStep::NeedsApproval {
                    ref pending_approval,
                    ref approved,
                } => {
                    messages.push(response.message.clone());
                    Self::append_denied_messages(
                        &forbidden,
                        "forbidden by execution policy",
                        &mut messages,
                    );

                    let handler = config.confirmation_handler.as_deref().ok_or_else(|| {
                        Error::agent(
                            "Tool execution requires approval but no confirmation handler is configured",
                        )
                    })?;

                    let (confirmed, denied) =
                        Self::seek_confirmations(pending_approval, handler, &mut auto_approved)
                            .await;

                    Self::append_denied_messages(&denied, "denied by user", &mut messages);

                    // Execute approved + confirmed calls.
                    let executable: Vec<ToolCallRequest> =
                        approved.iter().chain(&confirmed).cloned().collect();

                    let tool_records = if executable.is_empty() {
                        Vec::new()
                    } else {
                        Self::execute_tool_calls(
                            &executable,
                            agent,
                            &context,
                            &hooks,
                            &mut messages,
                            config.max_tool_concurrency,
                        )
                        .await?
                    };

                    step_history.push(StepInfo {
                        step,
                        response,
                        tool_calls: tool_records,
                    });
                }

                NextStep::MaxStepsExceeded => {
                    unreachable!("MaxStepsExceeded is only set outside the loop");
                }
            }
        }

        // Exceeded max steps.
        let err = Error::max_steps(max_steps);
        error!(error = %err, agent = %agent.name, max_steps, "Max steps exceeded");
        tracing::Span::current().record("error", tracing::field::display(&err));
        hooks.error(&context, &err).await;

        Err(err)
    }

    /// Collect [`ToolDefinition`]s from regular tools and managed agents.
    fn collect_all_definitions(agent: &Agent) -> Vec<ToolDefinition> {
        agent
            .tools
            .iter()
            .map(|t| t.definition())
            .chain(agent.managed_agents.iter().map(Agent::tool_definition))
            .collect()
    }

    /// Build a [`ChatRequest`] for the current step.
    fn build_request(
        agent: &Agent,
        messages: &[Message],
        definitions: &[ToolDefinition],
    ) -> ChatRequest {
        let mut request = ChatRequest::with_messages(&agent.model, messages.to_vec());
        if !definitions.is_empty() {
            request = request
                .tools(definitions.to_vec())
                .tool_choice(ToolChoice::Auto)
                .parallel_tool_calls(true);
        }
        // Apply structured output schema when configured on the agent.
        if let Some(ref schema) = agent.output_schema {
            request = request.response_format(schema.to_response_format());
        }
        request
    }

    /// Build a streaming [`ChatRequest`] for the current step.
    fn build_stream_request(
        agent: &Agent,
        messages: &[Message],
        definitions: &[ToolDefinition],
    ) -> ChatRequest {
        let mut request = Self::build_request(agent, messages, definitions);
        request.stream = true;
        request
    }

    /// Classify an LLM response into a [`NextStep`].
    ///
    /// When `structured_output` is `true`, the text content is parsed as JSON
    /// so that [`RunResult::output`] contains a structured [`Value`] rather
    /// than a plain string.
    fn classify_response(response: &ChatResponse, structured_output: bool) -> NextStep {
        if let Some(tool_calls) = response.tool_calls() {
            let calls: Vec<ToolCallRequest> =
                tool_calls.iter().map(ToolCallRequest::from).collect();
            if !calls.is_empty() {
                return NextStep::ToolCalls { calls };
            }
        }
        let output = if structured_output {
            // Parse the LLM's text as JSON for structured output.
            response.text().map_or(Value::Null, |text| {
                serde_json::from_str(&text).unwrap_or(Value::String(text))
            })
        } else {
            response.text().map_or(Value::Null, Value::String)
        };
        NextStep::FinalOutput { output }
    }

    /// Execute tool calls concurrently and append results to messages.
    ///
    /// Runs up to `max_concurrency` calls in parallel per chunk using
    /// [`futures::future::join_all`], preserving the original call order.
    /// When `max_concurrency` is `None`, all calls run simultaneously.
    async fn execute_tool_calls(
        calls: &[ToolCallRequest],
        agent: &Agent,
        context: &RunContext,
        hooks: &HookPair<'_>,
        messages: &mut Vec<Message>,
        max_concurrency: Option<usize>,
    ) -> Result<Vec<ToolCallRecord>> {
        let concurrency = max_concurrency.unwrap_or(calls.len()).max(1);
        let mut records = Vec::with_capacity(calls.len());

        for chunk in calls.chunks(concurrency) {
            let mut futs = Vec::with_capacity(chunk.len());
            for call in chunk {
                futs.push(Self::execute_single_tool(call, agent, context, hooks));
            }
            records.extend(futures::future::join_all(futs).await);
        }

        // Append tool result messages in original call order.
        for record in &records {
            messages.push(Message::tool(&record.id, &record.result));
        }

        Ok(records)
    }

    /// Execute a single tool call with lifecycle hooks.
    ///
    /// Fires `tool_start` before dispatch and `tool_end` after, then returns
    /// the completed [`ToolCallRecord`].
    async fn execute_single_tool(
        call: &ToolCallRequest,
        agent: &Agent,
        context: &RunContext,
        hooks: &HookPair<'_>,
    ) -> ToolCallRecord {
        let tool_span = info_span!(
            "tool",
            tool.name = %call.name,
            tool.id = %call.id,
            tool.input = %call.arguments,
            tool.output = tracing::field::Empty,
            tool.success = tracing::field::Empty,
            error = tracing::field::Empty,
        );

        async {
            hooks.tool_start(context, &call.name).await;

            let (result_str, success) =
                if let Some(sub) = agent.managed_agents.iter().find(|a| a.name == call.name) {
                    Self::dispatch_managed_agent(sub, &call.arguments).await
                } else if let Some(tool) = agent.tools.iter().find(|t| t.name() == call.name) {
                    Self::dispatch_tool(tool, call).await
                } else {
                    warn!(tool = %call.name, "Tool not found");
                    (format!("Tool '{}' not found", call.name), false)
                };

            let current = tracing::Span::current();
            current.record("tool.success", success);
            current.record("tool.output", result_str.as_str());
            if !success {
                current.record("error", result_str.as_str());
            }
            hooks.tool_end(context, &call.name, &result_str).await;

            ToolCallRecord {
                id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
                result: result_str,
                success,
            }
        }
        .instrument(tool_span)
        .await
    }

    /// Run a managed sub-agent with the given task arguments.
    async fn dispatch_managed_agent(sub_agent: &Agent, args: &Value) -> (String, bool) {
        let task = args.get("task").and_then(Value::as_str).unwrap_or_default();
        info!(
            from_agent = tracing::field::Empty,
            to_agent = %sub_agent.name,
            "Handoff to managed agent",
        );
        match Self::run(sub_agent, task, RunConfig::default()).await {
            Ok(result) => {
                let output = serde_json::to_string(&result.output)
                    .unwrap_or_else(|_| result.output.to_string());
                (output, true)
            }
            Err(e) => (
                format!("Managed agent '{}' failed: {e}", sub_agent.name),
                false,
            ),
        }
    }

    /// Execute a regular tool call and format the result for the LLM.
    async fn dispatch_tool(tool: &BoxedTool, call: &ToolCallRequest) -> (String, bool) {
        let result = tool.call_json(call.arguments.clone()).await;
        let record = ToolCallResult {
            id: call.id.clone(),
            name: call.name.clone(),
            result,
        };
        (record.to_string_for_llm(), record.is_success())
    }

    /// Partition tool calls by execution policy.
    ///
    /// Returns a `(next_step, forbidden)` tuple:
    /// - `next_step` is either `ToolCalls` (all auto-approved) or
    ///   `NeedsApproval` (some require confirmation).
    /// - `forbidden` contains calls blocked by [`ToolExecutionPolicy::Forbidden`].
    fn apply_policies(
        next: NextStep,
        agent: &Agent,
        auto_approved: &HashSet<String>,
    ) -> (NextStep, Vec<ToolCallRequest>) {
        let NextStep::ToolCalls { calls } = next else {
            return (next, Vec::new());
        };

        let mut approved = Vec::new();
        let mut pending = Vec::new();
        let mut forbidden = Vec::new();

        for call in calls {
            let policy = agent
                .tool_policies
                .get(&call.name)
                .copied()
                .unwrap_or(ToolExecutionPolicy::Auto);

            if policy.is_forbidden() {
                forbidden.push(call);
            } else if policy.requires_confirmation() && !auto_approved.contains(&call.name) {
                pending.push(call);
            } else {
                approved.push(call);
            }
        }

        let step = if pending.is_empty() {
            NextStep::ToolCalls { calls: approved }
        } else {
            NextStep::NeedsApproval {
                pending_approval: pending,
                approved,
            }
        };

        (step, forbidden)
    }

    /// Sequentially request confirmation for each pending tool call.
    ///
    /// Returns `(confirmed, denied)` vectors. Calls approved with
    /// [`ToolConfirmationResponse::ApproveAll`] are added to
    /// `auto_approved` so future invocations skip confirmation.
    async fn seek_confirmations(
        pending: &[ToolCallRequest],
        handler: &dyn ConfirmationHandler,
        auto_approved: &mut HashSet<String>,
    ) -> (Vec<ToolCallRequest>, Vec<ToolCallRequest>) {
        let mut confirmed = Vec::new();
        let mut denied = Vec::new();

        for call in pending {
            let request =
                ToolConfirmationRequest::new(&call.id, &call.name, call.arguments.clone());
            let response = handler.confirm(&request).await;

            if response.is_approved() {
                if matches!(response, ToolConfirmationResponse::ApproveAll) {
                    auto_approved.insert(call.name.clone());
                }
                confirmed.push(call.clone());
            } else {
                denied.push(call.clone());
            }
        }

        (confirmed, denied)
    }

    /// Append tool-result denial messages for forbidden or denied calls.
    fn append_denied_messages(
        calls: &[ToolCallRequest],
        reason: &str,
        messages: &mut Vec<Message>,
    ) {
        for call in calls {
            messages.push(Message::tool(
                &call.id,
                format!("Tool '{}' {reason}", call.name),
            ));
        }
    }

    /// Execute an agent run with streaming output.
    ///
    /// Returns a [`Stream`] of [`RunEvent`]s that the caller can consume
    /// in real-time. The stream yields lifecycle events (start/end),
    /// incremental text deltas, tool call progress, and the final result.
    ///
    /// The underlying LLM call uses [`ChatProvider::chat_stream`] so that
    /// text tokens are delivered as they are generated.
    ///
    /// # Arguments
    ///
    /// * `agent` — the agent to run (must have a provider configured)
    /// * `input` — the user's input (text, multimodal, or raw content parts)
    /// * `config` — run-level configuration (hooks, session, limits)
    ///
    /// # Returns
    ///
    /// A pinned stream of `Result<RunEvent>`. Errors terminate the stream.
    pub fn run_streamed<'a>(
        agent: &'a Agent,
        input: impl Into<UserInput>,
        config: RunConfig,
    ) -> Pin<Box<dyn Stream<Item = Result<RunEvent>> + Send + 'a>> {
        let input = input.into();
        Box::pin(Self::run_streamed_inner(agent, input, config))
    }

    /// Internal streaming implementation of the agent run loop.
    //
    // The `tail_expr_drop_order` warning originates inside the `try_stream!` macro
    // expansion, where temporaries in the generated async block's tail expression
    // have a different drop order under Rust 2024. This is harmless (no locks or
    // channels involved) and is a known upstream issue in `async-stream`.
    #[allow(tail_expr_drop_order)]
    fn run_streamed_inner(
        agent: &Agent,
        input: UserInput,
        config: RunConfig,
    ) -> impl Stream<Item = Result<RunEvent>> + Send + '_ {
        async_stream::try_stream! {
            let provider = agent.provider.as_deref().ok_or_else(|| {
                Error::agent(format!(
                    "Agent '{}' has no provider configured. Call .provider() before running.",
                    agent.name
                ))
            })?;
            let max_steps = config.max_steps.unwrap_or(agent.max_steps);
            let noop = NoopRunHooks;
            let run_hooks: &dyn RunHooks = config.hooks.as_deref().unwrap_or(&noop);
            let hooks = HookPair::new(run_hooks, agent.hooks.as_deref(), &agent.name);

            let mut context = RunContext::new().with_agent_name(&agent.name);
            let mut messages = Vec::new();
            let mut step_history = Vec::new();
            let mut cumulative_usage = Usage::zero();
            let mut auto_approved: HashSet<String> = HashSet::new();

            // Resolve system instructions.
            let system_prompt = agent.resolve_instructions();

            // Build initial messages: system + user input.
            if !system_prompt.is_empty() {
                messages.push(Message::system(&system_prompt));
            }
            let user_message = input.into_message();
            messages.push(user_message.clone());

            // Load session history.
            if let Some(ref session) = config.session {
                let history = session.get_messages(None).await?;
                if !history.is_empty() {
                    let insert_pos = messages.len().saturating_sub(1);
                    messages.splice(insert_pos..insert_pos, history);
                }
            }

            let all_definitions = Self::collect_all_definitions(agent);
            let tool_names: Vec<&str> = all_definitions.iter().map(ToolDefinition::name).collect();

            info!(
                agent = %agent.name,
                model = %agent.model,
                tools = ?tool_names,
                gen_ai.system = "machi",
                "Agent streamed run started",
            );

            hooks.agent_start(&context).await;
            yield RunEvent::RunStarted { agent_name: agent.name.clone() };

            let system_ref = (!system_prompt.is_empty()).then_some(system_prompt.as_str());

            for step in 1..=max_steps {
                context.advance_step();
                debug!(agent = %agent.name, step, "Starting streamed step");

                yield RunEvent::StepStarted { step };

                let request = Self::build_stream_request(agent, &messages, &all_definitions);

                hooks.llm_start(&context, system_ref, &messages).await;

                // Stream chunks from the LLM.
                let mut chunk_stream = provider.chat_stream(&request).await?;
                let mut aggregator = StreamAggregator::new();

                while let Some(chunk_result) = chunk_stream.next().await {
                    let chunk = chunk_result?;

                    // Yield real-time events for displayable content.
                    match &chunk {
                        StreamChunk::Text(delta) => {
                            yield RunEvent::TextDelta(delta.clone());
                        }
                        StreamChunk::ReasoningContent(delta) => {
                            yield RunEvent::ReasoningDelta(delta.clone());
                        }
                        StreamChunk::Audio { data, transcript } => {
                            yield RunEvent::AudioDelta {
                                data: data.clone(),
                                transcript: transcript.clone(),
                            };
                        }
                        StreamChunk::ToolUseStart { id, name, .. } => {
                            yield RunEvent::ToolCallStarted {
                                id: id.clone(),
                                name: name.clone(),
                            };
                        }
                        _ => {}
                    }

                    aggregator.apply(&chunk);
                }

                // Reconstruct a complete response from accumulated chunks.
                let response = aggregator.into_chat_response();

                hooks.llm_end(&context, &response).await;

                // Accumulate usage.
                if let Some(usage) = response.usage {
                    cumulative_usage += usage;
                    context.add_usage(usage);
                }

                let structured = agent.output_schema.is_some();
                let next_step = Self::classify_response(&response, structured);
                let (next_step, forbidden) = Self::apply_policies(next_step, agent, &auto_approved);

                match next_step {
                    NextStep::FinalOutput { ref output } => {
                        messages.push(response.message.clone());

                        step_history.push(StepInfo {
                            step,
                            response: response.clone(),
                            tool_calls: Vec::new(),
                        });

                        let output_value = output.clone();
                        hooks.agent_end(&context, &output_value).await;

                        yield RunEvent::StepCompleted {
                            step_info: Box::new(step_history.last().expect("just pushed").clone()),
                        };

                        // Persist to session if configured.
                        if let Some(ref session) = config.session {
                            let to_save = vec![user_message, response.message.clone()];
                            let _ = session.add_messages(&to_save).await;
                        }

                        tracing::Span::current().record("agent.result_steps", step);
                        info!(
                            agent = %agent.name,
                            steps = step,
                            input_tokens = cumulative_usage.input_tokens,
                            output_tokens = cumulative_usage.output_tokens,
                            "Agent streamed run completed",
                        );

                        yield RunEvent::RunCompleted {
                            result: Box::new(RunResult {
                                output: output_value,
                                usage: cumulative_usage,
                                steps: step,
                                step_history,
                                agent_name: agent.name.clone(),
                            }),
                        };
                        return;
                    }

                    NextStep::ToolCalls { ref calls } => {
                        messages.push(response.message.clone());
                        Self::append_denied_messages(
                            &forbidden,
                            "forbidden by execution policy",
                            &mut messages,
                        );

                        let tool_records = Self::execute_tool_calls(
                            calls,
                            agent,
                            &context,
                            &hooks,
                            &mut messages,
                            config.max_tool_concurrency,
                        )
                        .await?;

                        // Yield individual tool completion events.
                        for record in &tool_records {
                            yield RunEvent::ToolCallCompleted { record: record.clone() };
                        }

                        step_history.push(StepInfo {
                            step,
                            response,
                            tool_calls: tool_records,
                        });

                        yield RunEvent::StepCompleted {
                            step_info: Box::new(step_history.last().expect("just pushed").clone()),
                        };
                    }

                    NextStep::NeedsApproval {
                        ref pending_approval,
                        ref approved,
                    } => {
                        messages.push(response.message.clone());
                        Self::append_denied_messages(
                            &forbidden,
                            "forbidden by execution policy",
                            &mut messages,
                        );

                        let handler = config.confirmation_handler.as_deref().ok_or_else(|| {
                            Error::agent(
                                "Tool execution requires approval but no confirmation handler is configured",
                            )
                        })?;

                        let (confirmed, denied) =
                            Self::seek_confirmations(pending_approval, handler, &mut auto_approved)
                                .await;

                        Self::append_denied_messages(&denied, "denied by user", &mut messages);

                        let executable: Vec<ToolCallRequest> =
                            approved.iter().chain(&confirmed).cloned().collect();

                        let tool_records = if executable.is_empty() {
                            Vec::new()
                        } else {
                            Self::execute_tool_calls(
                                &executable,
                                agent,
                                &context,
                                &hooks,
                                &mut messages,
                                config.max_tool_concurrency,
                            )
                            .await?
                        };

                        for record in &tool_records {
                            yield RunEvent::ToolCallCompleted { record: record.clone() };
                        }

                        step_history.push(StepInfo {
                            step,
                            response,
                            tool_calls: tool_records,
                        });

                        yield RunEvent::StepCompleted {
                            step_info: Box::new(step_history.last().expect("just pushed").clone()),
                        };
                    }

                    NextStep::MaxStepsExceeded => {
                        unreachable!("MaxStepsExceeded is only set outside the loop");
                    }
                }
            }

            // Exceeded max steps.
            let err = Error::max_steps(max_steps);
            error!(error = %err, agent = %agent.name, max_steps, "Streamed max steps exceeded");
            hooks.error(&context, &err).await;
            Err(err)?;
        }
    }
}
