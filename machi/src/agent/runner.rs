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
//! # Architecture
//!
//! All shared per-run state and logic lives in [`RunState`], which is
//! initialised once and then driven by either the blocking ([`Runner::run`])
//! or streaming ([`Runner::run_streamed`]) entry-point. This eliminates the
//! code duplication that would otherwise exist between the two paths.
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
use crate::chat::{ChatProvider, ChatRequest, ChatResponse, ToolChoice};
use crate::error::{AgentError, Error, Result};
use crate::guardrail::{
    InputGuardrail, InputGuardrailResult, OutputGuardrail, OutputGuardrailResult,
};
use crate::message::Message;
use crate::stream::{StreamAggregator, StreamChunk};
use crate::tool::{
    BoxedTool, ConfirmationHandler, ToolConfirmationRequest, ToolConfirmationResponse,
    ToolDefinition, ToolExecutionPolicy,
};
use crate::usage::Usage;

use super::config::Agent;
use super::hook::HookPair;
use super::result::{
    NextStep, RunConfig, RunEvent, RunResult, StepInfo, ToolCallRecord, ToolCallRequest, UserInput,
};

// ---------------------------------------------------------------------------
// StepOutcome — result of processing one reasoning step
// ---------------------------------------------------------------------------

/// The outcome of a single reasoning step after the LLM response has been
/// classified and tool calls (if any) have been executed.
enum StepOutcome {
    /// The LLM produced a final answer — the run is complete.
    Done(RunResult),
    /// Tool calls were executed; continue to the next step.
    Continue,
}

/// Holds every piece of mutable state that accumulates during a single agent
/// run. Created once by [`RunState::init`] and then driven step-by-step by
/// the two execution paths in [`Runner`].
struct RunState<'a> {
    agent: &'a Agent,
    provider: &'a dyn ChatProvider,
    context: RunContext,
    messages: Vec<Message>,
    step_history: Vec<StepInfo>,
    cumulative_usage: Usage,
    auto_approved: HashSet<String>,
    user_message: Message,
    system_prompt: String,
    all_definitions: Vec<ToolDefinition>,
    all_output_guardrails: Vec<&'a OutputGuardrail>,
    input_guardrail_results: Vec<InputGuardrailResult>,
    parallel_guardrails: Vec<&'a InputGuardrail>,
    max_steps: usize,
    max_tool_concurrency: Option<usize>,
    structured_output: bool,
}

impl<'a> RunState<'a> {
    /// Initialize all per-run state from agent configuration and user input.
    ///
    /// This performs provider validation, message construction, session
    /// history loading, tool definition collection, and sequential input
    /// guardrail execution — everything that is identical between the
    /// blocking and streaming paths.
    async fn init(agent: &'a Agent, input: UserInput, config: &'a RunConfig) -> Result<Self> {
        let provider = agent.provider.as_deref().ok_or_else(|| {
            AgentError::runtime(format!(
                "Agent '{}' has no provider configured. Call .provider() before running.",
                agent.name
            ))
        })?;

        let max_steps = config.max_steps.unwrap_or(agent.max_steps);

        let context = RunContext::new().with_agent_name(&agent.name);
        let mut messages = Vec::new();

        // Resolve system instructions.
        let system_prompt = agent.resolve_instructions();
        if !system_prompt.is_empty() {
            messages.push(Message::system(&system_prompt));
        }

        // Build user message.
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
        let all_definitions = Runner::collect_all_definitions(agent);

        // Record tool names in the agent span.
        let tool_names: Vec<&str> = all_definitions.iter().map(ToolDefinition::name).collect();
        tracing::Span::current().record("agent.tools", tracing::field::debug(&tool_names));

        // Collect guardrails from agent + run config.
        let all_input_guardrails = Runner::collect_input_guardrails(agent, config);
        let all_output_guardrails = Runner::collect_output_guardrails(agent, config);
        let mut input_guardrail_results: Vec<InputGuardrailResult> = Vec::new();

        // Split input guardrails into sequential and parallel groups.
        let sequential: Vec<&InputGuardrail> = all_input_guardrails
            .iter()
            .filter(|g| !g.is_parallel())
            .copied()
            .collect();
        let parallel: Vec<&InputGuardrail> = all_input_guardrails
            .iter()
            .filter(|g| g.is_parallel())
            .copied()
            .collect();

        // Run sequential input guardrails before the loop starts.
        if !sequential.is_empty() {
            let seq_results =
                Runner::run_input_guardrails(&sequential, &context, &agent.name, &messages).await?;
            input_guardrail_results.extend(seq_results);
        }

        Ok(Self {
            agent,
            provider,
            context,
            messages,
            step_history: Vec::new(),
            cumulative_usage: Usage::zero(),
            auto_approved: HashSet::new(),
            user_message,
            system_prompt,
            all_definitions,
            all_output_guardrails,
            input_guardrail_results,
            parallel_guardrails: parallel,
            max_steps,
            max_tool_concurrency: config.max_tool_concurrency,
            structured_output: agent.output_schema.is_some(),
        })
    }

    /// Returns the system prompt as an `Option<&str>` for hook dispatch.
    fn system_ref(&self) -> Option<&str> {
        (!self.system_prompt.is_empty()).then_some(self.system_prompt.as_str())
    }

    /// Build a non-streaming [`ChatRequest`] for the current step.
    fn build_request(&self) -> ChatRequest {
        Runner::build_request(self.agent, &self.messages, &self.all_definitions)
    }

    /// Build a streaming [`ChatRequest`] for the current step.
    fn build_stream_request(&self) -> ChatRequest {
        let mut req = self.build_request();
        req.stream = true;
        req
    }

    /// Accumulate usage from an LLM response into the running totals.
    fn accumulate_usage(&mut self, response: &ChatResponse) {
        if let Some(usage) = response.usage {
            self.cumulative_usage += usage;
            self.context.add_usage(usage);
        }
    }

    /// Process a completed LLM response and return the step outcome.
    ///
    /// This is the **shared core** between blocking and streaming paths.
    /// It classifies the response, applies tool execution policies, runs
    /// output guardrails (for final output), executes tool calls, and
    /// updates the step history.
    async fn process_step(
        &mut self,
        step: usize,
        response: ChatResponse,
        hooks: &HookPair<'_>,
        config: &RunConfig,
    ) -> Result<StepOutcome> {
        let next_step = Runner::classify_response(&response, self.structured_output);
        let (next_step, forbidden) =
            Runner::apply_policies(next_step, self.agent, &self.auto_approved);

        match next_step {
            NextStep::FinalOutput { ref output } => {
                // Append assistant message to history.
                self.messages.push(response.message.clone());

                self.step_history.push(StepInfo {
                    step,
                    response: response.clone(),
                    tool_calls: Vec::new(),
                });

                let output_value = output.clone();

                // Run output guardrails before delivering the final output.
                let output_guardrail_results = Runner::run_output_guardrails(
                    &self.all_output_guardrails,
                    &self.context,
                    &self.agent.name,
                    &output_value,
                )
                .await?;

                hooks.agent_end(&self.context, &output_value).await;

                // Persist to session if configured.
                if let Some(ref session) = config.session {
                    let to_save = vec![self.user_message.clone(), response.message.clone()];
                    let _ = session.add_messages(&to_save).await;
                }

                tracing::Span::current().record("agent.result_steps", step);
                info!(
                    agent = %self.agent.name,
                    steps = step,
                    input_tokens = self.cumulative_usage.input_tokens,
                    output_tokens = self.cumulative_usage.output_tokens,
                    "Agent run completed",
                );

                // Move accumulated state into the result to avoid cloning.
                let result = RunResult {
                    output: output_value,
                    usage: self.cumulative_usage,
                    steps: step,
                    step_history: std::mem::take(&mut self.step_history),
                    agent_name: self.agent.name.clone(),
                    input_guardrail_results: std::mem::take(&mut self.input_guardrail_results),
                    output_guardrail_results,
                };

                Ok(StepOutcome::Done(result))
            }

            NextStep::ToolCalls { ref calls } => {
                self.messages.push(response.message.clone());
                Runner::append_denied_messages(
                    &forbidden,
                    "forbidden by execution policy",
                    &mut self.messages,
                );

                let tool_records = Runner::execute_tool_calls(
                    calls,
                    self.agent,
                    &self.context,
                    hooks,
                    &mut self.messages,
                    self.max_tool_concurrency,
                )
                .await?;

                self.step_history.push(StepInfo {
                    step,
                    response,
                    tool_calls: tool_records,
                });

                Ok(StepOutcome::Continue)
            }

            NextStep::NeedsApproval {
                ref pending_approval,
                ref approved,
            } => {
                self.messages.push(response.message.clone());
                Runner::append_denied_messages(
                    &forbidden,
                    "forbidden by execution policy",
                    &mut self.messages,
                );

                let handler = config.confirmation_handler.as_deref().ok_or_else(|| {
                    AgentError::runtime(
                        "Tool execution requires approval but no confirmation handler is configured",
                    )
                })?;

                let (confirmed, denied) =
                    Runner::seek_confirmations(pending_approval, handler, &mut self.auto_approved)
                        .await;

                Runner::append_denied_messages(&denied, "denied by user", &mut self.messages);

                // Execute approved + confirmed calls.
                let executable: Vec<ToolCallRequest> =
                    approved.iter().chain(&confirmed).cloned().collect();

                let tool_records = if executable.is_empty() {
                    Vec::new()
                } else {
                    Runner::execute_tool_calls(
                        &executable,
                        self.agent,
                        &self.context,
                        hooks,
                        &mut self.messages,
                        self.max_tool_concurrency,
                    )
                    .await?
                };

                self.step_history.push(StepInfo {
                    step,
                    response,
                    tool_calls: tool_records,
                });

                Ok(StepOutcome::Continue)
            }
        }
    }
}

/// Stateless execution engine that drives an [`Agent`] through its reasoning loop.
///
/// `Runner` owns no state — all per-run state lives in [`RunState`] within
/// the run functions. This makes it safe to call `run` concurrently for
/// different agents or even the same agent with different inputs.
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

    /// Internal async implementation of the blocking agent run loop.
    async fn run_inner(agent: &Agent, input: UserInput, config: RunConfig) -> Result<RunResult> {
        let noop = NoopRunHooks;
        let run_hooks: &dyn RunHooks = config.hooks.as_deref().unwrap_or(&noop);
        let hooks = HookPair::new(run_hooks, agent.hooks.as_deref(), &agent.name);

        let mut state = RunState::init(agent, input, &config).await?;

        hooks.agent_start(&state.context).await;

        for step in 1..=state.max_steps {
            state.context.advance_step();
            debug!(agent = %agent.name, step, "Starting step");

            let request = state.build_request();

            hooks
                .llm_start(&state.context, state.system_ref(), &state.messages)
                .await;

            // On the first step, run parallel input guardrails alongside the LLM call.
            let response = if step == 1 && !state.parallel_guardrails.is_empty() {
                let (guardrail_result, llm_result) = tokio::join!(
                    Self::run_input_guardrails(
                        &state.parallel_guardrails,
                        &state.context,
                        &agent.name,
                        &state.messages,
                    ),
                    state.provider.chat(&request),
                );
                // Check guardrails first — if triggered, discard the LLM result.
                state.input_guardrail_results.extend(guardrail_result?);
                llm_result
            } else {
                state.provider.chat(&request).await
            }
            .map_err(|e| {
                error!(error = %e, agent = %agent.name, step, "LLM call failed");
                tracing::Span::current().record("error", tracing::field::display(&e));
                e
            })?;

            hooks.llm_end(&state.context, &response).await;
            state.accumulate_usage(&response);

            match state.process_step(step, response, &hooks, &config).await? {
                StepOutcome::Done(result) => return Ok(result),
                StepOutcome::Continue => {}
            }
        }

        // Exceeded max steps.
        let err = Error::from(AgentError::max_steps(state.max_steps));
        error!(error = %err, agent = %agent.name, max_steps = state.max_steps, "Max steps exceeded");
        tracing::Span::current().record("error", tracing::field::display(&err));
        hooks.error(&state.context, &err).await;
        Err(err)
    }

    /// Execute an agent run with streaming output.
    ///
    /// Returns a [`Stream`] of [`RunEvent`]s that the caller can consume
    /// in real-time. The stream yields lifecycle events (start/end),
    /// incremental text deltas, tool call progress, and the final result.
    ///
    /// The underlying LLM call uses [`ChatProvider::chat_stream`] so that
    /// text tokens are delivered as they are generated.
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
    // have a different drop order under Rust 2024. This is harmless and is a known
    // upstream issue in `async-stream`.
    #[allow(tail_expr_drop_order)]
    fn run_streamed_inner(
        agent: &Agent,
        input: UserInput,
        config: RunConfig,
    ) -> impl Stream<Item = Result<RunEvent>> + Send + '_ {
        async_stream::try_stream! {
            let noop = NoopRunHooks;
            let run_hooks: &dyn RunHooks = config.hooks.as_deref().unwrap_or(&noop);
            let hooks = HookPair::new(run_hooks, agent.hooks.as_deref(), &agent.name);

            let mut state = RunState::init(agent, input, &config).await?;

            info!(
                agent = %agent.name,
                model = %agent.model,
                tools = ?state.all_definitions.iter().map(ToolDefinition::name).collect::<Vec<_>>(),
                gen_ai.system = "machi",
                "Agent streamed run started",
            );

            hooks.agent_start(&state.context).await;
            yield RunEvent::RunStarted { agent_name: agent.name.clone() };

            for step in 1..=state.max_steps {
                state.context.advance_step();
                debug!(agent = %agent.name, step, "Starting streamed step");

                yield RunEvent::StepStarted { step };

                let request = state.build_stream_request();

                hooks
                    .llm_start(&state.context, state.system_ref(), &state.messages)
                    .await;

                // In streaming mode, parallel guardrails run before the stream
                // starts (not truly concurrent) because we cannot fork a
                // try_stream. This still provides the safety check.
                if step == 1 && !state.parallel_guardrails.is_empty() {
                    let par_results = Self::run_input_guardrails(
                        &state.parallel_guardrails,
                        &state.context,
                        &agent.name,
                        &state.messages,
                    )
                    .await?;
                    state.input_guardrail_results.extend(par_results);
                }

                // Stream chunks from the LLM.
                let mut chunk_stream = state.provider.chat_stream(&request).await?;
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

                hooks.llm_end(&state.context, &response).await;
                state.accumulate_usage(&response);

                match state.process_step(step, response, &hooks, &config).await? {
                    StepOutcome::Done(result) => {
                        // Yield step completion for the final step.
                        if let Some(last_step) = result.step_history.last() {
                            yield RunEvent::StepCompleted {
                                step_info: Box::new(last_step.clone()),
                            };
                        }
                        yield RunEvent::RunCompleted {
                            result: Box::new(result),
                        };
                        return;
                    }
                    StepOutcome::Continue => {
                        // Yield tool completion + step completion events.
                        let last = state.step_history.last().expect("just pushed");
                        for record in &last.tool_calls {
                            yield RunEvent::ToolCallCompleted {
                                record: record.clone(),
                            };
                        }
                        yield RunEvent::StepCompleted {
                            step_info: Box::new(last.clone()),
                        };
                    }
                }
            }

            // Exceeded max steps.
            let err = Error::from(AgentError::max_steps(state.max_steps));
            error!(error = %err, agent = %agent.name, max_steps = state.max_steps, "Streamed max steps exceeded");
            hooks.error(&state.context, &err).await;
            Err(err)?;
        }
    }
}

impl Runner {
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

    /// Apply tool execution policies, partitioning calls into approved,
    /// needs-approval, and forbidden groups.
    ///
    /// Returns the (possibly rewritten) [`NextStep`] and a list of calls
    /// that were forbidden by policy.
    fn apply_policies(
        next_step: NextStep,
        agent: &Agent,
        auto_approved: &HashSet<String>,
    ) -> (NextStep, Vec<ToolCallRequest>) {
        let NextStep::ToolCalls { calls } = next_step else {
            return (next_step, Vec::new());
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

            match policy {
                ToolExecutionPolicy::Auto => approved.push(call),
                ToolExecutionPolicy::RequireConfirmation => {
                    if auto_approved.contains(&call.name) {
                        approved.push(call);
                    } else {
                        pending.push(call);
                    }
                }
                ToolExecutionPolicy::Forbidden => forbidden.push(call),
            }
        }

        let result = if pending.is_empty() {
            NextStep::ToolCalls { calls: approved }
        } else {
            NextStep::NeedsApproval {
                pending_approval: pending,
                approved,
            }
        };

        (result, forbidden)
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

    /// Dispatch a regular tool call via the [`DynTool`](crate::tool::DynTool) interface.
    async fn dispatch_tool(tool: &BoxedTool, call: &ToolCallRequest) -> (String, bool) {
        match tool.call_json(call.arguments.clone()).await {
            Ok(value) => {
                let output = serde_json::to_string(&value).unwrap_or_else(|_| value.to_string());
                (output, true)
            }
            Err(e) => {
                warn!(tool = %call.name, error = %e, "Tool execution failed");
                (format!("Tool error: {e}"), false)
            }
        }
    }

    /// Request human confirmation for tool calls that require approval.
    ///
    /// Returns `(confirmed, denied)` partitioning the pending calls based
    /// on the user's responses. Calls approved via `ApproveAll` are recorded
    /// in `auto_approved` so subsequent calls skip confirmation.
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

            match response {
                ToolConfirmationResponse::Approved => confirmed.push(call.clone()),
                ToolConfirmationResponse::ApproveAll => {
                    auto_approved.insert(call.name.clone());
                    confirmed.push(call.clone());
                }
                ToolConfirmationResponse::Denied => denied.push(call.clone()),
            }
        }

        (confirmed, denied)
    }

    /// Append tool-result messages for denied/forbidden calls.
    fn append_denied_messages(
        denied: &[ToolCallRequest],
        reason: &str,
        messages: &mut Vec<Message>,
    ) {
        for call in denied {
            messages.push(Message::tool(
                &call.id,
                format!("Tool '{}' was {reason}.", call.name),
            ));
        }
    }

    /// Collect input guardrails from both agent and run config.
    fn collect_input_guardrails<'a>(
        agent: &'a Agent,
        config: &'a RunConfig,
    ) -> Vec<&'a InputGuardrail> {
        agent
            .input_guardrails
            .iter()
            .chain(config.input_guardrails.iter())
            .collect()
    }

    /// Collect output guardrails from both agent and run config.
    fn collect_output_guardrails<'a>(
        agent: &'a Agent,
        config: &'a RunConfig,
    ) -> Vec<&'a OutputGuardrail> {
        agent
            .output_guardrails
            .iter()
            .chain(config.output_guardrails.iter())
            .collect()
    }

    /// Run input guardrails and check for tripwire triggers.
    ///
    /// If any guardrail triggers its tripwire, returns
    /// [`Error::InputGuardrailTriggered`](crate::Error) immediately.
    async fn run_input_guardrails(
        guardrails: &[&InputGuardrail],
        context: &RunContext,
        agent_name: &str,
        messages: &[Message],
    ) -> Result<Vec<InputGuardrailResult>> {
        let mut results = Vec::with_capacity(guardrails.len());

        for guardrail in guardrails {
            let result = guardrail.run(context, agent_name, messages).await?;
            if result.is_triggered() {
                return Err(AgentError::input_guardrail_triggered(
                    &result.guardrail_name,
                    result.output.output_info.clone(),
                )
                .into());
            }
            results.push(result);
        }

        Ok(results)
    }

    /// Run output guardrails concurrently and check for tripwire triggers.
    ///
    /// All guardrails run in parallel via `join_all`. If any guardrail
    /// triggers its tripwire, returns [`Error::OutputGuardrailTriggered`](crate::Error).
    async fn run_output_guardrails(
        guardrails: &[&OutputGuardrail],
        context: &RunContext,
        agent_name: &str,
        output: &Value,
    ) -> Result<Vec<OutputGuardrailResult>> {
        if guardrails.is_empty() {
            return Ok(Vec::new());
        }

        let futs: Vec<_> = guardrails
            .iter()
            .map(|g| g.run(context, agent_name, output))
            .collect();
        let all_results = futures::future::join_all(futs).await;

        let mut results = Vec::with_capacity(all_results.len());
        for r in all_results {
            let result = r?;
            if result.is_triggered() {
                return Err(AgentError::output_guardrail_triggered(
                    &result.guardrail_name,
                    result.output.output_info.clone(),
                )
                .into());
            }
            results.push(result);
        }

        Ok(results)
    }
}
