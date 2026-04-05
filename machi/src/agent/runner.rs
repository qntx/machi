//! Agent execution engine.
//!
//! [`Runner`] drives an [`Agent`] through a reasoning loop:
//!
//! 1. Build messages from instructions + conversation history
//! 2. Call the LLM with available tools
//! 3. Classify the response into a [`NextStep`]
//! 4. Execute tool calls (including managed-agent sub-runs)
//! 5. Append results and loop back to step 2
//!
//! The loop terminates on a final output, an error, or the step limit.
//! All per-run state lives in [`RunState`], initialised once and driven by
//! [`Runner::run`] (blocking) or [`Runner::run_streamed`] (streaming).

use std::{collections::HashSet, future::Future, pin::Pin};

use futures::{StreamExt as _, stream::Stream};
use serde_json::Value;
use tracing::{Instrument, debug, error, info, info_span, warn};

use super::{
    config::Agent,
    result::{
        NextStep, RunConfig, RunEvent, RunResult, StepInfo, ToolCallRecord, ToolCallRequest,
        UserInput,
    },
};
use crate::{
    chat::{ChatProvider, ChatRequest, ChatResponse, ToolChoice},
    context::SharedContextStrategy,
    error::{AgentError, Error, Result},
    guardrail::{InputGuardrail, InputGuardrailResult, OutputGuardrail, OutputGuardrailResult},
    hooks::{Hooks, NoopHooks, RunContext},
    message::Message,
    middleware::{self, MiddlewareContext, SharedMiddleware, ToolCallAction},
    stream::{StreamAggregator, StreamChunk},
    tool::{
        BoxedTool, ConcurrencyMode, ConfirmationHandler, ToolConfirmationRequest,
        ToolConfirmationResponse, ToolDefinition, ToolExecutionPolicy,
    },
    usage::Usage,
};

/// Outcome of processing one reasoning step.
enum StepOutcome {
    /// Final answer produced — run complete.
    Done(RunResult),
    /// Tool calls executed — continue looping.
    Continue,
}

/// Per-run mutable state, created once by [`init`](Self::init) and driven
/// step-by-step by [`Runner::run`] or [`Runner::run_streamed`].
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
    context_strategy: Option<SharedContextStrategy>,
    middleware: Vec<SharedMiddleware>,
}

impl<'a> RunState<'a> {
    /// Build all per-run state from agent config and user input.
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

        let system_prompt = agent.resolve_instructions();
        if !system_prompt.is_empty() {
            messages.push(Message::system(&system_prompt));
        }

        let user_message = input.into_message();
        messages.push(user_message.clone());

        // Insert session history before the user message.
        if let Some(ref session) = config.session {
            let history = session.get_messages(None).await?;
            if !history.is_empty() {
                let insert_pos = messages.len().saturating_sub(1);
                messages.splice(insert_pos..insert_pos, history);
            }
        }

        let all_definitions = Runner::collect_all_definitions(agent);
        let tool_names: Vec<&str> = all_definitions.iter().map(ToolDefinition::name).collect();
        tracing::Span::current().record("agent.tools", tracing::field::debug(&tool_names));

        // Guardrails: sequential ones run now, parallel ones run with the first LLM call.
        let all_input_guardrails = Runner::collect_input_guardrails(agent, config);
        let all_output_guardrails = Runner::collect_output_guardrails(agent, config);
        let mut input_guardrail_results = Vec::new();

        let sequential: Vec<_> = all_input_guardrails
            .iter()
            .filter(|g| !g.is_parallel())
            .copied()
            .collect();
        let parallel: Vec<_> = all_input_guardrails
            .iter()
            .filter(|g| g.is_parallel())
            .copied()
            .collect();

        if !sequential.is_empty() {
            let results =
                Runner::run_input_guardrails(&sequential, &context, &agent.name, &messages).await?;
            input_guardrail_results.extend(results);
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
            context_strategy: config.context_strategy.clone(),
            middleware: config.middleware.clone(),
        })
    }

    /// System prompt as `Option<&str>` for hook dispatch.
    fn system_ref(&self) -> Option<&str> {
        (!self.system_prompt.is_empty()).then_some(self.system_prompt.as_str())
    }

    /// Build a [`ChatRequest`] for the current step, applying context
    /// compaction if a strategy is configured.
    async fn build_request(&self) -> Result<ChatRequest> {
        if let Some(ref strategy) = self.context_strategy {
            let compacted = strategy.compact(&self.messages).await?;
            Ok(Runner::build_request(
                self.agent,
                &compacted,
                &self.all_definitions,
            ))
        } else {
            Ok(Runner::build_request(
                self.agent,
                &self.messages,
                &self.all_definitions,
            ))
        }
    }

    /// Build a streaming [`ChatRequest`] for the current step.
    async fn build_stream_request(&self) -> Result<ChatRequest> {
        let mut req = self.build_request().await?;
        req.stream = true;
        Ok(req)
    }

    /// Accumulate usage from an LLM response into the running totals.
    fn accumulate_usage(&mut self, response: &ChatResponse) {
        if let Some(usage) = response.usage {
            self.cumulative_usage += usage;
            self.context.add_usage(usage);
        }
    }

    /// Accumulate sub-agent usage from tool call records into the running totals.
    fn accumulate_tool_usage(&mut self, records: &[ToolCallRecord]) {
        for record in records {
            if record.sub_usage.total_tokens > 0 {
                self.cumulative_usage += record.sub_usage;
                self.context.add_usage(record.sub_usage);
            }
        }
    }

    /// Process a completed LLM response — the shared core of both paths.
    ///
    /// Classifies the response, applies tool policies, runs output
    /// guardrails, executes tool calls, and updates step history.
    async fn process_step(
        &mut self,
        step: usize,
        response: ChatResponse,
        hooks: &dyn Hooks,
        agent_name: &str,
        config: &RunConfig,
    ) -> Result<StepOutcome> {
        let next_step = Runner::classify_response(&response, self.structured_output);
        let (next_step, forbidden) =
            Runner::apply_policies(next_step, self.agent, &self.auto_approved);

        match next_step {
            NextStep::FinalOutput { ref output } => {
                self.messages.push(response.message.clone());
                self.step_history.push(StepInfo {
                    step,
                    response: response.clone(),
                    tool_calls: Vec::new(),
                });

                let output_value = output.clone();
                let output_guardrail_results = Runner::run_output_guardrails(
                    &self.all_output_guardrails,
                    &self.context,
                    &self.agent.name,
                    &output_value,
                )
                .await?;

                hooks
                    .on_agent_end(&self.context, agent_name, &output_value)
                    .await;
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
                    agent_name,
                    &mut self.messages,
                    self.max_tool_concurrency,
                    &self.middleware,
                )
                .await?;

                self.accumulate_tool_usage(&tool_records);
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
                        agent_name,
                        &mut self.messages,
                        self.max_tool_concurrency,
                        &self.middleware,
                    )
                    .await?
                };

                self.accumulate_tool_usage(&tool_records);
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

/// Stateless execution engine that drives an [`Agent`] through its reasoning
/// loop. All per-run state lives in [`RunState`], making concurrent calls safe.
#[derive(Debug, Clone, Copy)]
pub struct Runner;

impl Runner {
    /// Execute an agent run to completion, returning a [`RunResult`].
    ///
    /// Uses the agent's own provider for LLM calls. Each managed sub-agent
    /// uses its own provider, enabling heterogeneous multi-agent systems.
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

    /// Core async loop for the blocking execution path.
    async fn run_inner(agent: &Agent, input: UserInput, config: RunConfig) -> Result<RunResult> {
        let noop = NoopHooks;
        let hooks: &dyn Hooks = config.hooks.as_deref().unwrap_or(&noop);

        let mut state = RunState::init(agent, input, &config).await?;

        hooks.on_agent_start(&state.context, &agent.name).await;

        for step in 1..=state.max_steps {
            state.context.advance_step();
            debug!(agent = %agent.name, step, "Starting step");

            let request = state.build_request().await?;

            // Run pre-LLM middleware.
            if !state.middleware.is_empty() {
                let mw_ctx = MiddlewareContext::new(state.context.clone(), agent.name.clone());
                middleware::run_llm_request_middleware(&state.middleware, &mw_ctx, &state.messages)
                    .await?;
            }

            hooks
                .on_llm_start(
                    &state.context,
                    &agent.name,
                    state.system_ref(),
                    &state.messages,
                )
                .await;

            // First step: run parallel guardrails alongside the LLM call.
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

            hooks
                .on_llm_end(&state.context, &agent.name, &response)
                .await;
            state.accumulate_usage(&response);

            match state
                .process_step(step, response, hooks, &agent.name, &config)
                .await?
            {
                StepOutcome::Done(result) => return Ok(result),
                StepOutcome::Continue => {}
            }
        }

        let err = Error::from(AgentError::max_steps(state.max_steps));
        error!(error = %err, agent = %agent.name, max_steps = state.max_steps, "Max steps exceeded");
        tracing::Span::current().record("error", tracing::field::display(&err));
        hooks.on_error(&state.context, &agent.name, &err).await;
        Err(err)
    }

    /// Execute an agent run with streaming output.
    ///
    /// Returns a [`Stream`] of [`RunEvent`]s: lifecycle events, text deltas,
    /// tool-call progress, and the final result.
    pub fn run_streamed<'a>(
        agent: &'a Agent,
        input: impl Into<UserInput>,
        config: RunConfig,
    ) -> Pin<Box<dyn Stream<Item = Result<RunEvent>> + Send + 'a>> {
        let input = input.into();
        Box::pin(Self::run_streamed_inner(agent, input, config))
    }

    /// Core streaming loop.
    // `tail_expr_drop_order`: false positive from the `try_stream!` macro.
    #[allow(tail_expr_drop_order)]
    fn run_streamed_inner(
        agent: &Agent,
        input: UserInput,
        config: RunConfig,
    ) -> impl Stream<Item = Result<RunEvent>> + Send + '_ {
        async_stream::try_stream! {
            let noop = NoopHooks;
            let hooks: &dyn Hooks = config.hooks.as_deref().unwrap_or(&noop);

            let mut state = RunState::init(agent, input, &config).await?;

            info!(
                agent = %agent.name,
                model = %agent.model,
                tools = ?state.all_definitions.iter().map(ToolDefinition::name).collect::<Vec<_>>(),
                gen_ai.system = "machi",
                "Agent streamed run started",
            );

            hooks.on_agent_start(&state.context, &agent.name).await;
            yield RunEvent::RunStarted { agent_name: agent.name.clone() };

            for step in 1..=state.max_steps {
                state.context.advance_step();
                debug!(agent = %agent.name, step, "Starting streamed step");

                yield RunEvent::StepStarted { step };

                let request = state.build_stream_request().await?;

                // Run pre-LLM middleware.
                if !state.middleware.is_empty() {
                    let mw_ctx = MiddlewareContext::new(state.context.clone(), agent.name.clone());
                    middleware::run_llm_request_middleware(
                        &state.middleware,
                        &mw_ctx,
                        &state.messages,
                    )
                    .await?;
                }

                hooks
                    .on_llm_start(&state.context, &agent.name, state.system_ref(), &state.messages)
                    .await;

                // Parallel guardrails run sequentially here (cannot fork a try_stream).
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

                let mut chunk_stream = state.provider.chat_stream(&request).await?;
                let mut aggregator = StreamAggregator::new();

                while let Some(chunk_result) = chunk_stream.next().await {
                    let chunk = chunk_result?;

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

                let response = aggregator.into_chat_response();

                hooks.on_llm_end(&state.context, &agent.name, &response).await;
                state.accumulate_usage(&response);

                match state.process_step(step, response, hooks, &agent.name, &config).await? {
                    StepOutcome::Done(result) => {
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

            let err = Error::from(AgentError::max_steps(state.max_steps));
            error!(error = %err, agent = %agent.name, max_steps = state.max_steps, "Max steps exceeded");
            hooks.on_error(&state.context, &agent.name, &err).await;
            Err(err)?;
        }
    }
}

impl Runner {
    /// Collect tool definitions from regular tools and managed agents.
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
        if let Some(ref schema) = agent.output_schema {
            request = request.response_format(schema.to_response_format());
        }
        request
    }

    /// Classify an LLM response into a [`NextStep`].
    ///
    /// When `structured_output` is true, text is parsed as JSON.
    fn classify_response(response: &ChatResponse, structured_output: bool) -> NextStep {
        if let Some(tool_calls) = response.tool_calls() {
            let calls: Vec<ToolCallRequest> =
                tool_calls.iter().map(ToolCallRequest::from).collect();
            if !calls.is_empty() {
                return NextStep::ToolCalls { calls };
            }
        }
        let output = if structured_output {
            response.text().map_or(Value::Null, |text| {
                serde_json::from_str(&text).unwrap_or(Value::String(text))
            })
        } else {
            response.text().map_or(Value::Null, Value::String)
        };
        NextStep::FinalOutput { output }
    }

    /// Apply tool execution policies, returning the rewritten [`NextStep`]
    /// and any calls forbidden by policy.
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

    /// Execute tool calls with metadata-aware scheduling, appending results to messages.
    ///
    /// Scheduling strategy based on [`ConcurrencyMode`]:
    /// 1. All `ReadOnly` and `Safe` calls in a batch run concurrently (respecting
    ///    `max_concurrency`).
    /// 2. `Exclusive` calls run one at a time, after all concurrent calls finish.
    ///
    /// Results are appended to `messages` in the original call order regardless
    /// of execution order.
    async fn execute_tool_calls(
        calls: &[ToolCallRequest],
        agent: &Agent,
        context: &RunContext,
        hooks: &dyn Hooks,
        agent_name: &str,
        messages: &mut Vec<Message>,
        max_concurrency: Option<usize>,
        mw: &[SharedMiddleware],
    ) -> Result<Vec<ToolCallRecord>> {
        // Partition calls by concurrency mode.
        let mut concurrent_calls: Vec<&ToolCallRequest> = Vec::new();
        let mut exclusive_calls: Vec<&ToolCallRequest> = Vec::new();

        for call in calls {
            let mode = Self::tool_concurrency_mode(call, agent);
            match mode {
                ConcurrencyMode::ReadOnly | ConcurrencyMode::Safe => {
                    concurrent_calls.push(call);
                }
                ConcurrencyMode::Exclusive => {
                    exclusive_calls.push(call);
                }
            }
        }

        let mut all_records: Vec<(usize, ToolCallRecord)> = Vec::with_capacity(calls.len());

        // Phase 1: Run concurrent calls in bounded batches.
        if !concurrent_calls.is_empty() {
            let concurrency = max_concurrency.unwrap_or(concurrent_calls.len()).max(1);
            for chunk in concurrent_calls.chunks(concurrency) {
                let mut futs = Vec::with_capacity(chunk.len());
                for &call in chunk {
                    let idx = calls.iter().position(|c| c.id == call.id).unwrap_or(0);
                    let fut = async move {
                        let record =
                            Self::execute_single_tool(call, agent, context, hooks, agent_name, mw)
                                .await;
                        (idx, record)
                    };
                    futs.push(fut);
                }
                all_records.extend(futures::future::join_all(futs).await);
            }
        }

        // Phase 2: Run exclusive calls sequentially.
        for &call in &exclusive_calls {
            let idx = calls.iter().position(|c| c.id == call.id).unwrap_or(0);
            let record =
                Self::execute_single_tool(call, agent, context, hooks, agent_name, mw).await;
            all_records.push((idx, record));
        }

        // Sort by original call order for deterministic message ordering.
        all_records.sort_by_key(|(idx, _)| *idx);
        let records: Vec<ToolCallRecord> = all_records.into_iter().map(|(_, r)| r).collect();

        for record in &records {
            messages.push(Message::tool(&record.id, &record.result));
        }

        Ok(records)
    }

    /// Look up the concurrency mode for a tool call.
    fn tool_concurrency_mode(call: &ToolCallRequest, agent: &Agent) -> ConcurrencyMode {
        // Check regular tools first.
        if let Some(tool) = agent.tools.iter().find(|t| t.name() == call.name) {
            return tool.metadata().concurrency;
        }
        // Managed agents default to Safe (they run their own isolated sub-run).
        ConcurrencyMode::Safe
    }

    /// Execute a single tool call with lifecycle hooks, middleware, and tracing.
    async fn execute_single_tool(
        call: &ToolCallRequest,
        agent: &Agent,
        context: &RunContext,
        hooks: &dyn Hooks,
        agent_name: &str,
        mw: &[SharedMiddleware],
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
            // Run pre-tool middleware.
            if !mw.is_empty() {
                let mw_ctx = MiddlewareContext::new(context.clone(), agent_name.to_owned());
                match middleware::run_tool_call_middleware(mw, &mw_ctx, call).await {
                    Ok(ToolCallAction::Skip { result }) => {
                        return ToolCallRecord {
                            id: call.id.clone(),
                            name: call.name.clone(),
                            arguments: call.arguments.clone(),
                            result,
                            success: true,
                            sub_usage: Usage::zero(),
                        };
                    }
                    Ok(ToolCallAction::Replace { result }) => {
                        let result_str =
                            serde_json::to_string(&result).unwrap_or_else(|_| result.to_string());
                        return ToolCallRecord {
                            id: call.id.clone(),
                            name: call.name.clone(),
                            arguments: call.arguments.clone(),
                            result: result_str,
                            success: true,
                            sub_usage: Usage::zero(),
                        };
                    }
                    Ok(ToolCallAction::Execute) => {} // proceed normally
                    Err(err) => {
                        return ToolCallRecord {
                            id: call.id.clone(),
                            name: call.name.clone(),
                            arguments: call.arguments.clone(),
                            result: format!("Middleware rejected: {err}"),
                            success: false,
                            sub_usage: Usage::zero(),
                        };
                    }
                }
            }

            hooks.on_tool_start(context, agent_name, &call.name).await;

            let (result_str, success, sub_usage) =
                if let Some(sub) = agent.managed_agents.iter().find(|a| a.name == call.name) {
                    Self::dispatch_managed_agent(sub, &call.arguments).await
                } else if let Some(tool) = agent.tools.iter().find(|t| t.name() == call.name) {
                    let (r, s) = Self::dispatch_tool(tool, call).await;
                    (r, s, Usage::zero())
                } else {
                    warn!(tool = %call.name, "Tool not found");
                    (
                        format!("Tool '{}' not found", call.name),
                        false,
                        Usage::zero(),
                    )
                };

            // Run post-tool middleware.
            if !mw.is_empty() {
                let mw_ctx = MiddlewareContext::new(context.clone(), agent_name.to_owned());
                if let Err(err) = middleware::run_tool_result_middleware(
                    mw,
                    &mw_ctx,
                    &call.name,
                    &result_str,
                    success,
                )
                .await
                {
                    warn!(tool = %call.name, error = %err, "Post-tool middleware error");
                }
            }

            let current = tracing::Span::current();
            current.record("tool.success", success);
            current.record("tool.output", result_str.as_str());
            if !success {
                current.record("error", result_str.as_str());
            }
            hooks
                .on_tool_end(context, agent_name, &call.name, &result_str)
                .await;

            ToolCallRecord {
                id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
                result: result_str,
                success,
                sub_usage,
            }
        }
        .instrument(tool_span)
        .await
    }

    /// Dispatch a managed sub-agent with the given task arguments.
    ///
    /// Returns `(output, success, sub_agent_usage)` so the parent can
    /// accumulate the child's token consumption.
    async fn dispatch_managed_agent(sub_agent: &Agent, args: &Value) -> (String, bool, Usage) {
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
                (output, true, result.usage)
            }
            Err(e) => (
                format!("Managed agent '{}' failed: {e}", sub_agent.name),
                false,
                Usage::zero(),
            ),
        }
    }

    /// Dispatch a regular tool call via [`DynTool`](crate::tool::DynTool).
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

    /// Request human confirmation, returning `(confirmed, denied)`.
    ///
    /// `ApproveAll` responses are recorded in `auto_approved` for future calls.
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

    /// Merge input guardrails from agent and run config.
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

    /// Merge output guardrails from agent and run config.
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

    /// Run input guardrails sequentially; short-circuits on tripwire.
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

    /// Run output guardrails concurrently; short-circuits on tripwire.
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
