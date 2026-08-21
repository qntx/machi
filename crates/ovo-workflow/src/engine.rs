//! Rhai workflow engine with journaled host calls.

use std::sync::{Arc, Mutex};

use rhai::{Dynamic, Engine, EvalAltResult, Position};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::host::{AgentOpts, HostError, WorkflowHostRequest};
use crate::journal::{
    Journal, JournalError, host_error_message, host_error_sentinel, request_hash,
};
use crate::run::{PauseKind, WorkflowOutcome};
use crate::{MAX_HOST_CALLS, MAX_PARALLEL};

/// Rhai expression nesting limits (statement depth, expression depth).
const RHAI_MAX_EXPR_DEPTH_STMT: usize = 128;
const RHAI_MAX_EXPR_DEPTH_EXPR: usize = 64;
/// Max Rhai string size (bytes).
const RHAI_MAX_STRING_SIZE: usize = 16 * 1024 * 1024;
/// Max Rhai array / map length.
const RHAI_MAX_ARRAY_MAP_SIZE: usize = 64 * 1024;

/// Parameters for [`run_workflow`].
#[derive(Debug)]
pub struct WorkflowRunParams {
    /// Script source.
    pub script: String,
    /// Bound `args` global.
    pub args: serde_json::Value,
    /// Journal (may already contain entries for resume).
    pub journal: Journal,
    /// Host channel.
    pub host_tx: mpsc::UnboundedSender<WorkflowHostRequest>,
    /// Cancellation.
    pub cancel: CancellationToken,
    /// Max Rhai operations.
    pub max_ops: u64,
}

impl WorkflowRunParams {
    /// Default max ops.
    pub const DEFAULT_MAX_OPS: u64 = 100_000_000;
}

#[derive(Debug, Clone)]
enum ControlToken {
    Complete(serde_json::Value),
    Pause(PauseKind, String),
    Budget(String),
    Cancelled,
    Fatal(String),
}

struct Ctx {
    host_tx: mpsc::UnboundedSender<WorkflowHostRequest>,
    journal: Journal,
    seq: u64,
}

impl Ctx {
    fn next_seq(&mut self) -> Result<u64, Box<EvalAltResult>> {
        if self.seq >= MAX_HOST_CALLS {
            return Err(terminated(ControlToken::Fatal(
                "workflow exceeded max host calls".into(),
            )));
        }
        let seq = self.seq;
        self.seq += 1;
        Ok(seq)
    }
}

type ScriptResult<T> = Result<T, Box<EvalAltResult>>;

/// Run a workflow script to a terminal outcome.
#[must_use]
pub fn run_workflow(params: WorkflowRunParams) -> WorkflowOutcome {
    let WorkflowRunParams {
        script,
        args,
        journal,
        host_tx,
        cancel,
        max_ops,
    } = params;

    let ctx = Arc::new(Mutex::new(Ctx {
        host_tx,
        journal,
        seq: 0,
    }));

    let mut engine = Engine::new();
    engine.set_max_operations(max_ops);
    engine.set_max_call_levels(64);
    engine.set_max_expr_depths(RHAI_MAX_EXPR_DEPTH_STMT, RHAI_MAX_EXPR_DEPTH_EXPR);
    engine.set_max_string_size(RHAI_MAX_STRING_SIZE);
    engine.set_max_array_size(RHAI_MAX_ARRAY_MAP_SIZE);
    engine.set_max_map_size(RHAI_MAX_ARRAY_MAP_SIZE);
    engine.set_module_resolver(rhai::module_resolvers::DummyModuleResolver::new());
    engine.disable_symbol("eval");
    engine.register_fn("timestamp", || -> ScriptResult<()> {
        Err(runtime_error(
            "timestamp() is unavailable: workflows must be deterministic",
        ))
    });
    engine.register_fn("sleep", |_s: i64| -> ScriptResult<()> {
        Err(runtime_error("sleep() is unavailable in workflow scripts"))
    });

    let cancel_flag = cancel.clone();
    engine.on_progress(move |_| {
        if cancel_flag.is_cancelled() {
            Some(Dynamic::from(ControlToken::Cancelled))
        } else {
            None
        }
    });

    register_fns(&mut engine, &ctx);

    let mut scope = rhai::Scope::new();
    let args_dyn = match rhai::serde::to_dynamic(&args) {
        Ok(d) => d,
        Err(e) => {
            return WorkflowOutcome::Failed {
                error: format!("invalid args: {e}"),
            };
        }
    };
    scope.push_dynamic("args", args_dyn);

    match engine.eval_with_scope::<Dynamic>(&mut scope, &script) {
        Ok(value) => WorkflowOutcome::Completed {
            result: dynamic_to_value(value),
        },
        Err(err) => outcome_from_error(*err),
    }
}

fn register_fns(engine: &mut Engine, ctx: &Arc<Mutex<Ctx>>) {
    register_agent_fns(engine, ctx);
    register_notify_fns(engine, ctx);
    register_io_fns(engine, ctx);
    register_control_fns(engine, ctx);
}

fn register_agent_fns(engine: &mut Engine, ctx: &Arc<Mutex<Ctx>>) {
    let c = Arc::clone(ctx);
    engine.register_fn("agent", move |prompt: &str| -> ScriptResult<Dynamic> {
        spawn_agent(
            &c,
            AgentOpts {
                prompt: prompt.to_owned(),
                ..AgentOpts::default()
            },
        )
    });

    let c = Arc::clone(ctx);
    engine.register_fn(
        "agent",
        move |prompt: &str, opts: rhai::Map| -> ScriptResult<Dynamic> {
            let mut agent_opts = agent_opts_from_map(opts)?;
            if agent_opts.prompt.is_empty() {
                agent_opts.prompt = prompt.to_owned();
            }
            spawn_agent(&c, agent_opts)
        },
    );

    let c = Arc::clone(ctx);
    engine.register_fn(
        "parallel",
        move |items: rhai::Array| -> ScriptResult<rhai::Array> { spawn_agents_parallel(&c, items) },
    );
}

fn register_notify_fns(engine: &mut Engine, ctx: &Arc<Mutex<Ctx>>) {
    let c = Arc::clone(ctx);
    engine.register_fn("phase", move |title: &str| {
        fire_notify(&c, |replayed| WorkflowHostRequest::Phase {
            title: title.to_owned(),
            replayed,
        });
    });

    let c = Arc::clone(ctx);
    engine.register_fn("log", move |message: &str| {
        fire_notify(&c, |replayed| WorkflowHostRequest::Log {
            message: message.to_owned(),
            replayed,
        });
    });

    // `print` / `debug` map to the same host log channel (deterministic, non-journaled).
    let c = Arc::clone(ctx);
    engine.register_fn("print", move |message: &str| {
        fire_notify(&c, |replayed| WorkflowHostRequest::Log {
            message: message.to_owned(),
            replayed,
        });
    });

    let c = Arc::clone(ctx);
    engine.register_fn("debug", move |message: &str| {
        fire_notify(&c, |replayed| WorkflowHostRequest::Log {
            message: format!("debug: {message}"),
            replayed,
        });
    });

    let c = Arc::clone(ctx);
    engine.register_fn("telemetry_event", move |name: &str, fields: rhai::Map| {
        fire_notify(&c, |replayed| WorkflowHostRequest::Telemetry {
            name: name.to_owned(),
            fields: dynamic_to_value(Dynamic::from_map(fields)),
            replayed,
        });
    });
}

fn fire_notify(ctx: &Arc<Mutex<Ctx>>, make: impl FnOnce(bool) -> WorkflowHostRequest) {
    let (tx, replaying) = {
        let g = ctx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (g.host_tx.clone(), g.journal.covers(g.seq))
    };
    let _ = tx.send(make(replaying));
}

fn register_io_fns(engine: &mut Engine, ctx: &Arc<Mutex<Ctx>>) {
    let c = Arc::clone(ctx);
    engine.register_fn(
        "write_scratch_file",
        move |name: &str, content: &str| -> ScriptResult<String> {
            host_string_call(
                &c,
                "write_scratch_file",
                serde_json::json!({ "name": name, "content": content }),
                |reply| WorkflowHostRequest::WriteScratchFile {
                    name: name.to_owned(),
                    content: content.to_owned(),
                    reply,
                },
            )
        },
    );

    let c = Arc::clone(ctx);
    engine.register_fn(
        "read_scratch_file",
        move |name: &str| -> ScriptResult<String> {
            host_string_call(
                &c,
                "read_scratch_file",
                serde_json::json!({ "name": name }),
                |reply| WorkflowHostRequest::ReadScratchFile {
                    name: name.to_owned(),
                    reply,
                },
            )
        },
    );

    let c = Arc::clone(ctx);
    engine.register_fn(
        "render_template",
        move |name: &str, vars: Dynamic| -> ScriptResult<String> {
            let vars_v = dynamic_to_value(vars);
            let name_owned = name.to_owned();
            host_string_call(
                &c,
                "render_template",
                serde_json::json!({ "name": name_owned, "vars": vars_v }),
                move |reply| WorkflowHostRequest::RenderTemplate {
                    name: name_owned,
                    vars: vars_v,
                    reply,
                },
            )
        },
    );

    let c = Arc::clone(ctx);
    engine.register_fn(
        "git_diff_since",
        move |commit: &str| -> ScriptResult<String> {
            host_string_call(
                &c,
                "git_diff_since",
                serde_json::json!({ "commit": commit }),
                |reply| WorkflowHostRequest::GitDiffSince {
                    commit: commit.to_owned(),
                    reply,
                },
            )
        },
    );

    let c = Arc::clone(ctx);
    engine.register_fn("budget", move || -> ScriptResult<Dynamic> {
        let (reply_tx, reply_rx) = oneshot::channel();
        {
            let g = c.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            g.host_tx
                .send(WorkflowHostRequest::BudgetQuery { reply: reply_tx })
                .map_err(|_| {
                    terminated(ControlToken::Fatal("workflow host channel closed".into()))
                })?;
        }
        let state = reply_rx
            .blocking_recv()
            .map_err(|_| terminated(ControlToken::Fatal("workflow host dropped reply".into())))?
            .map_err(|e| runtime_error(e.to_string()))?;
        let value = serde_json::to_value(state).unwrap_or(serde_json::Value::Null);
        value_to_dynamic(&value)
    });
}

fn register_control_fns(engine: &mut Engine, ctx: &Arc<Mutex<Ctx>>) {
    engine.register_fn("json_encode", |value: Dynamic| -> ScriptResult<String> {
        serde_json::to_string(&dynamic_to_value(value))
            .map_err(|e| runtime_error(format!("json_encode failed: {e}")))
    });

    engine.register_fn("fingerprint", |text: &str| -> String {
        // Pure deterministic fingerprint (16-byte hex of SHA-256).
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(text.as_bytes());
        encode_hex16(digest.iter().take(16).copied())
    });

    engine.register_fn("complete", |value: Dynamic| -> ScriptResult<()> {
        Err(terminated(ControlToken::Complete(dynamic_to_value(value))))
    });

    engine.register_fn("complete", || -> ScriptResult<()> {
        Err(terminated(ControlToken::Complete(serde_json::Value::Null)))
    });

    engine.register_fn("pause", |kind: &str, message: &str| -> ScriptResult<()> {
        let kind = match kind {
            "user" => PauseKind::User,
            "back_off" | "backoff" => PauseKind::BackOff,
            "no_progress" => PauseKind::NoProgress,
            "verification" | "blocked" => PauseKind::Verification,
            "infra" => PauseKind::Infra,
            other => {
                return Err(runtime_error(format!("unknown pause kind: {other}")));
            }
        };
        Err(terminated(ControlToken::Pause(kind, message.to_owned())))
    });

    // Journaled pause: first run records then pauses; resume skips.
    let c = Arc::clone(ctx);
    engine.register_fn(
        "await_user",
        move |kind: &str, message: &str| -> ScriptResult<()> {
            let payload = serde_json::json!({ "kind": kind, "message": message });
            let hash = request_hash("await_user", &payload);
            let seq = {
                let mut g = c.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                g.next_seq()?
            };
            {
                let g = c.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                if g.journal
                    .replay(seq, "await_user", &hash)
                    .map_err(journal_fatal)?
                    .is_some()
                {
                    return Ok(());
                }
            }
            {
                let mut g = c.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                g.journal
                    .record(seq, "await_user", hash, serde_json::Value::Null)
                    .map_err(journal_fatal)?;
            }
            let pause_kind = match kind {
                "user" => PauseKind::User,
                "back_off" | "backoff" => PauseKind::BackOff,
                "no_progress" => PauseKind::NoProgress,
                "verification" | "blocked" => PauseKind::Verification,
                "infra" => PauseKind::Infra,
                _ => PauseKind::User,
            };
            Err(terminated(ControlToken::Pause(
                pause_kind,
                message.to_owned(),
            )))
        },
    );
}

enum ParallelSlot {
    Replayed(serde_json::Value),
    Pending {
        opts: Box<AgentOpts>,
        seq: u64,
        hash: String,
    },
    Live {
        seq: u64,
        hash: String,
        reply_rx: oneshot::Receiver<Result<crate::host::AgentResult, HostError>>,
    },
}

/// Fan-out: reserve live slots once, dispatch all host spawns, wait as a barrier.
#[allow(
    clippy::too_many_lines,
    clippy::excessive_nesting,
    reason = "parallel barrier + budget conservation is intentionally collocated"
)]
fn spawn_agents_parallel(ctx: &Arc<Mutex<Ctx>>, items: rhai::Array) -> ScriptResult<rhai::Array> {
    if items.len() > MAX_PARALLEL {
        return Err(runtime_error(format!(
            "parallel() accepts at most {MAX_PARALLEL} items"
        )));
    }

    let mut prepared: Vec<(AgentOpts, String, u64)> = Vec::with_capacity(items.len());
    for item in items {
        let map = item
            .try_cast::<rhai::Map>()
            .ok_or_else(|| runtime_error("parallel() items must be maps"))?;
        let opts = agent_opts_from_map(map)?;
        let payload = serde_json::to_value(&opts)
            .map_err(|e| runtime_error(format!("invalid agent options: {e}")))?;
        let hash = request_hash("spawn_agent", &payload);
        let seq = {
            let mut g = ctx
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            g.next_seq()?
        };
        prepared.push((opts, hash, seq));
    }

    let mut pending: Vec<ParallelSlot> = Vec::with_capacity(prepared.len());
    let mut live_count = 0u64;
    for (opts, hash, seq) in prepared {
        let replayed = {
            let g = ctx
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            g.journal
                .replay(seq, "spawn_agent", &hash)
                .map_err(journal_fatal)?
        };
        if let Some(value) = replayed {
            pending.push(ParallelSlot::Replayed(value));
        } else {
            live_count = live_count.saturating_add(1);
            pending.push(ParallelSlot::Pending {
                opts: Box::new(opts),
                seq,
                hash,
            });
        }
    }

    // Reserve before any live spawn so budget failure never races partial fan-out.
    reserve_n(ctx, live_count)?;

    let mut slots: Vec<ParallelSlot> = Vec::with_capacity(pending.len());
    for slot in pending {
        match slot {
            ParallelSlot::Replayed(v) => slots.push(ParallelSlot::Replayed(v)),
            ParallelSlot::Pending { opts, seq, hash } => {
                let (reply_tx, reply_rx) = oneshot::channel();
                {
                    let g = ctx
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    g.host_tx
                        .send(WorkflowHostRequest::SpawnAgent {
                            opts: *opts,
                            reply: reply_tx,
                        })
                        .map_err(|_| {
                            terminated(ControlToken::Fatal("workflow host channel closed".into()))
                        })?;
                }
                slots.push(ParallelSlot::Live {
                    seq,
                    hash,
                    reply_rx,
                });
            }
            ParallelSlot::Live { .. } => {
                return Err(terminated(ControlToken::Fatal(
                    "internal: live slot before dispatch".into(),
                )));
            }
        }
    }

    // Ordered outputs; live slots journaled after the barrier unless resumable.
    let mut ordered: Vec<Option<Dynamic>> = Vec::with_capacity(slots.len());
    let mut live_to_journal: Vec<(usize, u64, String, serde_json::Value)> = Vec::new();
    let mut resumable_terminal: Option<Box<EvalAltResult>> = None;
    let mut first_catchable: Option<Box<EvalAltResult>> = None;
    // Reserved slots that returned Quota on spawn and must be released (dense Null journaled).
    let mut quota_release: u64 = 0;

    for (idx, slot) in slots.into_iter().enumerate() {
        match slot {
            ParallelSlot::Replayed(value) => {
                if let Some(msg) = host_error_message(&value) {
                    first_catchable.get_or_insert_with(|| runtime_error(msg.to_owned()));
                    ordered.push(None);
                    continue;
                }
                // Prior Quota journal entry replays as unit.
                if value.is_null() {
                    ordered.push(Some(Dynamic::UNIT));
                    continue;
                }
                ordered.push(Some(value_to_dynamic(&value)?));
            }
            ParallelSlot::Pending { .. } => {
                return Err(terminated(ControlToken::Fatal(
                    "internal: pending slot after dispatch".into(),
                )));
            }
            ParallelSlot::Live {
                seq,
                hash,
                reply_rx,
            } => {
                let reply = reply_rx.blocking_recv().map_err(|_| {
                    terminated(ControlToken::Fatal("workflow host dropped reply".into()))
                })?;
                match reply {
                    Ok(result) => {
                        let value = serde_json::to_value(result).unwrap_or(serde_json::Value::Null);
                        live_to_journal.push((idx, seq, hash, value));
                        ordered.push(None); // filled after journal
                    }
                    Err(HostError::BudgetExceeded) => {
                        resumable_terminal.get_or_insert(terminated(ControlToken::Budget(
                            "workflow agent budget exceeded".into(),
                        )));
                        ordered.push(None);
                    }
                    Err(HostError::Cancelled) => {
                        resumable_terminal.get_or_insert(terminated(ControlToken::Cancelled));
                        ordered.push(None);
                    }
                    Err(HostError::AgentCallQuotaExceeded { .. }) => {
                        // Journal Null to keep dense sequence numbers.
                        live_to_journal.push((idx, seq, hash, serde_json::Value::Null));
                        quota_release = quota_release.saturating_add(1);
                        ordered.push(None);
                    }
                    Err(HostError::Unsupported(msg) | HostError::Failed(msg)) => {
                        live_to_journal.push((idx, seq, hash, host_error_sentinel(&msg)));
                        first_catchable.get_or_insert_with(|| runtime_error(msg));
                        ordered.push(None);
                    }
                }
            }
        }
    }

    // Budget conservation: resumable terminal releases all live reservations and journals nothing.
    if let Some(err) = resumable_terminal {
        release_n(ctx, live_count);
        return Err(err);
    }

    // Journal in seq order so dense validation cannot fail on out-of-order collection.
    live_to_journal.sort_by_key(|(_, seq, _, _)| *seq);
    for (idx, seq, hash, value) in live_to_journal {
        let is_host_err = host_error_message(&value).is_some();
        let is_quota_null = value.is_null();
        {
            let mut g = ctx
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            g.journal
                .record(seq, "spawn_agent", hash, value.clone())
                .map_err(journal_fatal)?;
        }
        if let Some(slot) = ordered.get_mut(idx) {
            if is_host_err {
                // Propagate catchable error after journaling.
            } else if is_quota_null {
                *slot = Some(Dynamic::UNIT);
            } else {
                *slot = Some(value_to_dynamic(&value)?);
            }
        }
    }

    // Hosts that reject spawn with Quota without consuming reserved slots need release.
    if quota_release > 0 {
        release_n(ctx, quota_release);
    }

    if let Some(err) = first_catchable {
        return Err(err);
    }

    let mut results = rhai::Array::with_capacity(ordered.len());
    for item in ordered {
        results.push(item.unwrap_or(Dynamic::UNIT));
    }
    Ok(results)
}

fn map_spawn_reply_live(
    reply: Result<crate::host::AgentResult, HostError>,
    seq: u64,
    hash: String,
) -> Result<serde_json::Value, SpawnLiveError> {
    match reply {
        Ok(result) => Ok(serde_json::to_value(result).unwrap_or(serde_json::Value::Null)),
        Err(HostError::BudgetExceeded) => Err(SpawnLiveError::Resumable(terminated(
            ControlToken::Budget("workflow agent budget exceeded".into()),
        ))),
        Err(HostError::Cancelled) => Err(SpawnLiveError::Resumable(terminated(
            ControlToken::Cancelled,
        ))),
        Err(HostError::AgentCallQuotaExceeded { requested, maximum }) => {
            Err(SpawnLiveError::Catchable(runtime_error(format!(
                "workflow agent-call quota exceeded: requested {requested}, maximum {maximum}"
            ))))
        }
        Err(HostError::Unsupported(msg) | HostError::Failed(msg)) => {
            Err(SpawnLiveError::JournalThenCatchable { seq, hash, msg })
        }
    }
}

enum SpawnLiveError {
    Resumable(Box<EvalAltResult>),
    Catchable(Box<EvalAltResult>),
    JournalThenCatchable { seq: u64, hash: String, msg: String },
}

fn spawn_agent(ctx: &Arc<Mutex<Ctx>>, opts: AgentOpts) -> ScriptResult<Dynamic> {
    let payload = serde_json::to_value(&opts)
        .map_err(|e| runtime_error(format!("invalid agent options: {e}")))?;
    let hash = request_hash("spawn_agent", &payload);
    let seq = {
        let mut g = ctx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        g.next_seq()?
    };

    {
        let g = ctx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(recorded) = g
            .journal
            .replay(seq, "spawn_agent", &hash)
            .map_err(journal_fatal)?
        {
            if let Some(msg) = host_error_message(&recorded) {
                return Err(runtime_error(msg.to_owned()));
            }
            return value_to_dynamic(&recorded);
        }
    }

    reserve_one(ctx)?;

    let (reply_tx, reply_rx) = oneshot::channel();
    {
        let g = ctx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        g.host_tx
            .send(WorkflowHostRequest::SpawnAgent {
                opts,
                reply: reply_tx,
            })
            .map_err(|_| terminated(ControlToken::Fatal("workflow host channel closed".into())))?;
    }

    let reply = reply_rx
        .blocking_recv()
        .map_err(|_| terminated(ControlToken::Fatal("workflow host dropped reply".into())))?;

    match map_spawn_reply_live(reply, seq, hash.clone()) {
        Ok(value) => {
            {
                let mut g = ctx
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                g.journal
                    .record(seq, "spawn_agent", hash, value.clone())
                    .map_err(journal_fatal)?;
            }
            value_to_dynamic(&value)
        }
        Err(SpawnLiveError::Resumable(err)) => {
            // Budget conservation: release reserved slot; journal nothing.
            release_n(ctx, 1);
            Err(err)
        }
        Err(SpawnLiveError::Catchable(err)) => {
            release_n(ctx, 1);
            Err(err)
        }
        Err(SpawnLiveError::JournalThenCatchable { seq, hash, msg }) => {
            let sentinel = host_error_sentinel(&msg);
            {
                let mut g = ctx
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                g.journal
                    .record(seq, "spawn_agent", hash, sentinel)
                    .map_err(journal_fatal)?;
            }
            // Slot was spent on a failed host call that is journaled — do not release.
            Err(runtime_error(msg))
        }
    }
}

/// Journaled host RPC returning a string (scratch / template / …).
fn host_string_call<F>(
    ctx: &Arc<Mutex<Ctx>>,
    kind: &str,
    payload: serde_json::Value,
    make_req: F,
) -> ScriptResult<String>
where
    F: FnOnce(oneshot::Sender<Result<String, HostError>>) -> WorkflowHostRequest,
{
    let hash = request_hash(kind, &payload);
    let seq = {
        let mut g = ctx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        g.next_seq()?
    };

    {
        let g = ctx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(recorded) = g.journal.replay(seq, kind, &hash).map_err(journal_fatal)? {
            if let Some(msg) = host_error_message(&recorded) {
                return Err(runtime_error(msg.to_owned()));
            }
            return Ok(recorded
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| recorded.to_string()));
        }
    }

    let (reply_tx, reply_rx) = oneshot::channel();
    {
        let g = ctx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        g.host_tx
            .send(make_req(reply_tx))
            .map_err(|_| terminated(ControlToken::Fatal("workflow host channel closed".into())))?;
    }

    let reply = reply_rx
        .blocking_recv()
        .map_err(|_| terminated(ControlToken::Fatal("workflow host dropped reply".into())))?;

    let value = match reply {
        Ok(s) => s,
        Err(HostError::Cancelled) => return Err(terminated(ControlToken::Cancelled)),
        Err(HostError::BudgetExceeded | HostError::AgentCallQuotaExceeded { .. }) => {
            return Err(terminated(ControlToken::Budget(
                "workflow agent budget exceeded".into(),
            )));
        }
        Err(HostError::Unsupported(msg) | HostError::Failed(msg)) => {
            let sentinel = host_error_sentinel(&msg);
            {
                let mut g = ctx
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                g.journal
                    .record(seq, kind, hash, sentinel)
                    .map_err(journal_fatal)?;
            }
            return Err(runtime_error(msg));
        }
    };

    {
        let mut g = ctx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        g.journal
            .record(seq, kind, hash, serde_json::Value::String(value.clone()))
            .map_err(journal_fatal)?;
    }
    Ok(value)
}

fn reserve_one(ctx: &Arc<Mutex<Ctx>>) -> ScriptResult<()> {
    reserve_n(ctx, 1)
}

fn reserve_n(ctx: &Arc<Mutex<Ctx>>, count: u64) -> ScriptResult<()> {
    if count == 0 {
        return Ok(());
    }
    let (reply_tx, reply_rx) = oneshot::channel();
    {
        let g = ctx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        g.host_tx
            .send(WorkflowHostRequest::ReserveAgentCalls {
                count,
                reply: reply_tx,
            })
            .map_err(|_| terminated(ControlToken::Fatal("workflow host channel closed".into())))?;
    }
    match reply_rx
        .blocking_recv()
        .map_err(|_| terminated(ControlToken::Fatal("workflow host dropped reply".into())))?
    {
        Ok(()) => Ok(()),
        Err(HostError::BudgetExceeded | HostError::AgentCallQuotaExceeded { .. }) => {
            Err(terminated(ControlToken::Budget(
                "workflow agent budget exceeded".into(),
            )))
        }
        Err(HostError::Cancelled) => Err(terminated(ControlToken::Cancelled)),
        Err(HostError::Unsupported(msg) | HostError::Failed(msg)) => Err(runtime_error(msg)),
    }
}

/// Release reserved agent-call slots (budget conservation on resumable termination).
fn release_n(ctx: &Arc<Mutex<Ctx>>, count: u64) {
    if count == 0 {
        return;
    }
    let (reply_tx, reply_rx) = oneshot::channel();
    let send_ok = {
        let g = ctx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        g.host_tx
            .send(WorkflowHostRequest::ReleaseAgentCalls {
                count,
                reply: reply_tx,
            })
            .is_ok()
    };
    if send_ok {
        let _ = reply_rx.blocking_recv();
    }
}

fn encode_hex16(bytes: impl IntoIterator<Item = u8>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(32);
    for b in bytes {
        let hi = usize::from(b >> 4);
        let lo = usize::from(b & 0x0f);
        if let (Some(&h), Some(&l)) = (HEX.get(hi), HEX.get(lo)) {
            out.push(char::from(h));
            out.push(char::from(l));
        }
    }
    out
}

fn agent_opts_from_map(map: rhai::Map) -> ScriptResult<AgentOpts> {
    let value = rhai::serde::from_dynamic::<serde_json::Value>(&Dynamic::from_map(map))
        .map_err(|e| runtime_error(format!("invalid options map: {e}")))?;
    serde_json::from_value(value).map_err(|e| runtime_error(format!("invalid agent options: {e}")))
}

fn dynamic_to_value(d: Dynamic) -> serde_json::Value {
    rhai::serde::from_dynamic(&d).unwrap_or(serde_json::Value::Null)
}

fn value_to_dynamic(v: &serde_json::Value) -> ScriptResult<Dynamic> {
    rhai::serde::to_dynamic(v).map_err(|e| runtime_error(format!("host result conversion: {e}")))
}

fn terminated(token: ControlToken) -> Box<EvalAltResult> {
    Box::new(EvalAltResult::ErrorTerminated(
        Dynamic::from(token),
        Position::NONE,
    ))
}

fn runtime_error(message: impl Into<String>) -> Box<EvalAltResult> {
    Box::new(EvalAltResult::ErrorRuntime(
        Dynamic::from(message.into()),
        Position::NONE,
    ))
}

fn journal_fatal(error: JournalError) -> Box<EvalAltResult> {
    terminated(ControlToken::Fatal(error.to_string()))
}

fn find_control_token(err: &EvalAltResult) -> Option<ControlToken> {
    match err {
        EvalAltResult::ErrorTerminated(token, _) => token.clone().try_cast::<ControlToken>(),
        EvalAltResult::ErrorInFunctionCall(_, _, inner, _) => find_control_token(inner),
        EvalAltResult::ErrorInModule(_, inner, _) => find_control_token(inner),
        _ => None,
    }
}

fn outcome_from_error(err: EvalAltResult) -> WorkflowOutcome {
    if let Some(token) = find_control_token(&err) {
        return match token {
            ControlToken::Complete(result) => WorkflowOutcome::Completed { result },
            ControlToken::Pause(kind, message) => WorkflowOutcome::Paused { kind, message },
            ControlToken::Budget(message) => WorkflowOutcome::BudgetExceeded { message },
            ControlToken::Cancelled => WorkflowOutcome::Cancelled,
            ControlToken::Fatal(error) => WorkflowOutcome::Failed { error },
        };
    }
    WorkflowOutcome::Failed {
        error: err.to_string(),
    }
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::excessive_nesting,
    reason = "unit tests use expect/panic for setup"
)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;
    use crate::host::{AgentResult, WorkflowHostRequest};

    fn spawn_host(
        budget: u64,
    ) -> (
        mpsc::UnboundedSender<WorkflowHostRequest>,
        tokio::task::JoinHandle<()>,
        Arc<AtomicU64>,
    ) {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let spent = Arc::new(AtomicU64::new(0));
        let spent2 = spent.clone();
        let handle = tokio::task::spawn(async move {
            let mut reserved = 0u64;
            while let Some(req) = rx.recv().await {
                match req {
                    WorkflowHostRequest::ReserveAgentCalls { count, reply } => {
                        if reserved + count + spent2.load(Ordering::SeqCst) > budget {
                            let _ = reply.send(Err(HostError::BudgetExceeded));
                        } else {
                            reserved += count;
                            let _ = reply.send(Ok(()));
                        }
                    }
                    WorkflowHostRequest::ReleaseAgentCalls { count, reply } => {
                        reserved = reserved.saturating_sub(count);
                        let _ = reply.send(Ok(()));
                    }
                    WorkflowHostRequest::SpawnAgent { opts, reply } => {
                        spent2.fetch_add(1, Ordering::SeqCst);
                        reserved = reserved.saturating_sub(1);
                        let _ = reply.send(Ok(AgentResult {
                            agent_id: format!("a-{}", spent2.load(Ordering::SeqCst)),
                            success: true,
                            output: serde_json::json!({"prompt": opts.prompt}),
                            cancelled: false,
                            tokens_used: 1,
                            duration_ms: 1,
                        }));
                    }
                    WorkflowHostRequest::Phase { .. }
                    | WorkflowHostRequest::Log { .. }
                    | WorkflowHostRequest::Telemetry { .. } => {}
                    WorkflowHostRequest::BudgetQuery { reply } => {
                        let spent = spent2.load(Ordering::SeqCst);
                        let _ = reply.send(Ok(crate::host::BudgetState {
                            total: Some(budget),
                            spent,
                            reserved,
                            remaining: Some(budget.saturating_sub(spent + reserved)),
                        }));
                    }
                    WorkflowHostRequest::RenderTemplate { reply, .. }
                    | WorkflowHostRequest::WriteScratchFile { reply, .. }
                    | WorkflowHostRequest::ReadScratchFile { reply, .. }
                    | WorkflowHostRequest::GitDiffSince { reply, .. } => {
                        let _ = reply.send(Err(HostError::Unsupported(
                            "not implemented in test host".into(),
                        )));
                    }
                }
            }
        });
        (tx, handle, spent)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn agent_and_complete() {
        let (tx, host, spent) = spawn_host(10);
        let script = r#"
            let meta = #{ name: "t", description: "t" };
            phase("go");
            let r = agent("hello");
            complete(r);
        "#;
        let outcome = tokio::task::spawn_blocking(move || {
            run_workflow(WorkflowRunParams {
                script: script.into(),
                args: serde_json::json!({}),
                journal: Journal::new(None),
                host_tx: tx,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            })
        })
        .await
        .expect("join");
        drop(host);
        assert!(
            matches!(outcome, WorkflowOutcome::Completed { .. }),
            "{outcome:?}"
        );
        assert_eq!(spent.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn journal_resume_skips_first_agent() {
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("j.jsonl");
        let (tx, host, spent) = spawn_host(10);
        let script = r#"
            let meta = #{ name: "t", description: "t" };
            let a = agent("one");
            let b = agent("two");
            complete(#{ a: a, b: b });
        "#;
        let outcome1 = {
            let tx = tx.clone();
            let path = path.clone();
            let script = script.to_owned();
            tokio::task::spawn_blocking(move || {
                run_workflow(WorkflowRunParams {
                    script,
                    args: serde_json::json!({}),
                    journal: Journal::new(Some(path)),
                    host_tx: tx,
                    cancel: CancellationToken::new(),
                    max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
                })
            })
            .await
            .expect("join")
        };
        assert!(matches!(outcome1, WorkflowOutcome::Completed { .. }));
        assert_eq!(spent.load(Ordering::SeqCst), 2);

        // Resume: both agent calls should replay from journal (no new spawns).
        spent.store(0, Ordering::SeqCst);
        let journal = Journal::load(path).expect("load");
        assert_eq!(journal.len(), 2);
        let outcome2 = tokio::task::spawn_blocking(move || {
            run_workflow(WorkflowRunParams {
                script: script.into(),
                args: serde_json::json!({}),
                journal,
                host_tx: tx,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            })
        })
        .await
        .expect("join");
        drop(host);
        assert!(matches!(outcome2, WorkflowOutcome::Completed { .. }));
        assert_eq!(
            spent.load(Ordering::SeqCst),
            0,
            "resume must not re-spawn agents"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn json_encode_and_scratch_surface() {
        let (tx, host, _) = spawn_host(4);
        // Override unsupported scratch replies: use a host that implements scratch.
        drop(host);
        drop(tx);
        let (tx, mut rx) = mpsc::unbounded_channel::<WorkflowHostRequest>();
        let host = tokio::spawn(async move {
            let mut spent = 0u64;
            while let Some(req) = rx.recv().await {
                match req {
                    WorkflowHostRequest::ReserveAgentCalls { count, reply } => {
                        let _ = reply.send(Ok(()));
                        let _ = count;
                    }
                    WorkflowHostRequest::ReleaseAgentCalls { count, reply } => {
                        let _ = reply.send(Ok(()));
                        let _ = count;
                    }
                    WorkflowHostRequest::SpawnAgent { opts, reply } => {
                        spent += 1;
                        let _ = reply.send(Ok(AgentResult {
                            agent_id: format!("a-{spent}"),
                            success: true,
                            output: serde_json::json!({"echo": opts.prompt}),
                            cancelled: false,
                            tokens_used: 1,
                            duration_ms: 1,
                        }));
                    }
                    WorkflowHostRequest::BudgetQuery { reply } => {
                        let _ = reply.send(Ok(crate::host::BudgetState {
                            total: Some(4),
                            spent,
                            reserved: 0,
                            remaining: Some(4_u64.saturating_sub(spent)),
                        }));
                    }
                    WorkflowHostRequest::WriteScratchFile {
                        name,
                        content,
                        reply,
                    } => {
                        let _ = reply.send(Ok(format!("scratch/{name}:{content}")));
                    }
                    WorkflowHostRequest::ReadScratchFile { name, reply } => {
                        let _ = reply.send(Ok(format!("read:{name}")));
                    }
                    WorkflowHostRequest::Phase { .. }
                    | WorkflowHostRequest::Log { .. }
                    | WorkflowHostRequest::Telemetry { .. } => {}
                    WorkflowHostRequest::RenderTemplate { reply, .. }
                    | WorkflowHostRequest::GitDiffSince { reply, .. } => {
                        let _ = reply.send(Err(HostError::Unsupported("n/a".into())));
                    }
                }
            }
        });
        let script = r#"
            let meta = #{ name: "json", description: "encode" };
            let enc = json_encode(#{ a: 1, b: "x" });
            let path = write_scratch_file("n.txt", enc);
            let b = budget();
            complete(#{ enc: enc, path: path, remaining: b.remaining });
        "#;
        let outcome = tokio::task::spawn_blocking(move || {
            run_workflow(WorkflowRunParams {
                script: script.into(),
                args: serde_json::json!({}),
                journal: Journal::new(None),
                host_tx: tx,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            })
        })
        .await
        .expect("join");
        drop(host);
        match outcome {
            WorkflowOutcome::Completed { result } => {
                let enc = result.get("enc").and_then(|v| v.as_str()).expect("enc");
                assert!(enc.contains("\"a\":") || enc.contains("'a'"), "{enc}");
                assert!(
                    result
                        .get("path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .contains("n.txt"),
                    "{result}"
                );
            }
            other => panic!("expected completed: {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn budget_exceeded() {
        let (tx, host, _) = spawn_host(0);
        let script = r#"
            let meta = #{ name: "t", description: "t" };
            agent("x");
            complete(1);
        "#;
        let outcome = tokio::task::spawn_blocking(move || {
            run_workflow(WorkflowRunParams {
                script: script.into(),
                args: serde_json::json!({}),
                journal: Journal::new(None),
                host_tx: tx,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            })
        })
        .await
        .expect("join");
        drop(host);
        assert!(
            matches!(outcome, WorkflowOutcome::BudgetExceeded { .. }),
            "{outcome:?}"
        );
    }

    /// Host that can cancel or budget-fail the Nth spawn while tracking reserved slots.
    fn spawn_host_with_policy(
        budget: u64,
        fail_mode: &'static str,
    ) -> (
        mpsc::UnboundedSender<WorkflowHostRequest>,
        tokio::task::JoinHandle<()>,
        Arc<AtomicU64>,
        Arc<AtomicU64>,
    ) {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let spent = Arc::new(AtomicU64::new(0));
        let reserved_peak = Arc::new(AtomicU64::new(0));
        let spent2 = spent.clone();
        let reserved_peak2 = reserved_peak.clone();
        let handle = tokio::task::spawn(async move {
            let mut reserved = 0u64;
            while let Some(req) = rx.recv().await {
                match req {
                    WorkflowHostRequest::ReserveAgentCalls { count, reply } => {
                        if reserved + count + spent2.load(Ordering::SeqCst) > budget {
                            let _ = reply.send(Err(HostError::BudgetExceeded));
                        } else {
                            reserved += count;
                            reserved_peak2.fetch_max(reserved, Ordering::SeqCst);
                            let _ = reply.send(Ok(()));
                        }
                    }
                    WorkflowHostRequest::ReleaseAgentCalls { count, reply } => {
                        reserved = reserved.saturating_sub(count);
                        let _ = reply.send(Ok(()));
                    }
                    WorkflowHostRequest::SpawnAgent { opts, reply } => match fail_mode {
                        "cancel" => {
                            let _ = reply.send(Err(HostError::Cancelled));
                        }
                        "budget_on_spawn" => {
                            let _ = reply.send(Err(HostError::BudgetExceeded));
                        }
                        "fail" => {
                            let _ = reply.send(Err(HostError::Failed("host boom".into())));
                        }
                        _ => {
                            spent2.fetch_add(1, Ordering::SeqCst);
                            reserved = reserved.saturating_sub(1);
                            let _ = reply.send(Ok(AgentResult {
                                agent_id: format!("a-{}", spent2.load(Ordering::SeqCst)),
                                success: true,
                                output: serde_json::json!({"prompt": opts.prompt}),
                                cancelled: false,
                                tokens_used: 1,
                                duration_ms: 1,
                            }));
                        }
                    },
                    WorkflowHostRequest::Phase { .. }
                    | WorkflowHostRequest::Log { .. }
                    | WorkflowHostRequest::Telemetry { .. } => {}
                    WorkflowHostRequest::BudgetQuery { reply } => {
                        let spent = spent2.load(Ordering::SeqCst);
                        let _ = reply.send(Ok(crate::host::BudgetState {
                            total: Some(budget),
                            spent,
                            reserved,
                            remaining: Some(budget.saturating_sub(spent + reserved)),
                        }));
                    }
                    WorkflowHostRequest::RenderTemplate { reply, .. }
                    | WorkflowHostRequest::WriteScratchFile { reply, .. }
                    | WorkflowHostRequest::ReadScratchFile { reply, .. }
                    | WorkflowHostRequest::GitDiffSince { reply, .. } => {
                        let _ = reply.send(Err(HostError::Unsupported("n/a".into())));
                    }
                }
            }
        });
        (tx, handle, spent, reserved_peak)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_live_agent_releases_budget_so_resume_does_not_double_charge() {
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("j.jsonl");
        let (tx, host, spent, _) = spawn_host_with_policy(4, "cancel");
        let script = r#"
            let meta = #{ name: "t", description: "t" };
            agent("one");
            complete(1);
        "#;
        let outcome = {
            let tx = tx.clone();
            let path = path.clone();
            let script = script.to_owned();
            tokio::task::spawn_blocking(move || {
                run_workflow(WorkflowRunParams {
                    script,
                    args: serde_json::json!({}),
                    journal: Journal::new(Some(path)),
                    host_tx: tx,
                    cancel: CancellationToken::new(),
                    max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
                })
            })
            .await
            .expect("join")
        };
        assert!(matches!(outcome, WorkflowOutcome::Cancelled), "{outcome:?}");
        // Cancelled path must not journal the interrupted spawn.
        let journal = Journal::load(path.clone()).expect("load");
        assert_eq!(journal.len(), 0, "cancelled spawn must not be journaled");
        assert_eq!(spent.load(Ordering::SeqCst), 0);

        // Resume with a healthy host must succeed without double-charging.
        drop(host);
        let (tx2, host2, spent2) = spawn_host(4);
        let journal = Journal::load(path).expect("load2");
        let script = r#"
            let meta = #{ name: "t", description: "t" };
            agent("one");
            complete(1);
        "#;
        let outcome2 = tokio::task::spawn_blocking(move || {
            run_workflow(WorkflowRunParams {
                script: script.into(),
                args: serde_json::json!({}),
                journal,
                host_tx: tx2,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            })
        })
        .await
        .expect("join");
        drop(host2);
        assert!(
            matches!(outcome2, WorkflowOutcome::Completed { .. }),
            "{outcome2:?}"
        );
        assert_eq!(spent2.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn budget_exceeded_live_agent_releases_and_journals_nothing() {
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("j.jsonl");
        let (tx, host, _, _) = spawn_host_with_policy(4, "budget_on_spawn");
        let script = r#"
            let meta = #{ name: "t", description: "t" };
            agent("one");
            complete(1);
        "#;
        let path_run = path.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            run_workflow(WorkflowRunParams {
                script: script.into(),
                args: serde_json::json!({}),
                journal: Journal::new(Some(path_run)),
                host_tx: tx,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            })
        })
        .await
        .expect("join");
        drop(host);
        assert!(
            matches!(outcome, WorkflowOutcome::BudgetExceeded { .. }),
            "{outcome:?}"
        );
        let journal = Journal::load(path).expect("load");
        assert_eq!(journal.len(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn host_error_sentinel_journals_and_replays() {
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("j.jsonl");
        let (tx, host, _, _) = spawn_host_with_policy(4, "fail");
        let script = r#"
            let meta = #{ name: "t", description: "t" };
            agent("one");
            complete(1);
        "#;
        let outcome = {
            let tx = tx.clone();
            let path = path.clone();
            let script = script.to_owned();
            tokio::task::spawn_blocking(move || {
                run_workflow(WorkflowRunParams {
                    script,
                    args: serde_json::json!({}),
                    journal: Journal::new(Some(path)),
                    host_tx: tx,
                    cancel: CancellationToken::new(),
                    max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
                })
            })
            .await
            .expect("join")
        };
        drop(host);
        assert!(
            matches!(outcome, WorkflowOutcome::Failed { .. }),
            "{outcome:?}"
        );
        let journal = Journal::load(path.clone()).expect("load");
        assert_eq!(journal.len(), 1);
        assert!(
            journal
                .entries()
                .first()
                .is_some_and(|e| crate::journal::is_host_error_sentinel(&e.result)),
            "expected host-error sentinel"
        );

        // Resume re-raises without calling host spawn.
        let (tx2, host2, spent2) = spawn_host(4);
        let outcome2 = tokio::task::spawn_blocking(move || {
            run_workflow(WorkflowRunParams {
                script: script.into(),
                args: serde_json::json!({}),
                journal: Journal::load(path).expect("load2"),
                host_tx: tx2,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            })
        })
        .await
        .expect("join");
        drop(host2);
        assert!(
            matches!(outcome2, WorkflowOutcome::Failed { .. }),
            "{outcome2:?}"
        );
        assert_eq!(spent2.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn await_user_pauses_once_then_passes_on_resume() {
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("j.jsonl");
        let (tx, host, spent) = spawn_host(4);
        let script = r#"
            let meta = #{ name: "t", description: "t" };
            await_user("user", "needs a human");
            agent("after");
            complete(1);
        "#;
        let outcome1 = {
            let tx = tx.clone();
            let path = path.clone();
            let script = script.to_owned();
            tokio::task::spawn_blocking(move || {
                run_workflow(WorkflowRunParams {
                    script,
                    args: serde_json::json!({}),
                    journal: Journal::new(Some(path)),
                    host_tx: tx,
                    cancel: CancellationToken::new(),
                    max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
                })
            })
            .await
            .expect("join")
        };
        assert!(
            matches!(
                outcome1,
                WorkflowOutcome::Paused {
                    kind: PauseKind::User,
                    ..
                }
            ),
            "{outcome1:?}"
        );
        assert_eq!(spent.load(Ordering::SeqCst), 0);

        let journal = Journal::load(path).expect("load");
        assert_eq!(journal.len(), 1);
        let outcome2 = tokio::task::spawn_blocking(move || {
            run_workflow(WorkflowRunParams {
                script: script.into(),
                args: serde_json::json!({}),
                journal,
                host_tx: tx,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            })
        })
        .await
        .expect("join");
        drop(host);
        assert!(
            matches!(outcome2, WorkflowOutcome::Completed { .. }),
            "{outcome2:?}"
        );
        assert_eq!(spent.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_parallel_releases_budget() {
        let (tx, host, spent, _) = spawn_host_with_policy(8, "cancel");
        let script = r#"
            let meta = #{ name: "t", description: "t" };
            parallel([#{ prompt: "a" }, #{ prompt: "b" }]);
            complete(1);
        "#;
        let outcome = tokio::task::spawn_blocking(move || {
            run_workflow(WorkflowRunParams {
                script: script.into(),
                args: serde_json::json!({}),
                journal: Journal::new(None),
                host_tx: tx,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            })
        })
        .await
        .expect("join");
        drop(host);
        assert!(matches!(outcome, WorkflowOutcome::Cancelled), "{outcome:?}");
        assert_eq!(spent.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn parallel_quota_journals_null_and_keeps_dense_seq() {
        // Host: first spawn succeeds, second returns Quota, third succeeds.
        let (tx, mut rx) = mpsc::unbounded_channel::<WorkflowHostRequest>();
        let host = tokio::spawn(async move {
            let mut reserved = 0u64;
            let mut spawn_n = 0u64;
            while let Some(req) = rx.recv().await {
                match req {
                    WorkflowHostRequest::ReserveAgentCalls { count, reply } => {
                        reserved = reserved.saturating_add(count);
                        let _ = reply.send(Ok(()));
                    }
                    WorkflowHostRequest::ReleaseAgentCalls { count, reply } => {
                        reserved = reserved.saturating_sub(count);
                        let _ = reply.send(Ok(()));
                    }
                    WorkflowHostRequest::SpawnAgent { opts, reply } => {
                        spawn_n = spawn_n.saturating_add(1);
                        if spawn_n == 2 {
                            // Do not consume reserved — engine must release.
                            let _ = reply.send(Err(HostError::AgentCallQuotaExceeded {
                                requested: 1,
                                maximum: 0,
                            }));
                        } else {
                            reserved = reserved.saturating_sub(1);
                            let _ = reply.send(Ok(AgentResult {
                                agent_id: format!("a-{spawn_n}"),
                                success: true,
                                output: serde_json::json!({"prompt": opts.prompt}),
                                cancelled: false,
                                tokens_used: 1,
                                duration_ms: 1,
                            }));
                        }
                    }
                    WorkflowHostRequest::BudgetQuery { reply } => {
                        let _ = reply.send(Ok(crate::host::BudgetState {
                            total: Some(10),
                            spent: 0,
                            reserved,
                            remaining: Some(10),
                        }));
                    }
                    WorkflowHostRequest::Phase { .. }
                    | WorkflowHostRequest::Log { .. }
                    | WorkflowHostRequest::Telemetry { .. } => {}
                    WorkflowHostRequest::RenderTemplate { reply, .. }
                    | WorkflowHostRequest::WriteScratchFile { reply, .. }
                    | WorkflowHostRequest::ReadScratchFile { reply, .. }
                    | WorkflowHostRequest::GitDiffSince { reply, .. } => {
                        let _ = reply.send(Err(HostError::Unsupported("n/a".into())));
                    }
                }
            }
        });
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("quota.jsonl");
        let script = r#"
            let meta = #{ name: "q", description: "q" };
            let rows = parallel([
                #{ prompt: "a" },
                #{ prompt: "b" },
                #{ prompt: "c" }
            ]);
            complete(#{ n: rows.len() });
        "#;
        let path_run = path.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            run_workflow(WorkflowRunParams {
                script: script.into(),
                args: serde_json::json!({}),
                journal: Journal::new(Some(path_run)),
                host_tx: tx,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            })
        })
        .await
        .expect("join");
        drop(host);
        assert!(
            matches!(outcome, WorkflowOutcome::Completed { .. }),
            "{outcome:?}"
        );
        let journal = Journal::load(path).expect("load");
        assert_eq!(journal.len(), 3, "dense seq: success, null, success");
        assert!(
            journal.entries().get(1).is_some_and(|e| e.result.is_null()),
            "middle slot must be journaled null"
        );
    }

    #[test]
    fn fingerprint_is_deterministic() {
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(b"hello");
        let a = encode_hex16(digest.iter().take(16).copied());
        let b = encode_hex16(digest.iter().take(16).copied());
        assert_eq!(a, b);
        assert_eq!(a.len(), 32);
    }
}
