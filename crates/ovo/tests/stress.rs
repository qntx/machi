//! Stress suite: cancel flood, parallel tools, large journals.
#![allow(
    unused_crate_dependencies,
    reason = "integration binary links facade feature deps"
)]

#[cfg(test)]
mod stress {
    #![allow(
        clippy::expect_used,
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::missing_assert_message,
        clippy::excessive_nesting,
        reason = "stress tests prioritize clarity over pedantic style"
    )]

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use ovo::{
        DynTool, ErrorCode, InProcessHost, Journal, LlmSampler, MockSampler, OvoError,
        PrometheusRecorder, SampleRequest, SampleResponse, SessionHost, SharedMetrics, SpawnOpts,
        ToolCallContext, ToolDispatch, ToolMetadata, ToolRegistry, ToolResult, WorkflowOutcome,
        WorkflowRunParams, emit_catalogue_smoke, metric_catalogue_snapshot, request_hash,
        run_workflow_on_host, span_catalogue_snapshot,
    };
    use serde_json::json;
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    #[test]
    fn catalogues_are_nonempty_contract() {
        assert!(metric_catalogue_snapshot().lines().count() >= 11);
        assert!(span_catalogue_snapshot().lines().count() >= 9);
    }

    #[test]
    fn prometheus_smoke_covers_kernel_series() {
        let p = PrometheusRecorder::new();
        emit_catalogue_smoke(&p);
        let text = p.render();
        for line in metric_catalogue_snapshot().lines() {
            assert!(
                text.contains(line) || p.series_names().iter().any(|n| n == line),
                "missing series {line} in\n{text}"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn cancel_flood_spawn_agents() {
        let sampler = Arc::new(MockSampler::new());
        // Enough replies if any slip through before cancel check.
        for i in 0..64 {
            sampler.map_user_text(format!("t{i}"), "x");
        }
        let host = Arc::new(
            InProcessHost::new(sampler, Vec::new())
                .with_agent_budget(128)
                .with_max_concurrent_children(Some(64)),
        );

        let mut handles = Vec::with_capacity(64);
        for i in 0..64 {
            let host = Arc::clone(&host);
            let cancel = CancellationToken::new();
            cancel.cancel();
            handles.push(tokio::spawn(async move {
                host.spawn_agent(SpawnOpts::new(format!("t{i}")).with_cancel(cancel))
                    .await
            }));
        }

        let mut cancelled = 0usize;
        let mut other_err = 0usize;
        for h in handles {
            match h.await.expect("join") {
                Err(e) if e.code() == ErrorCode::HostCancelled => cancelled += 1,
                Err(_) => other_err += 1,
                Ok(r) if r.cancelled => cancelled += 1,
                Ok(_) => other_err += 1,
            }
        }
        assert_eq!(other_err, 0, "unexpected non-cancel outcomes");
        assert_eq!(cancelled, 64);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn parallel_64_readonly_tools() {
        struct UnitTool {
            name: String,
        }

        #[async_trait]
        impl DynTool for UnitTool {
            fn name(&self) -> &str {
                &self.name
            }
            fn description(&self) -> &str {
                "unit"
            }
            fn parameters(&self) -> serde_json::Value {
                json!({"type":"object","properties":{}})
            }
            fn metadata(&self) -> ToolMetadata {
                ToolMetadata::read_only()
            }
            async fn call(
                &self,
                _ctx: ToolCallContext,
                _arguments: serde_json::Value,
            ) -> Result<ToolResult, OvoError> {
                Ok(ToolResult::text("ok"))
            }
        }

        let tools: Vec<Arc<dyn DynTool>> = (0..64)
            .map(|i| {
                let tool: Arc<dyn DynTool> = Arc::new(UnitTool {
                    name: format!("t{i}"),
                });
                tool
            })
            .collect();
        let reg = ToolRegistry::from_tools(tools);
        let requests: Vec<_> = (0..64)
            .map(|i| ovo::DispatchRequest {
                call: ovo::ToolCall {
                    id: ovo::ToolCallId::new(format!("c{i}")).expect("id"),
                    name: format!("t{i}"),
                    arguments: json!({}),
                },
            })
            .collect();

        let outs = ToolDispatch::default()
            .with_max_concurrency(64)
            .execute_batch(&reg, ToolCallContext::default(), requests)
            .await;
        assert_eq!(outs.len(), 64);
        assert!(outs.iter().all(|o| o.result.is_ok()));
    }

    #[test]
    fn journal_1k_durable_round_trip() {
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("j1k.jsonl");
        {
            let mut j = Journal::new(Some(path.clone()));
            for i in 0..1000u64 {
                let payload = json!({"i": i});
                let hash = request_hash("spawn_agent", &payload);
                j.record(i, "spawn_agent", hash, json!({"ok": i}))
                    .expect("record");
            }
            assert_eq!(j.len(), 1000);
        }
        let loaded = Journal::load(path).expect("load");
        assert_eq!(loaded.len(), 1000);
        let hash = request_hash("spawn_agent", &json!({"i": 999}));
        let v = loaded
            .replay(999, "spawn_agent", &hash)
            .expect("replay")
            .expect("hit");
        assert_eq!(v.get("ok").and_then(serde_json::Value::as_u64), Some(999));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn workflow_resume_after_many_journaled_agents() {
        // Smaller than 1k for wall-time, but exercises multi-entry durable resume.
        const N: usize = 32;

        struct CountMock {
            inner: MockSampler,
            calls: AtomicUsize,
        }

        #[async_trait]
        impl LlmSampler for CountMock {
            async fn sample(&self, request: SampleRequest) -> Result<SampleResponse, OvoError> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                self.inner.sample(request).await
            }
        }

        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("many.jsonl");
        let sampler = Arc::new(CountMock {
            inner: MockSampler::new(),
            calls: AtomicUsize::new(0),
        });
        for i in 0..N {
            sampler
                .inner
                .map_user_text(format!("p{i}"), format!("o{i}"));
        }

        let host: Arc<dyn SessionHost> = Arc::new(InProcessHost::new(sampler.clone(), Vec::new()));
        let mut agent_calls = String::new();
        for i in 0..N {
            agent_calls.push_str("outs.push(agent(\"p");
            agent_calls.push_str(&i.to_string());
            agent_calls.push_str("\", #{ label: \"l");
            agent_calls.push_str(&i.to_string());
            agent_calls.push_str("\" }));\n");
        }
        let mut script = String::from(
            "let meta = #{ name: \"stress-many\", description: \"many agents\" };\nlet outs = [];\n",
        );
        script.push_str(&agent_calls);
        script.push_str("complete(#{ outs: outs });\n");

        let (tx, _rx) = mpsc::unbounded_channel();
        let o1 = run_workflow_on_host(
            Arc::clone(&host),
            WorkflowRunParams {
                script: script.clone(),
                args: json!({}),
                journal: Journal::new(Some(path.clone())),
                host_tx: tx,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            },
            Some(128),
        )
        .await
        .expect("run1");
        assert!(matches!(o1, WorkflowOutcome::Completed { .. }));
        let first = sampler.calls.load(Ordering::SeqCst);
        assert_eq!(first, N);

        let journal = Journal::load(path).expect("load");
        assert_eq!(journal.len(), N);
        let (tx2, _rx2) = mpsc::unbounded_channel();
        let o2 = run_workflow_on_host(
            host,
            WorkflowRunParams {
                script,
                args: json!({}),
                journal,
                host_tx: tx2,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            },
            Some(128),
        )
        .await
        .expect("run2");
        assert!(matches!(o2, WorkflowOutcome::Completed { .. }));
        assert_eq!(sampler.calls.load(Ordering::SeqCst), first);
    }

    /// 64 concurrent spawns at depth 0 with `max_spawn_depth` 8.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_64_spawns_depth_cap_8() {
        let sampler = Arc::new(MockSampler::new());
        for i in 0..64 {
            sampler.map_user_text(format!("job{i}"), format!("ok{i}"));
        }
        let host = Arc::new(
            InProcessHost::new(sampler, Vec::new())
                .with_agent_budget(128)
                .with_max_spawn_depth(Some(8))
                .with_max_concurrent_children(Some(64)),
        );
        let mut handles = Vec::with_capacity(64);
        for i in 0..64 {
            let host = Arc::clone(&host);
            handles.push(tokio::spawn(async move {
                host.spawn_agent(SpawnOpts::new(format!("job{i}")).with_depth(0))
                    .await
            }));
        }
        let mut ok = 0usize;
        for h in handles {
            let run = h.await.expect("join").expect("spawn");
            assert!(run.success && !run.cancelled);
            ok += 1;
        }
        assert_eq!(ok, 64);
        assert_eq!(host.agents_spent(), 64);

        // Depth must be < max: 8 is rejected when max is 8.
        let err = host
            .spawn_agent(SpawnOpts::new("deep").with_depth(8))
            .await
            .expect_err("depth");
        assert_eq!(err.code(), ErrorCode::HostDepth);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn depth_7_ok_under_max_8() {
        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("d7", "ok");
        let host = InProcessHost::new(sampler, Vec::new()).with_max_spawn_depth(Some(8));
        let run = host
            .spawn_agent(SpawnOpts::new("d7").with_depth(7))
            .await
            .expect("depth 7");
        assert!(run.success);
    }

    /// journal load rejects files larger than `MAX_JOURNAL_BYTES` (64 MiB).
    #[test]
    fn journal_load_rejects_over_byte_cap() {
        use std::fs::OpenOptions;
        use std::io::{Seek, SeekFrom, Write};

        use ovo::{Journal, JournalError, MAX_JOURNAL_BYTES, workflow::JOURNAL_VERSION_HEADER};

        assert_eq!(MAX_JOURNAL_BYTES, 64 * 1024 * 1024);

        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("cap.jsonl");
        {
            let mut f = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&path)
                .expect("create");
            writeln!(f, "{JOURNAL_VERSION_HEADER}").expect("hdr");
            f.seek(SeekFrom::Start(MAX_JOURNAL_BYTES)).expect("seek");
            f.write_all(b"x").expect("grow past cap");
        }
        let err = Journal::load(path).expect_err("oversize must fail");
        let ok = match &err {
            JournalError::UnsafeRestore { limit, reason } => {
                *limit == MAX_JOURNAL_BYTES
                    && (reason.contains("exceeds") || reason.contains("byte"))
            }
            JournalError::Io(e) => {
                let msg = e.to_string();
                msg.contains("exceeds") || msg.contains("byte")
            }
            _ => false,
        };
        assert!(ok, "expected oversize restore/io reject, got {err:?}");
    }

    /// torn-write fuzz — incomplete tails of varying lengths are repaired.
    #[test]
    fn journal_torn_write_fuzz_tails() {
        use std::fs;
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("torn.jsonl");
        {
            let mut j = Journal::new(Some(path.clone()));
            for i in 0..20u64 {
                let payload = json!({"i": i, "pad": "x".repeat(64)});
                let hash = request_hash("spawn_agent", &payload);
                j.record(i, "spawn_agent", hash, json!({"ok": i}))
                    .expect("record");
            }
        }
        let intact = fs::read(&path).expect("read");
        // Corrupt by appending incomplete JSON prefixes of many lengths.
        for cut in [1usize, 3, 7, 15, 31, 63, 127, 255] {
            let mut raw = intact.clone();
            raw.extend_from_slice(
                format!("{{\"kind\":\"partial\",\"x\":{}", "y".repeat(cut)).as_bytes(),
            );
            fs::write(&path, &raw).expect("write");
            let loaded = Journal::load(path.clone()).expect("load torn");
            assert_eq!(loaded.len(), 20, "cut={cut}");
            // rewrite intact for next cut
            fs::write(&path, &intact).expect("restore");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn turn_path_records_into_prometheus() {
        use ovo::{AgentBuilder, Session, TurnInput, TurnOptions, VecConversationState};

        let prom = Arc::new(PrometheusRecorder::new());
        let metrics: SharedMetrics = prom.clone();
        let sampler = Arc::new(MockSampler::new());
        sampler.push_text("hello");
        let agent = AgentBuilder::named("a")
            .model("mock")
            .build()
            .expect("agent");
        let mut state = VecConversationState::new();
        let mut session = Session::new();
        session
            .run_turn_with_metrics(
                &agent,
                sampler.as_ref(),
                &mut state,
                TurnInput::Text("hi".into()),
                TurnOptions::default().with_metrics(metrics),
                prom.as_ref(),
            )
            .await
            .expect("turn");
        let text = prom.render();
        assert!(text.contains("ovo_turns_total"), "{text}");
        assert!(text.contains("# TYPE"));
    }
}
