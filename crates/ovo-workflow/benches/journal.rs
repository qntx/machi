//! Smoke bench: journal record + replay + durable round-trip.
//!
//! ```bash
//! cargo bench -p ovo-workflow --bench journal
//! ```

#![allow(
    missing_docs,
    unused_crate_dependencies,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::missing_assert_message,
    reason = "bench harness: transitive deps + timing output"
)]

use std::hint::black_box;
use std::time::Instant;

use ovo_workflow::{Journal, request_hash};
use serde_json::json;
use tempfile::tempdir;

fn main() {
    bench_memory(5_000);
    bench_durable(2_000);
}

fn bench_memory(iters: u32) {
    let mut j = Journal::new(None);
    let start = Instant::now();
    for i in 0..iters {
        let payload = json!({"prompt": format!("p{i}")});
        let h = request_hash("spawn_agent", &payload);
        j.record(u64::from(i), "spawn_agent", h, json!({"ok": true}))
            .expect("record");
    }
    for i in 0..iters {
        let payload = json!({"prompt": format!("p{i}")});
        let h = request_hash("spawn_agent", &payload);
        let v = j
            .replay(u64::from(i), "spawn_agent", &h)
            .expect("replay")
            .expect("hit");
        black_box(v);
    }
    let elapsed = start.elapsed();
    println!(
        "journal_memory_record_replay: {iters} pairs in {elapsed:?} ({:.0} ops/s)",
        f64::from(iters * 2) / elapsed.as_secs_f64()
    );
}

fn bench_durable(iters: u32) {
    let dir = tempdir().expect("tmp");
    let path = dir.path().join("j.nljson");
    let mut j = Journal::new(Some(path.clone()));
    let start = Instant::now();
    for i in 0..iters {
        let payload = json!({"n": i});
        let h = request_hash("complete", &payload);
        j.record(u64::from(i), "complete", h, json!(i))
            .expect("rec");
    }
    let loaded = Journal::load(path).expect("load");
    assert_eq!(loaded.entries().len(), iters as usize);
    for i in 0..iters {
        let payload = json!({"n": i});
        let h = request_hash("complete", &payload);
        let v = loaded
            .replay(u64::from(i), "complete", &h)
            .expect("replay")
            .expect("hit");
        black_box(v);
    }
    let elapsed = start.elapsed();
    println!(
        "journal_durable_write_load_replay: {iters} entries in {elapsed:?} ({:.0} e/s)",
        f64::from(iters) / elapsed.as_secs_f64()
    );
}
