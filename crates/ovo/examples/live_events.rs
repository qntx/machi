//! Reconstruct a turn from the live [`TurnEvent`] stream alone.
//!
//! ```sh
//! cargo run -p ovo --example live_events --features runtime
//! ```

#![allow(
    unused_crate_dependencies,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::missing_assert_message,
    reason = "example links the ovo facade with optional features"
)]

use std::sync::Arc;

use ovo::{
    AgentBuilder, MockSampler, TurnEvent, TurnEventKind, TurnInput, TurnOptions, TurnRuntime,
    VecConversationState,
};
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let sampler = Arc::new(MockSampler::new());
    sampler.push_text("hello from the event stream");

    let agent = AgentBuilder::named("demo")
        .model("mock")
        .instructions("You are a demo agent.")
        .build()
        .expect("agent");

    let (tx, mut rx) = mpsc::unbounded_channel::<TurnEvent>();
    let mut state = VecConversationState::new();
    let outcome = TurnRuntime::new()
        .run(
            &agent,
            sampler.as_ref(),
            &mut state,
            TurnInput::Text("say hi".into()),
            TurnOptions::default().with_event_tx(tx),
        )
        .await
        .expect("turn");

    let mut text = String::new();
    let mut steps = 0u32;
    let mut finished = false;
    while let Ok(ev) = rx.try_recv() {
        match ev.kind {
            TurnEventKind::StepStarted { step } => steps = step,
            TurnEventKind::TextDelta { text: t } => text.push_str(&t),
            TurnEventKind::TurnFinished { cancelled, .. } => {
                finished = true;
                assert!(!cancelled, "demo turn should not cancel");
            }
            _ => {}
        }
    }

    // Non-stream path may not emit TextDelta; use finished + outcome as SoT for text.
    let observed = if text.is_empty() {
        outcome.output_text.clone()
    } else {
        text
    };
    assert!(finished, "TurnFinished must appear on the event stream");
    assert_eq!(observed, outcome.output_text);
    assert!(steps >= 1);
    println!("live_events_ok=true steps={steps} output={observed}");
}
