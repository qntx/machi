//! Streaming tool execution protocol.
//!
//! Invariant: a tool stream yields zero or more [`ToolStreamItem::Progress`]
//! items followed by **exactly one** [`ToolStreamItem::Terminal`].

use std::future::Future;
use std::pin::Pin;

use futures::Stream;
use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{ToolError, codes};
use crate::tool::ToolResult;

/// Opaque pinned stream of tool items.
pub type ToolStream = Pin<Box<dyn Stream<Item = ToolStreamItem> + Send>>;

/// One item in a tool stream.
#[derive(Debug)]
pub enum ToolStreamItem {
    /// Intermediate progress (logs, partial stdout, custom payloads).
    Progress(ToolProgress),
    /// Terminal result — always last.
    Terminal(Result<ToolResult, ToolError>),
}

impl ToolStreamItem {
    /// Whether this is the terminal item.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(self, Self::Terminal(_))
    }
}

/// Default max bytes per partial delta frame (16 KiB).
pub const MAX_DELTA_BYTES: usize = 16 * 1024;
/// Default max total frame/stream bytes (16 MiB).
pub const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// Progress payload shapes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolProgress {
    /// Free-form text chunk.
    Text {
        /// Chunk body.
        text: String,
    },
    /// Incremental partial output.
    Partial {
        /// UTF-8-safe delta slice.
        delta: String,
        /// Total bytes emitted so far (including this delta).
        total_bytes: u64,
        /// Whether this delta was truncated to the frame cap.
        truncated: bool,
        /// Byte gap skipped since last partial (e.g. after truncation).
        gap: u64,
    },
    /// Tool-defined progress.
    Custom {
        /// Stable producer discriminator.
        subkind: String,
        /// Arbitrary payload.
        payload: Value,
    },
}

impl ToolProgress {
    /// Text progress helper.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Partial progress helper.
    #[must_use]
    pub fn partial(delta: impl Into<String>, total_bytes: u64, truncated: bool, gap: u64) -> Self {
        Self::Partial {
            delta: delta.into(),
            total_bytes,
            truncated,
            gap,
        }
    }
}

/// Split `input` into UTF-8-safe partial progress frames.
///
/// Each frame's `delta` is at most `max_delta_bytes` and ends on a char boundary.
/// Stops once cumulative bytes would exceed `max_frame_bytes`.
#[must_use]
pub fn partial_progress_frames(
    input: &str,
    max_delta_bytes: usize,
    max_frame_bytes: usize,
) -> Vec<ToolProgress> {
    let max_delta = max_delta_bytes.max(1);
    let max_frame = max_frame_bytes.max(max_delta);
    let mut out = Vec::new();
    let mut offset = 0usize;
    let mut total: u64 = 0;
    let bytes = input.as_bytes();
    while offset < bytes.len() {
        if total >= u64::try_from(max_frame).unwrap_or(u64::MAX) {
            break;
        }
        let remaining_frame =
            max_frame.saturating_sub(usize::try_from(total).unwrap_or(usize::MAX));
        let want = max_delta
            .min(remaining_frame)
            .min(bytes.len().saturating_sub(offset));
        if want == 0 {
            break;
        }
        let end = utf8_floor_end(input, offset, offset.saturating_add(want));
        if end <= offset {
            // Single multi-byte char larger than remaining budget — skip with gap.
            let next = input[offset..]
                .chars()
                .next()
                .map_or(offset.saturating_add(1), |c| {
                    offset.saturating_add(c.len_utf8())
                });
            let gap = u64::try_from(next.saturating_sub(offset)).unwrap_or(0);
            out.push(ToolProgress::partial(String::new(), total, true, gap));
            offset = next;
            continue;
        }
        let delta = input.get(offset..end).unwrap_or("").to_owned();
        let delta_len = u64::try_from(delta.len()).unwrap_or(0);
        total = total.saturating_add(delta_len);
        let truncated = end - offset < want && end < bytes.len();
        out.push(ToolProgress::partial(delta, total, truncated, 0));
        offset = end;
    }
    out
}

/// Largest `end` in `(start, start+want]` that is a char boundary of `s`.
fn utf8_floor_end(s: &str, start: usize, end: usize) -> usize {
    let end = end.min(s.len());
    if end <= start {
        return start;
    }
    if s.is_char_boundary(end) {
        return end;
    }
    let mut e = end;
    while e > start && !s.is_char_boundary(e) {
        e = e.saturating_sub(1);
    }
    e
}

/// Single-item terminal stream from a completed result.
#[must_use]
pub fn terminal_only(result: Result<ToolResult, ToolError>) -> ToolStream {
    Box::pin(stream::once(
        async move { ToolStreamItem::Terminal(result) },
    ))
}

/// Progress items then a terminal future.
pub fn with_progress<I, F, Fut>(progress: I, terminal: F) -> ToolStream
where
    I: IntoIterator<Item = ToolProgress> + Send + 'static,
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = Result<ToolResult, ToolError>> + Send + 'static,
{
    let items: Vec<ToolStreamItem> = progress.into_iter().map(ToolStreamItem::Progress).collect();
    let prog = stream::iter(items);
    let term = stream::once(async move { ToolStreamItem::Terminal(terminal().await) });
    Box::pin(prog.chain(term))
}

/// Drain a stream to the terminal result, discarding progress.
///
/// # Errors
///
/// Returns stream protocol error when the stream ends without a terminal item.
pub async fn drain_terminal(stream: ToolStream) -> Result<ToolResult, ToolError> {
    let (_progress, result) = drain_with_progress(stream).await;
    result
}

/// Drain a stream, collecting progress items and the terminal result.
///
/// # Errors
///
/// The terminal `Result` carries tool failures. When the stream ends without a
/// terminal item, returns protocol error in the terminal slot.
pub async fn drain_with_progress(
    mut stream: ToolStream,
) -> (Vec<ToolProgress>, Result<ToolResult, ToolError>) {
    let mut progress = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            ToolStreamItem::Progress(p) => progress.push(p),
            ToolStreamItem::Terminal(result) => return (progress, result),
        }
    }
    (
        progress,
        Err(codes::stream_protocol(
            "tool stream ended without a terminal item",
        )),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn terminal_only_drains() {
        let s = terminal_only(Ok(ToolResult::text("ok")));
        let r = drain_terminal(s).await.expect("drain");
        assert_eq!(r.content, "ok");
    }

    #[tokio::test]
    async fn progress_then_terminal() {
        let s = with_progress(vec![ToolProgress::text("working")], || async {
            Ok(ToolResult::text("done"))
        });
        let r = drain_terminal(s).await.expect("drain");
        assert_eq!(r.content, "done");
    }

    #[tokio::test]
    async fn drain_with_progress_captures_chunks() {
        let s = with_progress(
            vec![ToolProgress::text("a"), ToolProgress::text("b")],
            || async { Ok(ToolResult::text("done")) },
        );
        let (progress, result) = drain_with_progress(s).await;
        assert_eq!(progress.len(), 2);
        assert_eq!(result.expect("ok").content, "done");
    }

    #[tokio::test]
    async fn empty_stream_is_protocol_error() {
        let s: ToolStream = Box::pin(stream::empty());
        let err = drain_terminal(s).await.expect_err("proto");
        assert_eq!(err.code(), ovo_types::ErrorCode::ToolStreamProtocol);
    }

    #[test]
    fn partial_frames_respect_utf8_and_caps() {
        let s = "hello🎉world";
        let frames = partial_progress_frames(s, 4, 10_000);
        assert!(!frames.is_empty());
        let mut rebuilt = String::new();
        for f in &frames {
            if let ToolProgress::Partial { delta, .. } = f {
                rebuilt.push_str(delta);
            }
        }
        // May skip oversized multi-byte with gap; all deltas must be valid UTF-8.
        assert!(rebuilt.is_char_boundary(rebuilt.len()));
        for f in frames {
            if let ToolProgress::Partial { delta, .. } = f {
                assert!(delta.len() <= 4 || delta.is_empty());
            }
        }
    }

    #[test]
    fn partial_frames_honor_frame_budget() {
        let s = "abcdefghij";
        let frames = partial_progress_frames(s, 3, 6);
        let total: usize = frames
            .iter()
            .map(|f| match f {
                ToolProgress::Partial { delta, .. } => delta.len(),
                _ => 0,
            })
            .sum();
        assert!(total <= 6);
    }
}
