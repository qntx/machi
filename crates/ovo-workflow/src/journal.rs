//! Append-only host-call journal for resume (format v2).
//!
//!
//! Format v2:
//! - optional first line: `# ovo-journal/2`
//! - dense JSONL entries with canonical 16-byte request hashes
//! - `MAX_JOURNAL_BYTES` enforced on load and append
//! - torn-write repair of an incomplete final line

use std::io::Write as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::MAX_HOST_CALLS;

/// Maximum journal file size in bytes.
pub const MAX_JOURNAL_BYTES: u64 = 64 * 1024 * 1024;

/// Maximum journal entries (same ceiling as host-call budget).
#[allow(
    clippy::cast_possible_truncation,
    reason = "MAX_HOST_CALLS fits usize on all supported targets"
)]
pub const MAX_JOURNAL_ENTRIES: usize = MAX_HOST_CALLS as usize;

/// First-line format marker for durable journals.
pub const JOURNAL_VERSION_HEADER: &str = "# ovo-journal/2";

/// Sentinel key for journaled, re-raiseable host failures (`Failed` / `Unsupported`).
pub const HOST_ERROR_KEY: &str = "__ovo_host_error";

/// One recorded host call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(
    clippy::derive_partial_eq_without_eq,
    reason = "result contains JSON Value"
)]
pub struct JournalEntry {
    /// Dense sequence number starting at 0.
    pub seq: u64,
    /// Request kind.
    pub kind: String,
    /// Hash of canonical request payload (32 hex chars = 16 digest bytes).
    pub req_hash: String,
    /// Recorded result JSON.
    pub result: serde_json::Value,
    /// Host wall-clock ms (not used for script determinism).
    pub at_ms: u64,
}

/// Journal failures.
#[derive(Debug, thiserror::Error)]
pub enum JournalError {
    /// I/O failure.
    #[error("journal io: {0}")]
    Io(#[from] std::io::Error),
    /// Parse failure on a complete line.
    #[error("journal parse at line {line}: {error}")]
    Parse {
        /// Line number (1-based among file lines).
        line: usize,
        /// Parse error.
        error: String,
    },
    /// Restore rejected for safety (symlink, size, entry count).
    #[error("journal restore rejected (limit {limit}): {reason}")]
    UnsafeRestore {
        /// Cap that was violated.
        limit: u64,
        /// Human reason.
        reason: String,
    },
    /// Sequence gap or mismatch.
    #[error("journal is not dense at entry {index}: expected sequence {expected}, found {actual}")]
    Sequence {
        /// Index.
        index: usize,
        /// Expected seq.
        expected: u64,
        /// Actual seq.
        actual: u64,
    },
    /// Replay saw a different request than recorded.
    #[error(
        "replay divergence at seq {seq} ({kind}): the script issued a different call than the \
         recorded run — the workflow script is nondeterministic or was edited mid-run"
    )]
    Divergence {
        /// Sequence.
        seq: u64,
        /// Kind.
        kind: String,
    },
    /// Append would exceed the durable byte cap.
    #[error(
        "journal full: appending seq {seq} would exceed the {limit}-byte cap \
         that restore enforces, which would strand the run unresumable"
    )]
    Full {
        /// Sequence that would have been written.
        seq: u64,
        /// Byte limit.
        limit: u64,
    },
}

/// In-memory journal with optional durable path.
#[derive(Debug, Default)]
pub struct Journal {
    entries: Vec<JournalEntry>,
    path: Option<PathBuf>,
    bytes: u64,
    /// Byte offset of each entry's line start (parallel to [`Self::entries`]).
    /// Enables repeated [`Self::prune_trailing_host_error`] without reload.
    line_starts: Vec<u64>,
    /// Whether a version header is present / should be written on first append.
    has_header: bool,
}

impl Journal {
    /// Empty journal, optionally bound to a path for appends.
    #[must_use]
    pub const fn new(path: Option<PathBuf>) -> Self {
        Self {
            entries: Vec::new(),
            path,
            bytes: 0,
            line_starts: Vec::new(),
            has_header: false,
        }
    }

    /// Load from jsonl path (missing file => empty bound journal).
    ///
    /// # Errors
    ///
    /// Returns parse/IO/unsafe-restore errors. Torn final lines are repaired
    /// (completed with `\n` when valid JSON, otherwise truncated).
    #[allow(
        clippy::too_many_lines,
        clippy::excessive_nesting,
        clippy::indexing_slicing,
        clippy::single_match_else,
        reason = "byte-oriented journal parser with torn-write repair is inherently nested"
    )]
    pub fn load(path: PathBuf) -> Result<Self, JournalError> {
        let content = match read_journal_bounded(&path) {
            Ok(content) => content,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::new(Some(path)));
            }
            Err(error) if error.kind() == std::io::ErrorKind::InvalidData => {
                return Err(JournalError::UnsafeRestore {
                    limit: MAX_JOURNAL_BYTES,
                    reason: error.to_string(),
                });
            }
            Err(error) => return Err(error.into()),
        };

        let mut entries = Vec::new();
        let mut line_starts = Vec::new();
        let mut offset = 0usize;
        let mut line_number = 0usize;
        let mut bytes = u64::try_from(content.len()).unwrap_or(u64::MAX);
        let mut has_header = false;

        while offset < content.len() {
            line_number = line_number.saturating_add(1);
            let Some(relative_newline) = content
                .get(offset..)
                .and_then(|s| s.iter().position(|b| *b == b'\n'))
            else {
                // Torn final line (no trailing newline).
                let tail = content.get(offset..).unwrap_or(&[]);
                if tail.iter().all(u8::is_ascii_whitespace) {
                    truncate_tail(&path, u64::try_from(offset).unwrap_or(0))?;
                    bytes = u64::try_from(offset).unwrap_or(0);
                    break;
                }
                match serde_json::from_slice::<JournalEntry>(tail) {
                    Ok(entry) => {
                        if entries.len() >= MAX_JOURNAL_ENTRIES {
                            return Err(JournalError::UnsafeRestore {
                                limit: u64::try_from(MAX_JOURNAL_ENTRIES).unwrap_or(u64::MAX),
                                reason: "too many journal entries".into(),
                            });
                        }
                        if validate_sequence(&entries, &entry).is_err() {
                            // Wrong-seq torn tail: drop it so prior entries stay loadable.
                            truncate_tail(&path, u64::try_from(offset).unwrap_or(0))?;
                            bytes = u64::try_from(offset).unwrap_or(0);
                            break;
                        }
                        let line_start = u64::try_from(offset).unwrap_or(0);
                        // Completing the line adds one byte; must stay within the restore cap.
                        if bytes.saturating_add(1) > MAX_JOURNAL_BYTES {
                            truncate_tail(&path, line_start)?;
                            bytes = line_start;
                            break;
                        }
                        entries.push(entry);
                        line_starts.push(line_start);
                        terminate_line(&path)?;
                        bytes = bytes.saturating_add(1);
                    }
                    Err(_) => {
                        truncate_tail(&path, u64::try_from(offset).unwrap_or(0))?;
                        bytes = u64::try_from(offset).unwrap_or(0);
                    }
                }
                break;
            };

            let end = offset.saturating_add(relative_newline);
            let line = content.get(offset..end).unwrap_or(&[]);
            let line_start = u64::try_from(offset).unwrap_or(0);
            offset = end.saturating_add(1);

            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }

            // Version header (format evolution anchor).
            if line_number == 1 && line.starts_with(b"#") {
                let header = std::str::from_utf8(line).unwrap_or("");
                if header.trim() == JOURNAL_VERSION_HEADER {
                    has_header = true;
                    continue;
                }
                return Err(JournalError::UnsafeRestore {
                    limit: 2,
                    reason: format!("unsupported journal header: {header}"),
                });
            }

            let entry = serde_json::from_slice::<JournalEntry>(line).map_err(|error| {
                JournalError::Parse {
                    line: line_number,
                    error: error.to_string(),
                }
            })?;
            if entries.len() >= MAX_JOURNAL_ENTRIES {
                return Err(JournalError::UnsafeRestore {
                    limit: u64::try_from(MAX_JOURNAL_ENTRIES).unwrap_or(u64::MAX),
                    reason: "too many journal entries".into(),
                });
            }
            validate_sequence(&entries, &entry)?;
            entries.push(entry);
            line_starts.push(line_start);
        }

        debug_assert_eq!(
            entries.len(),
            line_starts.len(),
            "line_starts must track every journal entry"
        );
        Ok(Self {
            entries,
            path: Some(path),
            bytes,
            line_starts,
            has_header,
        })
    }

    /// Number of entries.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Empty check.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Durable byte length (header + lines, after repairs).
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Count of recorded `spawn_agent` entries (reservation accounting helper).
    #[must_use]
    pub fn agent_reservation_count(&self) -> u64 {
        u64::try_from(
            self.entries
                .iter()
                .filter(|entry| entry.kind == "spawn_agent")
                .count(),
        )
        .unwrap_or(u64::MAX)
    }

    /// Whether `seq` is already covered.
    #[must_use]
    pub fn covers(&self, seq: u64) -> bool {
        usize::try_from(seq).is_ok_and(|i| i < self.entries.len())
    }

    /// Replay a covered call or return `None` to execute live.
    ///
    /// # Errors
    ///
    /// Divergence when kind/hash mismatch.
    pub fn replay(
        &self,
        seq: u64,
        kind: &str,
        hash: &str,
    ) -> Result<Option<serde_json::Value>, JournalError> {
        let Some(entry) = usize::try_from(seq).ok().and_then(|i| self.entries.get(i)) else {
            return Ok(None);
        };
        if entry.seq != seq || entry.kind != kind || entry.req_hash != hash {
            return Err(JournalError::Divergence {
                seq,
                kind: kind.to_owned(),
            });
        }
        Ok(Some(entry.result.clone()))
    }

    /// Append a live result.
    ///
    /// # Errors
    ///
    /// Full journal, sequence errors, or IO failures.
    #[allow(
        clippy::excessive_nesting,
        reason = "durable append has header + TOCTOU + path branches collocated"
    )]
    pub fn record(
        &mut self,
        seq: u64,
        kind: &str,
        hash: String,
        result: serde_json::Value,
    ) -> Result<(), JournalError> {
        let entry = JournalEntry {
            seq,
            kind: kind.to_owned(),
            req_hash: hash,
            result,
            at_ms: unix_now_ms(),
        };
        validate_sequence(&self.entries, &entry)?;
        if self.entries.len() >= MAX_JOURNAL_ENTRIES {
            return Err(JournalError::Full {
                seq,
                limit: u64::try_from(MAX_JOURNAL_ENTRIES).unwrap_or(u64::MAX),
            });
        }

        let mut line = serde_json::to_string(&entry).map_err(|error| {
            JournalError::Io(std::io::Error::other(format!("serialize entry: {error}")))
        })?;
        line.push('\n');
        let line_len = u64::try_from(line.len()).unwrap_or(u64::MAX);

        // Ensure version header on first durable write.
        let header_extra = if self.path.is_some() && !self.has_header && self.entries.is_empty() {
            u64::try_from(JOURNAL_VERSION_HEADER.len().saturating_add(1)).unwrap_or(0)
        } else {
            0
        };

        if self
            .bytes
            .saturating_add(header_extra)
            .saturating_add(line_len)
            > MAX_JOURNAL_BYTES
        {
            return Err(JournalError::Full {
                seq,
                limit: MAX_JOURNAL_BYTES,
            });
        }

        if let Some(path) = &self.path {
            // TOCTOU re-check: reject when on-disk size + this write would exceed the cap.
            if path.exists() {
                let disk = std::fs::metadata(path)?.len();
                if disk.saturating_add(header_extra).saturating_add(line_len) > MAX_JOURNAL_BYTES {
                    return Err(JournalError::Full {
                        seq,
                        limit: MAX_JOURNAL_BYTES,
                    });
                }
            }
            if !self.has_header && self.entries.is_empty() {
                write_version_header(path)?;
                self.has_header = true;
                self.bytes = self.bytes.saturating_add(header_extra);
            }
            let line_start = self.bytes;
            append_line(path, &line)?;
            self.line_starts.push(line_start);
        } else {
            self.line_starts.push(self.bytes);
        }
        self.bytes = self.bytes.saturating_add(line_len);

        self.entries.push(entry);
        Ok(())
    }

    /// Drop the trailing host-error sentinel when it matches `failure_detail`.
    ///
    /// Used so a recoverable host failure can be retried on resume without
    /// replaying the same error. Safe to call repeatedly: each successful prune
    /// restores the prior entry's line offset.
    ///
    /// # Errors
    ///
    /// IO failures while truncating the durable file.
    pub fn prune_trailing_host_error(
        &mut self,
        failure_detail: &str,
    ) -> Result<bool, JournalError> {
        let Some(last) = self.entries.last() else {
            return Ok(false);
        };
        let Some(message) = last.result.get(HOST_ERROR_KEY).and_then(|v| v.as_str()) else {
            return Ok(false);
        };
        if message.is_empty() || !failure_detail.contains(message) {
            return Ok(false);
        }
        let Some(new_len) = self.line_starts.last().copied() else {
            return Err(JournalError::Io(std::io::Error::other(
                "journal cannot locate the trailing entry's byte offset",
            )));
        };
        if let Some(path) = &self.path {
            truncate_tail(path, new_len)?;
        }
        self.entries.pop();
        self.line_starts.pop();
        self.bytes = new_len;
        Ok(true)
    }

    /// Drop trailing host-error sentinels while `failure_detail` matches the
    /// current last entry's message (repeated prune).
    ///
    /// # Errors
    ///
    /// IO failures while truncating.
    pub fn prune_trailing_host_errors(
        &mut self,
        failure_detail: &str,
    ) -> Result<usize, JournalError> {
        let mut n = 0usize;
        while self.prune_trailing_host_error(failure_detail)? {
            n = n.saturating_add(1);
        }
        Ok(n)
    }

    /// Optional durable path.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Entries as a slice (for inspection / tests).
    #[must_use]
    pub fn entries(&self) -> &[JournalEntry] {
        &self.entries
    }
}

/// Build a host-error sentinel payload for journal + replay.
#[must_use]
pub fn host_error_sentinel(message: &str) -> serde_json::Value {
    serde_json::json!({ HOST_ERROR_KEY: message })
}

/// Whether `value` is a host-error sentinel.
#[must_use]
pub fn is_host_error_sentinel(value: &serde_json::Value) -> bool {
    value
        .get(HOST_ERROR_KEY)
        .and_then(serde_json::Value::as_str)
        .is_some()
}

/// Extract host-error message from a sentinel, if any.
#[must_use]
pub fn host_error_message(value: &serde_json::Value) -> Option<&str> {
    value
        .get(HOST_ERROR_KEY)
        .and_then(serde_json::Value::as_str)
}

/// Recursively sort object keys for stable hashing.
#[must_use]
pub fn canonical_json(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
            entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
            serde_json::Value::Object(
                entries
                    .into_iter()
                    .map(|(k, v)| (k.clone(), canonical_json(v)))
                    .collect(),
            )
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(canonical_json).collect())
        }
        other => other.clone(),
    }
}

/// Hash a host request for divergence detection (16 digest bytes → 32 hex chars).
#[must_use]
pub fn request_hash(kind: &str, payload: &serde_json::Value) -> String {
    let mut hasher = Sha256::new();
    hasher.update(kind.as_bytes());
    hasher.update([0u8]);
    hasher.update(canonical_json(payload).to_string().as_bytes());
    let digest = hasher.finalize();
    encode_hex(digest.iter().take(16).copied())
}

fn encode_hex(bytes: impl IntoIterator<Item = u8>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(32);
    for b in bytes {
        let hi = usize::from(b >> 4);
        let lo = usize::from(b & 0x0f);
        if let (Some(&h), Some(&l)) = (HEX.get(hi), HEX.get(lo)) {
            s.push(char::from(h));
            s.push(char::from(l));
        }
    }
    s
}

fn validate_sequence(entries: &[JournalEntry], entry: &JournalEntry) -> Result<(), JournalError> {
    let expected = u64::try_from(entries.len()).unwrap_or(u64::MAX);
    if entry.seq != expected {
        return Err(JournalError::Sequence {
            index: entries.len(),
            expected,
            actual: entry.seq,
        });
    }
    Ok(())
}

fn read_journal_bounded(path: &Path) -> std::io::Result<Vec<u8>> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("journal is not a regular file: {}", path.display()),
        ));
    }
    if metadata.len() > MAX_JOURNAL_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("journal exceeds {MAX_JOURNAL_BYTES} bytes"),
        ));
    }

    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        // Reject opening through a symlink (TOCTOU defense after symlink_metadata).
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let mut file = options.open(path)?;
    let opened = file.metadata()?;
    if !opened.is_file() || opened.len() > MAX_JOURNAL_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "journal changed during open",
        ));
    }
    let mut content = Vec::with_capacity(usize::try_from(opened.len()).unwrap_or(0));
    std::io::Read::read_to_end(&mut file, &mut content)?;
    if u64::try_from(content.len()).unwrap_or(u64::MAX) > MAX_JOURNAL_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("journal exceeds {MAX_JOURNAL_BYTES} bytes"),
        ));
    }
    Ok(content)
}

fn truncate_tail(path: &Path, len: u64) -> Result<(), JournalError> {
    let file = std::fs::OpenOptions::new().write(true).open(path)?;
    file.set_len(len)?;
    file.sync_data()?;
    Ok(())
}

fn terminate_line(path: &Path) -> Result<(), JournalError> {
    let mut file = std::fs::OpenOptions::new().append(true).open(path)?;
    file.write_all(b"\n")?;
    file.sync_data()?;
    Ok(())
}

fn write_version_header(path: &Path) -> Result<(), JournalError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    // Only write if file is empty (first durable record).
    if file.metadata()?.len() == 0 {
        file.write_all(JOURNAL_VERSION_HEADER.as_bytes())?;
        file.write_all(b"\n")?;
        file.sync_data()?;
    }
    Ok(())
}

fn append_line(path: &Path, line: &str) -> Result<(), JournalError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    file.write_all(line.as_bytes())?;
    file.sync_data()?;
    Ok(())
}

fn unix_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "unit tests use expect/unwrap"
)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn replay_and_divergence() {
        let mut j = Journal::new(None);
        let hash = request_hash("spawn_agent", &json!({"prompt": "a"}));
        j.record(0, "spawn_agent", hash.clone(), json!({"ok": true}))
            .expect("record");
        let replayed = j
            .replay(0, "spawn_agent", &hash)
            .expect("replay")
            .expect("hit");
        assert_eq!(replayed.get("ok"), Some(&json!(true)));
        let err = j
            .replay(0, "spawn_agent", "deadbeef")
            .expect_err("divergence");
        assert!(matches!(err, JournalError::Divergence { .. }));
    }

    #[test]
    fn canonical_hash_is_key_order_independent() {
        let a = request_hash("k", &json!({"b": 1, "a": 2}));
        let b = request_hash("k", &json!({"a": 2, "b": 1}));
        assert_eq!(a, b);
        assert_eq!(a.len(), 32);
    }

    #[test]
    fn durable_round_trip_with_version_header() {
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("j.jsonl");
        let mut j = Journal::new(Some(path.clone()));
        let hash = request_hash("spawn_agent", &json!(1));
        j.record(0, "spawn_agent", hash.clone(), json!(42))
            .expect("rec");
        let raw = std::fs::read_to_string(&path).expect("read");
        assert!(
            raw.starts_with(JOURNAL_VERSION_HEADER),
            "missing header: {raw}"
        );
        let loaded = Journal::load(path).expect("load");
        assert_eq!(loaded.len(), 1);
        assert_eq!(
            loaded.replay(0, "spawn_agent", &hash).expect("r"),
            Some(json!(42))
        );
    }

    #[test]
    fn durable_load_after_drop_simulates_cross_process() {
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("cross.jsonl");
        {
            let mut j = Journal::new(Some(path.clone()));
            let h0 = request_hash("spawn_agent", &json!({"prompt": "a"}));
            let h1 = request_hash("spawn_agent", &json!({"prompt": "b"}));
            j.record(0, "spawn_agent", h0, json!({"output": "A"}))
                .expect("r0");
            j.record(1, "spawn_agent", h1, json!({"output": "B"}))
                .expect("r1");
        }
        let loaded = Journal::load(path).expect("load");
        assert_eq!(loaded.len(), 2);
        let h0 = request_hash("spawn_agent", &json!({"prompt": "a"}));
        let replayed = loaded
            .replay(0, "spawn_agent", &h0)
            .expect("replay")
            .expect("hit");
        assert_eq!(replayed.get("output"), Some(&json!("A")));
    }

    #[test]
    fn sequence_gap_on_record() {
        let mut j = Journal::new(None);
        let err = j
            .record(1, "spawn_agent", "h".into(), json!(null))
            .expect_err("seq");
        assert!(matches!(err, JournalError::Sequence { .. }));
    }

    #[test]
    fn torn_tail_valid_json_gets_newline() {
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("torn.jsonl");
        let entry = JournalEntry {
            seq: 0,
            kind: "spawn_agent".into(),
            req_hash: "aa".into(),
            result: json!({"ok": true}),
            at_ms: 0,
        };
        let body = serde_json::to_string(&entry).expect("ser");
        // Header + body without trailing newline.
        let mut raw = String::new();
        raw.push_str(JOURNAL_VERSION_HEADER);
        raw.push('\n');
        raw.push_str(&body);
        std::fs::write(&path, raw.as_bytes()).expect("write");

        let loaded = Journal::load(path.clone()).expect("load");
        assert_eq!(loaded.len(), 1);
        let disk = std::fs::read_to_string(&path).expect("reread");
        assert!(disk.ends_with('\n'));
    }

    #[test]
    fn torn_tail_invalid_json_is_truncated() {
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("torn_bad.jsonl");
        let mut raw = String::new();
        raw.push_str(JOURNAL_VERSION_HEADER);
        raw.push('\n');
        // One good entry.
        let good = JournalEntry {
            seq: 0,
            kind: "spawn_agent".into(),
            req_hash: "aa".into(),
            result: json!(1),
            at_ms: 0,
        };
        raw.push_str(&serde_json::to_string(&good).expect("ser"));
        raw.push('\n');
        raw.push_str("{\"seq\":1, incomplete");
        std::fs::write(&path, raw.as_bytes()).expect("write");

        let loaded = Journal::load(path.clone()).expect("load");
        assert_eq!(loaded.len(), 1);
        let disk = std::fs::read_to_string(&path).expect("reread");
        assert!(!disk.contains("incomplete"));
    }

    #[test]
    fn prune_trailing_host_error_truncates_and_allows_reappend() {
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("prune.jsonl");
        let mut j = Journal::new(Some(path.clone()));
        let h0 = request_hash("spawn_agent", &json!({"p": "a"}));
        j.record(0, "spawn_agent", h0, json!({"ok": true}))
            .expect("r0");
        let h1 = request_hash("spawn_agent", &json!({"p": "b"}));
        j.record(1, "spawn_agent", h1, host_error_sentinel("boom"))
            .expect("r1");
        assert!(j.prune_trailing_host_error("error: boom").expect("prune"));
        assert_eq!(j.len(), 1);
        let h1b = request_hash("spawn_agent", &json!({"p": "b"}));
        j.record(1, "spawn_agent", h1b, json!({"ok": 2}))
            .expect("reappend");
        let loaded = Journal::load(path).expect("load");
        assert_eq!(loaded.len(), 2);
    }

    #[test]
    fn prune_twice_without_reload_keeps_line_offsets() {
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("prune2.jsonl");
        let mut j = Journal::new(Some(path.clone()));
        j.record(0, "spawn_agent", "h0".into(), json!({"ok": true}))
            .expect("r0");
        j.record(1, "spawn_agent", "h1".into(), host_error_sentinel("e1"))
            .expect("r1");
        j.record(2, "spawn_agent", "h2".into(), host_error_sentinel("e2"))
            .expect("r2");
        assert_eq!(j.prune_trailing_host_errors("e1 e2").expect("prune"), 2);
        assert_eq!(j.len(), 1);
        j.record(1, "spawn_agent", "h1b".into(), json!(1))
            .expect("re");
        let loaded = Journal::load(path).expect("load");
        assert_eq!(loaded.len(), 2);
    }

    #[test]
    fn prune_is_noop_when_last_is_success() {
        let mut j = Journal::new(None);
        j.record(0, "spawn_agent", "h".into(), json!(true))
            .expect("r");
        assert!(!j.prune_trailing_host_error("boom").expect("p"));
        assert_eq!(j.len(), 1);
    }

    #[test]
    fn torn_tail_at_byte_cap_is_dropped_not_extended() {
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("cap_torn.jsonl");
        // Build a file that is exactly MAX_JOURNAL_BYTES ending mid-entry without newline.
        // Use a tiny payload repeated until near the cap is impractical; instead write a
        // header + valid entry, then pad with spaces to MAX-5 and a torn '{' without newline.
        let entry = JournalEntry {
            seq: 0,
            kind: "spawn_agent".into(),
            req_hash: "aa".into(),
            result: json!(1),
            at_ms: 0,
        };
        let body = serde_json::to_string(&entry).expect("ser");
        let mut raw = String::new();
        raw.push_str(JOURNAL_VERSION_HEADER);
        raw.push('\n');
        raw.push_str(&body);
        raw.push('\n');
        let max_usize = usize::try_from(MAX_JOURNAL_BYTES).unwrap_or(usize::MAX);
        let pad = max_usize.saturating_sub(raw.len()).saturating_sub(1);
        raw.extend(std::iter::repeat_n(' ', pad));
        raw.push('{'); // torn, no newline; file len == MAX
        assert_eq!(
            u64::try_from(raw.len()).unwrap_or(u64::MAX),
            MAX_JOURNAL_BYTES,
            "fixture must be exactly MAX_JOURNAL_BYTES"
        );
        std::fs::write(&path, raw.as_bytes()).expect("write");
        let loaded = Journal::load(path.clone()).expect("load");
        assert_eq!(loaded.len(), 1);
        // Must still load under the cap.
        let meta = std::fs::metadata(&path).expect("meta");
        assert!(meta.len() <= MAX_JOURNAL_BYTES);
    }

    #[test]
    fn oversized_file_rejected_on_load() {
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("big.jsonl");
        // Write a sparse file larger than the cap without filling RAM.
        let file = std::fs::File::create(&path).expect("create");
        file.set_len(MAX_JOURNAL_BYTES.saturating_add(1))
            .expect("set_len");
        let err = Journal::load(path).expect_err("must reject");
        assert!(matches!(err, JournalError::UnsafeRestore { .. }));
    }

    #[test]
    fn append_respects_byte_cap() {
        let mut j = Journal::new(None);
        // Force full by setting bytes near the cap without a path.
        j.bytes = MAX_JOURNAL_BYTES;
        let err = j
            .record(0, "spawn_agent", "h".into(), json!(null))
            .expect_err("full");
        assert!(matches!(err, JournalError::Full { .. }));
    }

    #[cfg(unix)]
    #[test]
    fn symlink_journal_rejected() {
        let dir = tempfile::tempdir().expect("tmp");
        let target = dir.path().join("real.jsonl");
        std::fs::write(&target, format!("{JOURNAL_VERSION_HEADER}\n")).expect("write");
        let link = dir.path().join("link.jsonl");
        std::os::unix::fs::symlink(&target, &link).expect("symlink");
        let err = Journal::load(link).expect_err("symlink");
        assert!(matches!(err, JournalError::UnsafeRestore { .. }));
    }
}
