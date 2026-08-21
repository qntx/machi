//! Workflow run metadata store (journal path + status — not UI).

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::run::{PauseKind, WorkflowOutcome};

/// Coarse status for listing / resume UX.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum WorkflowRunStatus {
    /// Still running (host-owned; optional).
    Running,
    /// Completed successfully.
    Completed,
    /// Paused (resumable).
    Paused,
    /// Agent budget exceeded (resumable with higher budget).
    BudgetExceeded,
    /// Cancelled.
    Cancelled,
    /// Hard failure.
    Failed,
}

impl WorkflowRunStatus {
    /// Derive status from a terminal outcome.
    #[must_use]
    pub const fn from_outcome(outcome: &WorkflowOutcome) -> Self {
        match outcome {
            WorkflowOutcome::Completed { .. } => Self::Completed,
            WorkflowOutcome::Paused { .. } => Self::Paused,
            WorkflowOutcome::BudgetExceeded { .. } => Self::BudgetExceeded,
            WorkflowOutcome::Cancelled => Self::Cancelled,
            WorkflowOutcome::Failed { .. } => Self::Failed,
        }
    }
}

/// Durable metadata for one workflow run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkflowRunRecord {
    /// Stable run id (host-assigned, often [`ovo_types::WorkflowRunId`]).
    pub run_id: String,
    /// Workflow meta name when known.
    pub name: String,
    /// Optional description.
    #[serde(default)]
    pub description: String,
    /// Current status.
    pub status: WorkflowRunStatus,
    /// Absolute or relative path to the journal jsonl.
    pub journal_path: PathBuf,
    /// Optional script fingerprint (hash or path).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script_ref: Option<String>,
    /// Unix ms created.
    pub created_at_ms: u64,
    /// Unix ms last updated.
    pub updated_at_ms: u64,
    /// Last pause kind when paused.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pause_kind: Option<PauseKind>,
    /// Last error or pause message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Completed workflow result payload (for host `resume_from` replay).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
}

impl WorkflowRunRecord {
    /// Start a running record.
    #[must_use]
    pub fn new_running(
        run_id: impl Into<String>,
        name: impl Into<String>,
        journal_path: PathBuf,
    ) -> Self {
        let now = unix_now_ms();
        Self {
            run_id: run_id.into(),
            name: name.into(),
            description: String::new(),
            status: WorkflowRunStatus::Running,
            journal_path,
            script_ref: None,
            created_at_ms: now,
            updated_at_ms: now,
            pause_kind: None,
            message: None,
            result: None,
        }
    }

    /// Apply a terminal outcome.
    pub fn apply_outcome(&mut self, outcome: &WorkflowOutcome) {
        self.status = WorkflowRunStatus::from_outcome(outcome);
        self.updated_at_ms = unix_now_ms();
        match outcome {
            WorkflowOutcome::Paused { kind, message } => {
                self.pause_kind = Some(*kind);
                self.message = Some(message.clone());
                self.result = None;
            }
            WorkflowOutcome::BudgetExceeded { message }
            | WorkflowOutcome::Failed { error: message } => {
                self.pause_kind = None;
                self.message = Some(message.clone());
                self.result = None;
            }
            WorkflowOutcome::Completed { result } => {
                self.pause_kind = None;
                self.message = None;
                self.result = Some(result.clone());
            }
            WorkflowOutcome::Cancelled => {
                self.pause_kind = None;
                self.message = None;
                self.result = None;
            }
        }
    }
}

/// Host port for listing and resuming workflow runs.
pub trait WorkflowRunStore: Send + Sync {
    /// Insert or replace a record.
    ///
    /// # Errors
    ///
    /// Backend I/O failures.
    fn put(&self, record: WorkflowRunRecord) -> Result<(), StoreError>;

    /// Fetch by run id.
    ///
    /// # Errors
    ///
    /// Backend I/O failures.
    fn get(&self, run_id: &str) -> Result<Option<WorkflowRunRecord>, StoreError>;

    /// List all runs (newest-updated first when possible).
    ///
    /// # Errors
    ///
    /// Backend I/O failures.
    fn list(&self) -> Result<Vec<WorkflowRunRecord>, StoreError>;

    /// Remove a run metadata entry (does not delete the journal file).
    ///
    /// # Errors
    ///
    /// Backend I/O failures.
    fn delete(&self, run_id: &str) -> Result<bool, StoreError>;
}

/// Store failures.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// I/O.
    #[error("workflow store io: {0}")]
    Io(#[from] std::io::Error),
    /// Parse.
    #[error("workflow store parse: {0}")]
    Parse(String),
}

/// In-memory store for tests.
#[derive(Debug, Default)]
pub struct MemoryWorkflowRunStore {
    map: Mutex<BTreeMap<String, WorkflowRunRecord>>,
}

impl MemoryWorkflowRunStore {
    /// Empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl WorkflowRunStore for MemoryWorkflowRunStore {
    fn put(&self, record: WorkflowRunRecord) -> Result<(), StoreError> {
        self.map
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(record.run_id.clone(), record);
        Ok(())
    }

    fn get(&self, run_id: &str) -> Result<Option<WorkflowRunRecord>, StoreError> {
        Ok(self
            .map
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(run_id)
            .cloned())
    }

    fn list(&self) -> Result<Vec<WorkflowRunRecord>, StoreError> {
        let mut rows: Vec<_> = self
            .map
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .cloned()
            .collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.updated_at_ms));
        Ok(rows)
    }

    fn delete(&self, run_id: &str) -> Result<bool, StoreError> {
        Ok(self
            .map
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(run_id)
            .is_some())
    }
}

/// Directory of `*.json` records + optional shared journal root.
///
/// Layout:
/// ```text
/// {root}/
///   index.jsonl          # optional append-only index (rewritten on put for simplicity: one file per run)
///   runs/{run_id}.json
///   journals/{run_id}.jsonl   # conventional journal path helper
/// ```
#[derive(Debug, Clone)]
pub struct FileWorkflowRunStore {
    root: PathBuf,
}

impl FileWorkflowRunStore {
    /// Root directory (created on demand).
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Root accessor.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Conventional journal path for a run id.
    #[must_use]
    pub fn journal_path_for(&self, run_id: &str) -> PathBuf {
        self.root.join("journals").join(format!("{run_id}.jsonl"))
    }

    fn runs_dir(&self) -> PathBuf {
        self.root.join("runs")
    }

    fn record_path(&self, run_id: &str) -> PathBuf {
        self.runs_dir().join(format!("{run_id}.json"))
    }
}

impl WorkflowRunStore for FileWorkflowRunStore {
    fn put(&self, record: WorkflowRunRecord) -> Result<(), StoreError> {
        let dir = self.runs_dir();
        fs::create_dir_all(&dir)?;
        let path = self.record_path(&record.run_id);
        let tmp = path.with_extension("json.tmp");
        let body =
            serde_json::to_vec_pretty(&record).map_err(|e| StoreError::Parse(e.to_string()))?;
        {
            let mut f = File::create(&tmp)?;
            f.write_all(&body)?;
            f.sync_data()?;
        }
        fs::rename(&tmp, &path)?;
        Ok(())
    }

    fn get(&self, run_id: &str) -> Result<Option<WorkflowRunRecord>, StoreError> {
        let path = self.record_path(run_id);
        if !path.is_file() {
            return Ok(None);
        }
        let bytes = fs::read(&path)?;
        let rec = serde_json::from_slice(&bytes).map_err(|e| StoreError::Parse(e.to_string()))?;
        Ok(Some(rec))
    }

    fn list(&self) -> Result<Vec<WorkflowRunRecord>, StoreError> {
        let dir = self.runs_dir();
        if !dir.is_dir() {
            return Ok(Vec::new());
        }
        let mut rows = Vec::new();
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let bytes = fs::read(&path)?;
            let rec: WorkflowRunRecord =
                serde_json::from_slice(&bytes).map_err(|e| StoreError::Parse(e.to_string()))?;
            rows.push(rec);
        }
        rows.sort_by_key(|r| std::cmp::Reverse(r.updated_at_ms));
        Ok(rows)
    }

    fn delete(&self, run_id: &str) -> Result<bool, StoreError> {
        let path = self.record_path(run_id);
        if !path.is_file() {
            return Ok(false);
        }
        fs::remove_file(path)?;
        Ok(true)
    }
}

/// Read first line of a jsonl index if present (diagnostic helper).
///
/// # Errors
///
/// Returns I/O errors when the path exists but cannot be read.
pub fn peek_jsonl_line(path: &Path) -> Result<Option<String>, StoreError> {
    if !path.is_file() {
        return Ok(None);
    }
    let f = File::open(path)?;
    let mut lines = BufReader::new(f).lines();
    match lines.next() {
        Some(Ok(line)) => Ok(Some(line)),
        Some(Err(e)) => Err(e.into()),
        None => Ok(None),
    }
}

fn unix_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn memory_put_list_delete() {
        let store = MemoryWorkflowRunStore::new();
        let mut rec = WorkflowRunRecord::new_running("run_1", "demo", PathBuf::from("j.jsonl"));
        store.put(rec.clone()).expect("put");
        rec.apply_outcome(&WorkflowOutcome::Completed {
            result: serde_json::json!({"ok": true}),
        });
        store.put(rec).expect("put2");
        let listed = store.list().expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(
            listed.first().map(|r| r.status),
            Some(WorkflowRunStatus::Completed)
        );
        assert!(store.delete("run_1").expect("del"));
        assert!(store.get("run_1").expect("get").is_none());
    }

    #[test]
    fn file_store_round_trip() {
        let dir = tempdir().expect("tmp");
        let store = FileWorkflowRunStore::new(dir.path());
        let journal = store.journal_path_for("wf_abc");
        let mut rec = WorkflowRunRecord::new_running("wf_abc", "fanout", journal);
        rec.description = "test".into();
        store.put(rec.clone()).expect("put");
        let got = store.get("wf_abc").expect("get").expect("some");
        assert_eq!(got.name, "fanout");
        assert_eq!(store.list().expect("list").len(), 1);
        rec.apply_outcome(&WorkflowOutcome::Failed {
            error: "boom".into(),
        });
        store.put(rec).expect("put fail");
        let got = store.get("wf_abc").expect("g2").expect("s");
        assert_eq!(got.status, WorkflowRunStatus::Failed);
        assert_eq!(got.message.as_deref(), Some("boom"));
    }
}
