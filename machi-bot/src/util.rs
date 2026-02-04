//! Common utility functions and types.
//!
//! This module provides shared utilities used across the crate to avoid code duplication.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Get current timestamp in milliseconds since Unix epoch.
#[inline]
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn timestamp_ms() -> u64 {
    // Truncation is safe: timestamp won't overflow u64 for ~500 million years
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis() as u64
}

/// Get current timestamp in microseconds since Unix epoch.
#[inline]
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn timestamp_us() -> u64 {
    // Truncation is safe: timestamp won't overflow u64 for ~500 million years
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_micros() as u64
}

/// Global counter for unique ID generation.
static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a unique message ID.
#[must_use]
pub fn generate_id(prefix: &str) -> String {
    let ts = timestamp_us();
    let counter = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}_{ts:x}_{counter:04x}")
}

/// Generate a unique message ID with default prefix.
#[inline]
#[must_use]
pub fn generate_message_id() -> String {
    generate_id("msg")
}

/// Generate a unique job ID.
#[inline]
#[must_use]
pub fn generate_job_id() -> String {
    generate_id("job")
}

/// Get the user's home directory.
#[must_use]
pub fn home_dir() -> PathBuf {
    dirs_next::home_dir().unwrap_or_else(|| PathBuf::from("."))
}

/// Get the default machi-bot configuration directory.
#[must_use]
pub fn config_dir() -> PathBuf {
    home_dir().join(".machi")
}

/// Get the default configuration file path.
#[must_use]
pub fn config_path() -> PathBuf {
    config_dir().join("config.toml")
}

/// Get the default sessions directory.
#[must_use]
pub fn sessions_dir() -> PathBuf {
    config_dir().join("sessions")
}

/// Get the default workspace directory.
#[must_use]
pub fn workspace_dir() -> PathBuf {
    config_dir().join("workspace")
}

/// Sanitize a string for use as a filename.
#[must_use]
pub fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            ':' | '/' | '\\' | '<' | '>' | '"' | '|' | '?' | '*' => '_',
            c if c.is_control() => '_',
            c => c,
        })
        .collect()
}

/// Truncate a string to a maximum length, adding ellipsis if truncated.
#[must_use]
pub fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len <= 3 {
        s.chars().take(max_len).collect()
    } else {
        let truncated: String = s.chars().take(max_len - 3).collect();
        format!("{truncated}...")
    }
}

/// Split a string into chunks of maximum length, preserving line breaks.
#[must_use]
pub fn split_into_chunks(text: &str, max_len: usize) -> Vec<String> {
    if text.len() <= max_len {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();

    for line in text.lines() {
        // Handle lines longer than max_len
        if line.len() > max_len {
            // Flush current buffer first
            if !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
            // Split the long line
            let mut remaining = line;
            while remaining.len() > max_len {
                let (chunk, rest) = remaining.split_at(max_len);
                chunks.push(chunk.to_string());
                remaining = rest;
            }
            if !remaining.is_empty() {
                current = remaining.to_string();
            }
            continue;
        }

        // Check if adding this line would exceed max_len
        let new_len = if current.is_empty() {
            line.len()
        } else {
            current.len() + 1 + line.len() // +1 for newline
        };

        if new_len > max_len {
            chunks.push(std::mem::take(&mut current));
        }

        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_ms() {
        let ts = timestamp_ms();
        assert!(ts > 0);
        // Should be after 2020-01-01
        assert!(ts > 1_577_836_800_000);
    }

    #[test]
    fn test_generate_id_unique() {
        let id1 = generate_id("test");
        let id2 = generate_id("test");
        assert_ne!(id1, id2);
        assert!(id1.starts_with("test_"));
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("hello:world"), "hello_world");
        assert_eq!(sanitize_filename("a/b\\c"), "a_b_c");
        assert_eq!(sanitize_filename("normal"), "normal");
    }

    #[test]
    fn test_truncate_str() {
        assert_eq!(truncate_str("hello", 10), "hello");
        assert_eq!(truncate_str("hello world", 8), "hello...");
        assert_eq!(truncate_str("hi", 2), "hi");
    }

    #[test]
    fn test_split_into_chunks() {
        let text = "line1\nline2\nline3";
        let chunks = split_into_chunks(text, 100);
        assert_eq!(chunks.len(), 1);

        let chunks = split_into_chunks(text, 6);
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_config_paths() {
        let cfg = config_dir();
        assert!(
            cfg.file_name().is_some_and(|n| n == ".machi"),
            "config dir should end with .machi"
        );

        let cfg_file = config_path();
        assert!(
            cfg_file.file_name().is_some_and(|n| n == "config.toml"),
            "config path should end with config.toml"
        );
    }
}
