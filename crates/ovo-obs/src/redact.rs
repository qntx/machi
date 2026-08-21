//! Log / telemetry field redaction.

/// Replacement token for redacted values.
pub const REDACTED: &str = "[REDACTED]";

/// Keys that typically carry secrets (case-insensitive match on last path segment).
const SECRET_KEY_FRAGMENTS: &[&str] = &[
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "cookie",
    "session",
    "private_key",
    "access_key",
    "client_secret",
    "bearer",
];

/// True when a field name looks secret-bearing.
#[must_use]
pub fn looks_like_secret_key(key: &str) -> bool {
    let lower = key.to_ascii_lowercase().replace('-', "_");
    SECRET_KEY_FRAGMENTS.iter().any(|frag| {
        lower == *frag
            || lower.ends_with(&format!("_{frag}"))
            || lower.ends_with(frag)
            || lower.contains(&format!("_{frag}_"))
            || lower.contains(frag)
    })
}

/// Redact a single key/value pair for logging.
#[must_use]
pub fn redact_key_value<'a>(key: &str, value: &'a str) -> &'a str {
    if looks_like_secret_key(key) {
        REDACTED
    } else {
        value
    }
}

/// Produce a redacted owned map suitable for debug dumps.
#[must_use]
pub fn redact_map(entries: &[(&str, &str)]) -> Vec<(String, String)> {
    entries
        .iter()
        .map(|(k, v)| {
            let value = if looks_like_secret_key(k) {
                REDACTED.to_owned()
            } else {
                (*v).to_owned()
            };
            ((*k).to_owned(), value)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_api_key() {
        assert!(looks_like_secret_key("api_key"));
        assert!(looks_like_secret_key("x-api-key"));
        assert_eq!(redact_key_value("Authorization", "Bearer x"), REDACTED);
        assert_eq!(redact_key_value("model", "gpt"), "gpt");
    }

    #[test]
    fn redact_map_preserves_safe_fields() {
        let out = redact_map(&[("model", "m"), ("token", "secret")]);
        assert_eq!(out.len(), 2);
        assert_eq!(out.first().map(|e| e.1.as_str()), Some("m"));
        assert_eq!(out.get(1).map(|e| e.1.as_str()), Some(REDACTED));
    }
}
