//! JSON Schema validation for structured agent outputs.

use jsonschema::Validator;
use ovo_types::{ErrorCode, OvoError};
use serde_json::Value;

/// Max corrective re-samples when structured output fails validation.
pub const STRUCTURED_OUTPUT_MAX_RETRIES: u32 = 3;

/// Compile a JSON Schema once per turn (or once per agent definition).
///
/// # Errors
///
/// Returns [`ErrorCode::RuntimeStructuredOutput`] when the schema itself is invalid.
pub fn compile_schema(schema: &Value) -> Result<Validator, OvoError> {
    Validator::new(schema).map_err(|e| {
        OvoError::new(
            ErrorCode::RuntimeStructuredOutput,
            format!("invalid output schema: {e}"),
        )
    })
}

/// Parse model text as JSON and validate against a compiled schema.
///
/// # Errors
///
/// Returns a human-readable validation error suitable for model feedback.
pub fn validate_structured_output(validator: &Validator, raw: &str) -> Result<Value, String> {
    let value: Value = serde_json::from_str(raw.trim())
        .map_err(|e| format!("model output was not valid JSON: {e}"))?;
    validator
        .validate(&value)
        .map_err(|e| format!("output does not match the required schema: {e}"))?;
    Ok(value)
}

/// Build a corrective user reminder after a schema failure.
#[must_use]
pub fn schema_retry_reminder(error: &str) -> String {
    format!(
        "Your previous response failed structured-output validation:\n{error}\n\
         Reply with JSON only that satisfies the required schema."
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn accepts_valid() {
        let schema = json!({
            "type": "object",
            "properties": { "ok": { "type": "boolean" } },
            "required": ["ok"],
            "additionalProperties": false
        });
        let v = compile_schema(&schema).expect("schema");
        let out = validate_structured_output(&v, r#"{"ok": true}"#).expect("ok");
        assert_eq!(out.get("ok").and_then(Value::as_bool), Some(true));
    }

    #[test]
    fn rejects_invalid() {
        let schema = json!({
            "type": "object",
            "properties": { "ok": { "type": "boolean" } },
            "required": ["ok"]
        });
        let v = compile_schema(&schema).expect("schema");
        let err = validate_structured_output(&v, r#"{"ok": "nope"}"#).expect_err("bad");
        assert!(err.contains("schema") || err.contains("type"), "{err}");
    }
}
