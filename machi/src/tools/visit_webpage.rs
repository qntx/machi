//! Tool for visiting and reading webpage content.

use crate::tool::{Tool, ToolError};
use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::LazyLock;

/// Tool for visiting a webpage and extracting its content as text.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct VisitWebpageTool {
    /// Maximum output length in characters.
    pub max_output_length: usize,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
}

impl Default for VisitWebpageTool {
    fn default() -> Self {
        Self {
            max_output_length: 40000,
            timeout_secs: 20,
        }
    }
}

/// Arguments for visiting a webpage.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct VisitWebpageArgs {
    /// The URL of the webpage to visit.
    pub url: String,
}

// Pre-compiled regex patterns for HTML to text conversion
static SCRIPT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?is)<script[^>]*>.*?</script>").expect("valid regex"));
static STYLE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?is)<style[^>]*>.*?</style>").expect("valid regex"));
static TAG_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]+>").expect("valid regex"));
static MULTILINE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\n{3,}").expect("valid regex"));

// HTML element conversion patterns
static HTML_PATTERNS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    vec![
        (
            Regex::new(r"<h1[^>]*>([^<]*)</h1>").expect("valid regex"),
            "\n# $1\n",
        ),
        (
            Regex::new(r"<h2[^>]*>([^<]*)</h2>").expect("valid regex"),
            "\n## $1\n",
        ),
        (
            Regex::new(r"<h3[^>]*>([^<]*)</h3>").expect("valid regex"),
            "\n### $1\n",
        ),
        (
            Regex::new(r"<h4[^>]*>([^<]*)</h4>").expect("valid regex"),
            "\n#### $1\n",
        ),
        (Regex::new(r"<p[^>]*>").expect("valid regex"), "\n"),
        (Regex::new(r"<br\s*/?>").expect("valid regex"), "\n"),
        (Regex::new(r"<li[^>]*>").expect("valid regex"), "\n- "),
        (
            Regex::new(r#"<a[^>]*href=["']([^"']*)["'][^>]*>([^<]*)</a>"#).expect("valid regex"),
            "[$2]($1)",
        ),
        (
            Regex::new(r"<strong[^>]*>([^<]*)</strong>").expect("valid regex"),
            "**$1**",
        ),
        (
            Regex::new(r"<b[^>]*>([^<]*)</b>").expect("valid regex"),
            "**$1**",
        ),
        (
            Regex::new(r"<em[^>]*>([^<]*)</em>").expect("valid regex"),
            "*$1*",
        ),
        (
            Regex::new(r"<i[^>]*>([^<]*)</i>").expect("valid regex"),
            "*$1*",
        ),
        (
            Regex::new(r"<code[^>]*>([^<]*)</code>").expect("valid regex"),
            "`$1`",
        ),
    ]
});

impl VisitWebpageTool {
    /// Create a new webpage visitor tool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum output length.
    #[must_use]
    pub const fn with_max_output_length(mut self, max: usize) -> Self {
        self.max_output_length = max;
        self
    }

    /// Set request timeout.
    #[must_use]
    pub const fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Truncate content to max length.
    fn truncate_content(&self, content: &str) -> String {
        if content.len() <= self.max_output_length {
            content.to_string()
        } else {
            format!(
                "{}...\n\n_Content truncated to {} characters_",
                &content[..self.max_output_length],
                self.max_output_length
            )
        }
    }

    /// Convert HTML to simple text/markdown.
    fn html_to_text(html: &str) -> String {
        // Remove scripts and styles
        let text = SCRIPT_RE.replace_all(html, "");
        let text = STYLE_RE.replace_all(&text, "");
        let mut text = text.into_owned();

        // Convert common HTML elements to markdown using pre-compiled patterns
        for (re, replacement) in HTML_PATTERNS.iter() {
            text = re.replace_all(&text, *replacement).into_owned();
        }

        // Handle trivial closing tags with simple string replacement
        text = text.replace("</p>", "\n").replace("</li>", "");

        // Remove remaining HTML tags
        text = TAG_RE.replace_all(&text, "").into_owned();

        // Decode common HTML entities
        text = text
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&apos;", "'")
            .replace("&nbsp;", " ")
            .replace("&#39;", "'");

        // Clean up whitespace
        text = MULTILINE_RE.replace_all(&text, "\n\n").into_owned();

        text.trim().to_string()
    }
}

#[async_trait]
impl Tool for VisitWebpageTool {
    const NAME: &'static str = "visit_webpage";
    type Args = VisitWebpageArgs;
    type Output = String;
    type Error = ToolError;

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn description(&self) -> String {
        "Visits a webpage at the given URL and reads its content as a markdown string. Use this to browse webpages.".to_string()
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "format": "uri",
                    "description": "The URL of the webpage to visit (must be a valid HTTP/HTTPS URL)"
                }
            },
            "required": ["url"]
        })
    }

    fn output_type(&self) -> &'static str {
        "string"
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // Validate URL format
        if !args.url.starts_with("http://") && !args.url.starts_with("https://") {
            return Err(ToolError::InvalidArguments(
                "URL must start with http:// or https://".to_string(),
            ));
        }

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            .build()
            .map_err(|e| ToolError::ExecutionError(e.to_string()))?;

        let response = client.get(&args.url).send().await.map_err(|e| {
            if e.is_timeout() {
                ToolError::ExecutionError("Request timed out. Please try again later.".to_string())
            } else {
                ToolError::ExecutionError(format!("Error fetching webpage: {e}"))
            }
        })?;

        if !response.status().is_success() {
            return Err(ToolError::ExecutionError(format!(
                "HTTP error: {}",
                response.status()
            )));
        }

        let html = response
            .text()
            .await
            .map_err(|e| ToolError::ExecutionError(format!("Failed to read response: {e}")))?;

        let text = Self::html_to_text(&html);
        Ok(self.truncate_content(&text))
    }
}
