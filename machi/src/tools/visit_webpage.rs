//! Tool for visiting and reading webpage content.

use crate::tool::{Tool, ToolError};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Tool for visiting a webpage and extracting its content as text.
#[derive(Debug, Clone, Copy)]
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
pub struct VisitWebpageArgs {
    /// The URL of the webpage to visit.
    pub url: String,
}

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
        let mut text = html.to_string();

        // Remove scripts and styles
        let script_re = regex::Regex::new(r"(?is)<script[^>]*>.*?</script>").ok();
        let style_re = regex::Regex::new(r"(?is)<style[^>]*>.*?</style>").ok();

        if let Some(re) = script_re {
            text = re.replace_all(&text, "").to_string();
        }
        if let Some(re) = style_re {
            text = re.replace_all(&text, "").to_string();
        }

        // Convert common HTML elements to markdown
        let replacements = [
            (r"<h1[^>]*>([^<]*)</h1>", "\n# $1\n"),
            (r"<h2[^>]*>([^<]*)</h2>", "\n## $1\n"),
            (r"<h3[^>]*>([^<]*)</h3>", "\n### $1\n"),
            (r"<h4[^>]*>([^<]*)</h4>", "\n#### $1\n"),
            (r"<p[^>]*>", "\n"),
            (r"</p>", "\n"),
            (r"<br\s*/?>", "\n"),
            (r"<li[^>]*>", "\n- "),
            (r"</li>", ""),
            (
                r#"<a[^>]*href=["']([^"']*)["'][^>]*>([^<]*)</a>"#,
                "[$2]($1)",
            ),
            (r"<strong[^>]*>([^<]*)</strong>", "**$1**"),
            (r"<b[^>]*>([^<]*)</b>", "**$1**"),
            (r"<em[^>]*>([^<]*)</em>", "*$1*"),
            (r"<i[^>]*>([^<]*)</i>", "*$1*"),
            (r"<code[^>]*>([^<]*)</code>", "`$1`"),
        ];

        for (pattern, replacement) in replacements {
            if let Ok(re) = regex::Regex::new(pattern) {
                text = re.replace_all(&text, replacement).to_string();
            }
        }

        // Remove remaining HTML tags
        if let Ok(re) = regex::Regex::new(r"<[^>]+>") {
            text = re.replace_all(&text, "").to_string();
        }

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
        let multiline_re = regex::Regex::new(r"\n{3,}").ok();
        if let Some(re) = multiline_re {
            text = re.replace_all(&text, "\n\n").to_string();
        }

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
                    "description": "The URL of the webpage to visit"
                }
            },
            "required": ["url"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
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
