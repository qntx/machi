//! Web search tools for querying the internet.

use crate::tool::{Tool, ToolError};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Generic web search tool with configurable backend.
#[derive(Debug, Clone, Copy)]
pub struct WebSearchTool {
    /// Maximum number of results to return.
    pub max_results: usize,
    /// Search engine to use.
    pub engine: SearchEngine,
}

/// Supported search engines.
#[derive(Debug, Clone, Copy, Default)]
pub enum SearchEngine {
    #[default]
    DuckDuckGo,
    Bing,
}

impl Default for WebSearchTool {
    fn default() -> Self {
        Self {
            max_results: 10,
            engine: SearchEngine::default(),
        }
    }
}

/// Arguments for web search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSearchArgs {
    /// The search query to perform.
    pub query: String,
}

/// A single search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Title of the result.
    pub title: String,
    /// URL of the result.
    pub link: String,
    /// Description/snippet of the result.
    pub description: String,
}

impl WebSearchTool {
    /// Create a new web search tool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum results.
    #[must_use]
    pub const fn with_max_results(mut self, max: usize) -> Self {
        self.max_results = max;
        self
    }

    /// Set search engine.
    #[must_use]
    pub const fn with_engine(mut self, engine: SearchEngine) -> Self {
        self.engine = engine;
        self
    }

    /// Parse results into markdown format.
    fn format_results(results: &[SearchResult]) -> String {
        if results.is_empty() {
            return "No results found.".to_string();
        }

        let mut output = String::from("## Search Results\n\n");
        for result in results {
            output.push_str(&format!(
                "[{}]({})\n{}\n\n",
                result.title, result.link, result.description
            ));
        }
        output
    }

    /// Perform `DuckDuckGo` search using lite HTML interface.
    async fn search_duckduckgo(&self, query: &str) -> Result<Vec<SearchResult>, ToolError> {
        let client = reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            .build()
            .map_err(|e| ToolError::ExecutionError(e.to_string()))?;

        let url = format!(
            "https://lite.duckduckgo.com/lite/?q={}",
            urlencoding::encode(query)
        );

        let response = client
            .get(&url)
            .send()
            .await
            .map_err(|e| ToolError::ExecutionError(format!("Request failed: {e}")))?;

        let html = response
            .text()
            .await
            .map_err(|e| ToolError::ExecutionError(format!("Failed to read response: {e}")))?;

        // Simple HTML parsing for DuckDuckGo Lite results
        let results = self.parse_duckduckgo_html(&html);

        Ok(results.into_iter().take(self.max_results).collect())
    }

    /// Parse `DuckDuckGo` Lite HTML response.
    fn parse_duckduckgo_html(&self, html: &str) -> Vec<SearchResult> {
        let mut results = Vec::new();

        // Simple regex-based parsing for result links
        let link_re =
            regex::Regex::new(r#"class="result-link"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>"#).ok();
        let snippet_re = regex::Regex::new(r#"class="result-snippet"[^>]*>([^<]+)"#).ok();

        if let (Some(link_regex), Some(snippet_regex)) = (link_re, snippet_re) {
            let links: Vec<_> = link_regex.captures_iter(html).collect();
            let snippets: Vec<_> = snippet_regex.captures_iter(html).collect();

            for (i, link_cap) in links.iter().enumerate() {
                let url = link_cap.get(1).map(|m| m.as_str()).unwrap_or_default();
                let title = link_cap.get(2).map(|m| m.as_str()).unwrap_or_default();
                let description = snippets
                    .get(i)
                    .and_then(|c| c.get(1))
                    .map(|m| m.as_str())
                    .unwrap_or_default();

                if !url.is_empty() && !title.is_empty() {
                    results.push(SearchResult {
                        title: title.trim().to_string(),
                        link: url.to_string(),
                        description: description.trim().to_string(),
                    });
                }
            }
        }

        results
    }

    /// Perform Bing search using RSS feed.
    async fn search_bing(&self, query: &str) -> Result<Vec<SearchResult>, ToolError> {
        let client = reqwest::Client::new();
        let url = format!(
            "https://www.bing.com/search?q={}&format=rss",
            urlencoding::encode(query)
        );

        let response = client
            .get(&url)
            .send()
            .await
            .map_err(|e| ToolError::ExecutionError(format!("Request failed: {e}")))?;

        let xml = response
            .text()
            .await
            .map_err(|e| ToolError::ExecutionError(format!("Failed to read response: {e}")))?;

        // Simple XML parsing for RSS items
        let results = self.parse_rss_xml(&xml);

        Ok(results.into_iter().take(self.max_results).collect())
    }

    /// Parse RSS XML response.
    fn parse_rss_xml(&self, xml: &str) -> Vec<SearchResult> {
        let mut results = Vec::new();

        // Simple regex-based parsing for RSS items
        let item_re = regex::Regex::new(
            r"<item>.*?<title>([^<]*)</title>.*?<link>([^<]*)</link>.*?<description>([^<]*)</description>.*?</item>"
        ).ok();

        if let Some(regex) = item_re {
            for cap in regex.captures_iter(xml) {
                results.push(SearchResult {
                    title: cap
                        .get(1)
                        .map(|m| m.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    link: cap
                        .get(2)
                        .map(|m| m.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    description: cap
                        .get(3)
                        .map(|m| m.as_str())
                        .unwrap_or_default()
                        .to_string(),
                });
            }
        }

        results
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    const NAME: &'static str = "web_search";
    type Args = WebSearchArgs;
    type Output = String;
    type Error = ToolError;

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn description(&self) -> String {
        "Performs a web search for a query and returns the top search results formatted as markdown.".to_string()
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to perform"
                }
            },
            "required": ["query"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let results = match self.engine {
            SearchEngine::DuckDuckGo => self.search_duckduckgo(&args.query).await?,
            SearchEngine::Bing => self.search_bing(&args.query).await?,
        };

        if results.is_empty() {
            return Err(ToolError::ExecutionError(
                "No results found! Try a less restrictive/shorter query.".to_string(),
            ));
        }

        Ok(Self::format_results(&results))
    }
}

/// DuckDuckGo-specific search tool.
pub type DuckDuckGoSearchTool = WebSearchTool;
