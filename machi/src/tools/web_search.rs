//! Web search tool for agents.
//!
//! Provides a configurable web search tool backed by pluggable search providers.
//! Built-in providers include [Tavily](https://tavily.com),
//! [SearXNG](https://docs.searxng.org/), and
//! [Brave Search](https://brave.com/search/api/).
//!
//! # Architecture
//!
//! ```text
//! WebSearchTool (implements Tool)
//!   └── dyn SearchProvider
//!         ├── TavilyProvider   (AI-optimised, requires API key)
//!         ├── SearxngProvider  (self-hosted, no API key)
//!         └── BraveProvider    (requires API key)
//! ```
//!
//! Users can implement [`SearchProvider`] to add custom backends.
//!
//! # Examples
//!
//! ```rust
//! use machi::tools::WebSearchTool;
//!
//! let tool = WebSearchTool::tavily("tvly-...")
//!     .with_max_results(5);
//! ```

use std::fmt;
use std::fmt::Write as _;
use std::sync::LazyLock;

use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Percent-encode a query string value (minimal subset for URL safety).
fn url_encode(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            _ => {
                let _ = write!(out, "%{byte:02X}");
            }
        }
    }
    out
}

use crate::error::ToolError;
use crate::tool::Tool;

/// A single web search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SearchResult {
    /// Title of the search result.
    pub title: String,
    /// URL of the search result.
    pub url: String,
    /// Snippet / description text.
    pub snippet: String,
}

impl fmt::Display for SearchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]({})\n{}", self.title, self.url, self.snippet)
    }
}

/// Pluggable search backend trait.
///
/// Implement this to add custom search providers to [`WebSearchTool`].
#[async_trait]
#[allow(clippy::unnecessary_literal_bound)]
pub trait SearchProvider: Send + Sync + fmt::Debug {
    /// A human-readable name for this provider (used in debug/tracing output).
    fn provider_name(&self) -> &str;

    /// Execute a search query and return up to `max_results` results.
    async fn search(&self, query: &str, max_results: usize)
    -> Result<Vec<SearchResult>, ToolError>;
}

/// A boxed search provider for dynamic dispatch.
pub type BoxedSearchProvider = Box<dyn SearchProvider>;

/// Search provider backed by the [Tavily](https://tavily.com) API.
///
/// Tavily is an AI-optimised search API that returns clean, relevant results.
/// A free-tier API key is available at <https://tavily.com>.
///
/// # Authentication
///
/// The API key is sent via the `Authorization: Bearer` header.
#[derive(Debug, Clone)]
pub struct TavilyProvider {
    api_key: String,
    client: reqwest::Client,
    search_depth: SearchDepth,
}

/// Tavily search depth.
#[derive(Debug, Clone, Copy, Default)]
pub enum SearchDepth {
    /// Fast, lower-cost search.
    #[default]
    Basic,
    /// Deeper, more thorough search.
    Advanced,
}

impl SearchDepth {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Basic => "basic",
            Self::Advanced => "advanced",
        }
    }
}

impl TavilyProvider {
    /// Create a new Tavily provider with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: reqwest::Client::new(),
            search_depth: SearchDepth::default(),
        }
    }

    /// Use the "advanced" search depth for higher-quality results.
    #[must_use]
    pub const fn with_advanced_depth(mut self) -> Self {
        self.search_depth = SearchDepth::Advanced;
        self
    }

    /// Provide a custom [`reqwest::Client`] (e.g. with proxy or timeout).
    #[must_use]
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }
}

// Tavily API response types (private).
#[derive(Deserialize)]
struct TavilyResponse {
    results: Vec<TavilyResult>,
}

#[derive(Deserialize)]
struct TavilyResult {
    title: String,
    url: String,
    content: String,
}

#[async_trait]
#[allow(clippy::unnecessary_literal_bound)]
impl SearchProvider for TavilyProvider {
    fn provider_name(&self) -> &str {
        "tavily"
    }

    async fn search(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, ToolError> {
        let body = serde_json::json!({
            "query": query,
            "max_results": max_results,
            "search_depth": self.search_depth.as_str(),
        });

        let response = self
            .client
            .post("https://api.tavily.com/search")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| ToolError::Execution(format!("Tavily request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(ToolError::Execution(format!(
                "Tavily API error (HTTP {status}): {text}"
            )));
        }

        let parsed: TavilyResponse = response
            .json()
            .await
            .map_err(|e| ToolError::Execution(format!("Failed to parse Tavily response: {e}")))?;

        Ok(parsed
            .results
            .into_iter()
            .map(|r| SearchResult {
                title: r.title,
                url: r.url,
                snippet: r.content,
            })
            .collect())
    }
}

/// Search provider backed by a [SearXNG](https://docs.searxng.org/) instance.
///
/// `SearXNG` is an open-source, self-hostable meta-search engine that aggregates
/// results from multiple search engines. **No API key required.**
///
/// # Examples
///
/// ```rust
/// use machi::tools::SearxngProvider;
///
/// let provider = SearxngProvider::new("https://searx.example.com");
/// ```
#[derive(Debug, Clone)]
pub struct SearxngProvider {
    base_url: String,
    client: reqwest::Client,
}

impl SearxngProvider {
    /// Create a new `SearXNG` provider pointing at the given instance URL.
    pub fn new(base_url: impl Into<String>) -> Self {
        let mut url = base_url.into();
        // Normalise: strip trailing slash
        if url.ends_with('/') {
            url.pop();
        }
        Self {
            base_url: url,
            client: reqwest::Client::new(),
        }
    }

    /// Provide a custom [`reqwest::Client`].
    #[must_use]
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }
}

// SearXNG JSON response types (private).
#[derive(Deserialize)]
struct SearxngResponse {
    results: Vec<SearxngResult>,
}

#[derive(Deserialize)]
struct SearxngResult {
    title: String,
    url: String,
    #[serde(default)]
    content: String,
}

#[async_trait]
#[allow(clippy::unnecessary_literal_bound)]
impl SearchProvider for SearxngProvider {
    fn provider_name(&self) -> &str {
        "searxng"
    }

    async fn search(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, ToolError> {
        let url = format!(
            "{}/search?q={}&format=json",
            self.base_url,
            url_encode(query),
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ToolError::Execution(format!("SearXNG request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(ToolError::Execution(format!(
                "SearXNG error (HTTP {status}): {text}"
            )));
        }

        let parsed: SearxngResponse = response
            .json()
            .await
            .map_err(|e| ToolError::Execution(format!("Failed to parse SearXNG response: {e}")))?;

        Ok(parsed
            .results
            .into_iter()
            .take(max_results)
            .map(|r| SearchResult {
                title: r.title,
                url: r.url,
                snippet: r.content,
            })
            .collect())
    }
}

/// Search provider backed by the [Brave Search](https://brave.com/search/api/) API.
///
/// A free-tier API key is available at <https://brave.com/search/api/>.
///
/// # Authentication
///
/// The API key is sent via the `X-Subscription-Token` header.
#[derive(Debug, Clone)]
pub struct BraveProvider {
    api_key: String,
    client: reqwest::Client,
}

impl BraveProvider {
    /// Create a new Brave Search provider with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: reqwest::Client::new(),
        }
    }

    /// Provide a custom [`reqwest::Client`].
    #[must_use]
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }
}

// Brave Search API response types (private).
#[derive(Deserialize)]
struct BraveResponse {
    web: Option<BraveWebResults>,
}

#[derive(Deserialize)]
struct BraveWebResults {
    results: Vec<BraveResult>,
}

#[derive(Deserialize)]
struct BraveResult {
    title: String,
    url: String,
    #[serde(default)]
    description: String,
}

#[async_trait]
#[allow(clippy::unnecessary_literal_bound)]
impl SearchProvider for BraveProvider {
    fn provider_name(&self) -> &str {
        "brave"
    }

    async fn search(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, ToolError> {
        let url = format!(
            "https://api.search.brave.com/res/v1/web/search?q={}&count={max_results}",
            url_encode(query),
        );

        let response = self
            .client
            .get(&url)
            .header("X-Subscription-Token", &self.api_key)
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| ToolError::Execution(format!("Brave Search request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(ToolError::Execution(format!(
                "Brave Search error (HTTP {status}): {text}"
            )));
        }

        let parsed: BraveResponse = response.json().await.map_err(|e| {
            ToolError::Execution(format!("Failed to parse Brave Search response: {e}"))
        })?;

        let results = parsed.web.map(|w| w.results).unwrap_or_default();

        Ok(results
            .into_iter()
            .take(max_results)
            .map(|r| SearchResult {
                title: r.title,
                url: r.url,
                snippet: r.description,
            })
            .collect())
    }
}

// Pre-compiled regex patterns for parsing DuckDuckGo Lite HTML.
static DDG_LINK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"class="result-link"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>"#)
        .expect("valid DDG link regex")
});

static DDG_SNIPPET_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"class="result-snippet"[^>]*>([^<]+)"#).expect("valid DDG snippet regex")
});

/// Search provider backed by [DuckDuckGo](https://duckduckgo.com) Lite.
///
/// Uses the `DuckDuckGo` Lite HTML interface �?**no API key required**.
///
/// > **Note:** DuckDuckGo may occasionally serve a CAPTCHA page instead of
/// > results. This provider works best for low-volume or infrequent queries.
///
/// # Examples
///
/// ```rust
/// use machi::tools::{DuckDuckGoProvider, WebSearchTool};
///
/// let provider = DuckDuckGoProvider::new();
/// let tool = WebSearchTool::new(provider);
/// ```
#[derive(Debug, Clone)]
pub struct DuckDuckGoProvider {
    client: reqwest::Client,
}

impl Default for DuckDuckGoProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl DuckDuckGoProvider {
    /// Create a new `DuckDuckGo` provider with a browser-like User-Agent.
    #[must_use]
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .user_agent(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
                 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            )
            .build()
            .unwrap_or_default();
        Self { client }
    }

    /// Provide a custom [`reqwest::Client`].
    #[must_use]
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    /// Parse `DuckDuckGo` Lite HTML into search results.
    fn parse_html(html: &str) -> Vec<SearchResult> {
        let links: Vec<_> = DDG_LINK_RE.captures_iter(html).collect();
        let snippets: Vec<_> = DDG_SNIPPET_RE.captures_iter(html).collect();

        links
            .iter()
            .enumerate()
            .filter_map(|(i, cap)| {
                let url = cap.get(1).map(|m| m.as_str()).unwrap_or_default();
                let title = cap.get(2).map(|m| m.as_str()).unwrap_or_default();
                let snippet = snippets
                    .get(i)
                    .and_then(|c| c.get(1))
                    .map(|m| m.as_str().trim())
                    .unwrap_or_default();

                if url.is_empty() || title.is_empty() {
                    None
                } else {
                    Some(SearchResult {
                        title: title.trim().to_owned(),
                        url: url.to_owned(),
                        snippet: snippet.to_owned(),
                    })
                }
            })
            .collect()
    }
}

#[async_trait]
#[allow(clippy::unnecessary_literal_bound)]
impl SearchProvider for DuckDuckGoProvider {
    fn provider_name(&self) -> &str {
        "duckduckgo"
    }

    async fn search(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, ToolError> {
        let url = format!("https://lite.duckduckgo.com/lite/?q={}", url_encode(query));

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ToolError::Execution(format!("DuckDuckGo request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(ToolError::Execution(format!(
                "DuckDuckGo error (HTTP {status}): {text}"
            )));
        }

        let html = response.text().await.map_err(|e| {
            ToolError::Execution(format!("Failed to read DuckDuckGo response: {e}"))
        })?;

        Ok(Self::parse_html(&html)
            .into_iter()
            .take(max_results)
            .collect())
    }
}

// Pre-compiled regex for parsing Bing RSS XML items.
static RSS_ITEM_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"<item>.*?<title>([^<]*)</title>.*?<link>([^<]*)</link>.*?<description>([^<]*)</description>.*?</item>",
    )
    .expect("valid RSS item regex")
});

/// Search provider backed by [Bing](https://www.bing.com) RSS feed.
///
/// Uses Bing's public RSS search endpoint �?**no API key required**.
///
/// # Examples
///
/// ```rust
/// use machi::tools::{BingProvider, WebSearchTool};
///
/// let provider = BingProvider::new();
/// let tool = WebSearchTool::new(provider);
/// ```
#[derive(Debug, Clone)]
pub struct BingProvider {
    client: reqwest::Client,
}

impl Default for BingProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl BingProvider {
    /// Create a new Bing RSS provider.
    #[must_use]
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    /// Provide a custom [`reqwest::Client`].
    #[must_use]
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    /// Parse Bing RSS XML into search results.
    fn parse_rss(xml: &str) -> Vec<SearchResult> {
        RSS_ITEM_RE
            .captures_iter(xml)
            .map(|cap| SearchResult {
                title: cap
                    .get(1)
                    .map(|m| m.as_str())
                    .unwrap_or_default()
                    .to_owned(),
                url: cap
                    .get(2)
                    .map(|m| m.as_str())
                    .unwrap_or_default()
                    .to_owned(),
                snippet: cap
                    .get(3)
                    .map(|m| m.as_str())
                    .unwrap_or_default()
                    .to_owned(),
            })
            .collect()
    }
}

#[async_trait]
#[allow(clippy::unnecessary_literal_bound)]
impl SearchProvider for BingProvider {
    fn provider_name(&self) -> &str {
        "bing"
    }

    async fn search(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, ToolError> {
        let url = format!(
            "https://www.bing.com/search?q={}&format=rss",
            url_encode(query),
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ToolError::Execution(format!("Bing request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(ToolError::Execution(format!(
                "Bing error (HTTP {status}): {text}"
            )));
        }

        let xml = response
            .text()
            .await
            .map_err(|e| ToolError::Execution(format!("Failed to read Bing response: {e}")))?;

        Ok(Self::parse_rss(&xml)
            .into_iter()
            .take(max_results)
            .collect())
    }
}

/// Arguments for [`WebSearchTool`].
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct WebSearchArgs {
    /// The search query string.
    pub query: String,
    /// Override the default maximum number of results. Optional.
    pub max_results: Option<usize>,
}

/// Web search tool backed by a configurable [`SearchProvider`].
///
/// # Examples
///
/// ```rust
/// use machi::tools::WebSearchTool;
///
/// // Tavily (AI-optimised, requires API key)
/// let tool = WebSearchTool::tavily("tvly-...");
///
/// // SearXNG (self-hosted, no API key)
/// let tool = WebSearchTool::searxng("https://searx.example.com");
///
/// // Brave Search (requires API key)
/// let tool = WebSearchTool::brave("BSA...");
///
/// // DuckDuckGo (free, no API key)
/// let tool = WebSearchTool::duckduckgo();
///
/// // Bing RSS (free, no API key)
/// let tool = WebSearchTool::bing();
/// ```
pub struct WebSearchTool {
    provider: BoxedSearchProvider,
    max_results: usize,
}

impl fmt::Debug for WebSearchTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WebSearchTool")
            .field("provider", &self.provider.provider_name())
            .field("max_results", &self.max_results)
            .finish()
    }
}

impl WebSearchTool {
    /// Create a web search tool with the given provider.
    pub fn new(provider: impl SearchProvider + 'static) -> Self {
        Self {
            provider: Box::new(provider),
            max_results: 5,
        }
    }

    /// Create a web search tool backed by [Tavily](https://tavily.com).
    pub fn tavily(api_key: impl Into<String>) -> Self {
        Self::new(TavilyProvider::new(api_key))
    }

    /// Create a web search tool backed by a
    /// [SearXNG](https://docs.searxng.org/) instance.
    pub fn searxng(base_url: impl Into<String>) -> Self {
        Self::new(SearxngProvider::new(base_url))
    }

    /// Create a web search tool backed by
    /// [Brave Search](https://brave.com/search/api/).
    pub fn brave(api_key: impl Into<String>) -> Self {
        Self::new(BraveProvider::new(api_key))
    }

    /// Create a web search tool backed by
    /// [DuckDuckGo](https://duckduckgo.com) Lite. **No API key required.**
    #[must_use]
    pub fn duckduckgo() -> Self {
        Self::new(DuckDuckGoProvider::new())
    }

    /// Create a web search tool backed by
    /// [Bing](https://www.bing.com) RSS feed. **No API key required.**
    #[must_use]
    pub fn bing() -> Self {
        Self::new(BingProvider::new())
    }

    /// Set the default maximum number of results returned per query.
    #[must_use]
    pub const fn with_max_results(mut self, max: usize) -> Self {
        self.max_results = max;
        self
    }

    /// Format search results as numbered Markdown for LLM consumption.
    fn format_results(results: &[SearchResult]) -> String {
        if results.is_empty() {
            return "No results found.".to_owned();
        }

        let mut output = String::from("## Search Results\n\n");
        for (i, r) in results.iter().enumerate() {
            let _ = write!(
                output,
                "{}. [{}]({})\n{}\n\n",
                i + 1,
                r.title,
                r.url,
                r.snippet,
            );
        }
        output
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    const NAME: &'static str = "web_search";
    type Args = WebSearchArgs;
    type Output = String;
    type Error = ToolError;

    fn description(&self) -> String {
        "Search the web for information. Returns the top search results as formatted markdown."
            .to_owned()
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to perform"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Optional."
                }
            },
            "required": ["query"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if args.query.trim().is_empty() {
            return Err(ToolError::InvalidArguments(
                "Search query cannot be empty".into(),
            ));
        }

        let max = args.max_results.unwrap_or(self.max_results);
        let results = self.provider.search(&args.query, max).await?;

        if results.is_empty() {
            return Err(ToolError::Execution(
                "No results found. Try a different or less restrictive query.".into(),
            ));
        }

        Ok(Self::format_results(&results))
    }
}
