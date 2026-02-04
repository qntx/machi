//! Machi Bot CLI - Personal AI Assistant
//!
//! A command-line interface for running the Machi Bot framework.

#![allow(clippy::print_stdout)] // CLI program intentionally uses stdout

use clap::{Args, Parser, Subcommand};
use machi_bot::error::{BotError, Result};
use machi_bot::prelude::*;
use std::path::PathBuf;
use std::process::ExitCode;
use tracing::Level;
use tracing_subscriber::EnvFilter;

/// Machi Bot - Personal AI Assistant with multi-channel support
#[derive(Parser)]
#[command(name = "machi-bot")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    /// Configuration file path
    #[arg(short, long, env = "MACHI_BOT_CONFIG", global = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize configuration and setup
    Init(InitArgs),

    /// Start the bot gateway (all channels + agent)
    Gateway(GatewayArgs),

    /// Start an interactive chat session
    Chat(ChatArgs),

    /// Show bot status and configuration
    Status,

    /// Manage configuration
    Config(ConfigArgs),
}

/// Arguments for the init command
#[derive(Args)]
struct InitArgs {
    /// Force overwrite existing configuration
    #[arg(short, long)]
    force: bool,
}

/// Arguments for the gateway command
#[derive(Args)]
struct GatewayArgs {
    /// Disable CLI channel
    #[arg(long)]
    no_cli: bool,

    /// Disable Telegram channel
    #[arg(long)]
    no_telegram: bool,

    /// Model to use (overrides config)
    #[arg(short, long, env = "MACHI_MODEL")]
    model: Option<String>,
}

/// Arguments for the chat command
#[derive(Args)]
struct ChatArgs {
    /// Initial message to send
    #[arg(short, long)]
    message: Option<String>,

    /// Model to use
    #[arg(short = 'M', long, env = "MACHI_MODEL")]
    model: Option<String>,

    /// Custom prompt prefix
    #[arg(short, long, default_value = "You: ")]
    prompt: String,

    /// Session ID for conversation persistence
    #[arg(short, long, default_value = "cli")]
    session: String,
}

/// Arguments for the config command
#[derive(Args)]
struct ConfigArgs {
    #[command(subcommand)]
    command: ConfigCommands,
}

#[derive(Subcommand)]
enum ConfigCommands {
    /// Show current configuration
    Show,
    /// Show configuration file path
    Path,
    /// Edit configuration in default editor
    Edit,
    /// Validate configuration
    Validate,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    // Initialize logging based on verbosity
    init_logging(cli.verbose);

    // Run the async main
    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");

    match rt.block_on(run(cli)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!("{e}");
            ExitCode::FAILURE
        }
    }
}

/// Initialize logging with the given verbosity level.
fn init_logging(verbosity: u8) {
    let level = match verbosity {
        0 => Level::INFO,
        1 => Level::DEBUG,
        _ => Level::TRACE,
    };

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(format!(
            "machi_bot={level},machi={level},{}",
            if verbosity >= 2 { "debug" } else { "warn" }
        ))
    });

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(verbosity >= 2)
        .init();
}

/// Main async entry point.
async fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Init(args) => cmd_init(args).await,
        Commands::Gateway(args) => cmd_gateway(args, cli.config).await,
        Commands::Chat(args) => cmd_chat(args, cli.config).await,
        Commands::Status => cmd_status(cli.config).await,
        Commands::Config(args) => cmd_config(args, cli.config).await,
    }
}

/// Initialize configuration.
async fn cmd_init(args: InitArgs) -> Result<()> {
    use machi_bot::config::{config_path, init_config};

    let config_file = config_path();

    if config_file.exists() && !args.force {
        println!("Configuration already exists at: {}", config_file.display());
        println!("Use --force to overwrite.");
        return Ok(());
    }

    init_config()
        .await
        .map_err(|e| BotError::config(format!("failed to initialize config: {e}")))?;

    println!("Configuration created: {}", config_file.display());
    println!();
    println!("Next steps:");
    println!("  1. machi-bot config edit");
    println!("  2. export ANTHROPIC_API_KEY=<key>");
    println!("  3. machi-bot gateway");

    Ok(())
}

/// Start the gateway.
async fn cmd_gateway(args: GatewayArgs, config_path: Option<PathBuf>) -> Result<()> {
    use machi_bot::config::load_config;

    tracing::info!("Starting Machi Bot Gateway...");

    // Load config
    let mut config = if let Some(path) = config_path {
        let content = tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| BotError::config(format!("failed to read config: {e}")))?;
        toml::from_str(&content)
            .map_err(|e| BotError::config(format!("failed to parse config: {e}")))?
    } else {
        load_config().await.unwrap_or_default()
    };

    // Override model if specified
    if let Some(model) = args.model {
        config.agents.defaults.model = model;
    }

    // Disable channels if requested
    if args.no_telegram {
        config.channels.telegram.enabled = false;
    }

    // Create model based on configuration
    let model = create_model(&config)?;

    // Build gateway
    let gateway = GatewayBuilder::new()
        .model(model)
        .bot_config(config)
        .enable_cli(!args.no_cli)
        .build();

    println!("Gateway running. Press Ctrl+C to stop.\n");

    // Run with graceful shutdown
    tokio::select! {
        result = gateway.run() => result,
        _ = tokio::signal::ctrl_c() => {
            println!("\nShutting down...");
            Ok(())
        }
    }
}

/// Start interactive chat.
async fn cmd_chat(args: ChatArgs, config_path: Option<PathBuf>) -> Result<()> {
    use machi_bot::channels::cli::CliChannelConfig;
    use machi_bot::config::load_config;
    use machi_bot::gateway::GatewayBuilder;

    // Load config
    let mut config = if let Some(path) = config_path {
        let content = tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| BotError::config(format!("failed to read config: {e}")))?;
        toml::from_str(&content).unwrap_or_default()
    } else {
        load_config().await.unwrap_or_default()
    };

    // Override model if specified
    if let Some(model) = args.model {
        config.agents.defaults.model = model;
    }

    // Create model based on configuration
    let model = create_model(&config)?;

    // Handle initial message
    if let Some(ref msg) = args.message {
        println!("You: {msg}");
    }

    // Configure CLI
    let _cli_config = CliChannelConfig::new()
        .prompt(args.prompt)
        .session_id(args.session);

    println!("Machi Bot Chat | type 'exit' to quit\n");

    // Build and run gateway with CLI only
    let gateway = GatewayBuilder::new()
        .model(model)
        .bot_config(config)
        .enable_cli(true)
        .build();

    // Run gateway
    gateway.run().await
}

/// Show status.
async fn cmd_status(config_path: Option<PathBuf>) -> Result<()> {
    use machi_bot::config::{config_path as default_config_path, load_config};

    let config_file = config_path.unwrap_or_else(default_config_path);

    println!("Machi Bot Status\n");

    // Configuration
    println!("Configuration:");
    println!("  Path:   {}", config_file.display());
    println!(
        "  Exists: {}",
        if config_file.exists() { "yes" } else { "no" }
    );

    if config_file.exists() {
        match load_config().await {
            Ok(config) => {
                println!("  Valid:  yes");
                println!();
                println!("Channels:");
                println!(
                    "  Telegram: {}",
                    if config.channels.telegram.enabled {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
                println!(
                    "  WhatsApp: {}",
                    if config.channels.whatsapp.enabled {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
                println!();
                println!("Agent:");
                println!("  Model:          {}", config.agents.defaults.model);
                println!(
                    "  Max iterations: {}",
                    config.agents.defaults.max_iterations
                );
            }
            Err(e) => {
                println!("  Valid:  no ({e})");
            }
        }
    }

    println!();
    println!("Environment:");
    print_env_status("ANTHROPIC_API_KEY");
    print_env_status("OPENAI_API_KEY");
    print_env_status("TELEGRAM_BOT_TOKEN");
    print_env_status("MACHI_MODEL");

    Ok(())
}

/// Configuration management.
async fn cmd_config(args: ConfigArgs, config_path: Option<PathBuf>) -> Result<()> {
    use machi_bot::config::{config_path as default_config_path, load_config};

    let config_file = config_path.unwrap_or_else(default_config_path);

    match args.command {
        ConfigCommands::Path => {
            println!("{}", config_file.display());
        }
        ConfigCommands::Show => {
            if config_file.exists() {
                let content = tokio::fs::read_to_string(&config_file)
                    .await
                    .map_err(|e| BotError::config(format!("failed to read config: {e}")))?;
                println!("{content}");
            } else {
                println!("Configuration file does not exist.");
                println!("Run 'machi-bot init' to create one.");
            }
        }
        ConfigCommands::Edit => {
            let editor = std::env::var("EDITOR").unwrap_or_else(|_| "notepad".to_string());
            std::process::Command::new(&editor)
                .arg(&config_file)
                .status()
                .map_err(|e| BotError::config(format!("failed to open editor: {e}")))?;
        }
        ConfigCommands::Validate => {
            if !config_file.exists() {
                println!("error: configuration file does not exist");
                return Ok(());
            }

            match load_config().await {
                Ok(_) => println!("Configuration is valid"),
                Err(e) => println!("error: {e}"),
            }
        }
    }

    Ok(())
}

/// Model provider enum to support multiple backends.
#[derive(Clone)]
enum ModelProvider {
    Ollama(machi::providers::ollama::CompletionModel),
    Anthropic(machi::providers::anthropic::CompletionModel),
    OpenAI(machi::providers::openai::CompletionModel),
}

#[async_trait::async_trait]
impl machi::prelude::Model for ModelProvider {
    fn model_id(&self) -> &str {
        match self {
            Self::Ollama(m) => m.model_id(),
            Self::Anthropic(m) => m.model_id(),
            Self::OpenAI(m) => m.model_id(),
        }
    }

    async fn generate(
        &self,
        messages: Vec<machi::message::ChatMessage>,
        options: machi::providers::common::GenerateOptions,
    ) -> std::result::Result<machi::providers::common::ModelResponse, machi::error::AgentError>
    {
        match self {
            Self::Ollama(m) => m.generate(messages, options).await,
            Self::Anthropic(m) => m.generate(messages, options).await,
            Self::OpenAI(m) => m.generate(messages, options).await,
        }
    }

    fn supports_streaming(&self) -> bool {
        match self {
            Self::Ollama(m) => m.supports_streaming(),
            Self::Anthropic(m) => m.supports_streaming(),
            Self::OpenAI(m) => m.supports_streaming(),
        }
    }

    fn supports_tool_calling(&self) -> bool {
        match self {
            Self::Ollama(m) => m.supports_tool_calling(),
            Self::Anthropic(m) => m.supports_tool_calling(),
            Self::OpenAI(m) => m.supports_tool_calling(),
        }
    }
}

/// Create model based on configuration.
///
/// Priority:
/// 1. Ollama (if configured)
/// 2. Anthropic (if API key available)
/// 3. OpenAI (if API key available)
fn create_model(config: &BotConfig) -> Result<ModelProvider> {
    use machi::prelude::*;

    let model_name = &config.agents.defaults.model;

    // Check Ollama first (local, no API key needed)
    if let Some(ref ollama_config) = config.providers.ollama {
        tracing::info!(model = %model_name, "Using Ollama backend");
        let client = OllamaClient::builder()
            .base_url(&ollama_config.api_base)
            .build();
        return Ok(ModelProvider::Ollama(client.completion_model(model_name)));
    }

    // Check Anthropic
    if let Some(ref anthropic_config) = config.providers.anthropic {
        tracing::info!(model = %model_name, "Using Anthropic backend");
        return Ok(ModelProvider::Anthropic(
            AnthropicClient::new(&anthropic_config.api_key).completion_model(model_name),
        ));
    }

    // Check environment variable for Anthropic
    if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
        tracing::info!(model = %model_name, "Using Anthropic backend (from env)");
        return Ok(ModelProvider::Anthropic(
            AnthropicClient::new(&api_key).completion_model(model_name),
        ));
    }

    // Check OpenAI
    if let Some(ref openai_config) = config.providers.openai {
        tracing::info!(model = %model_name, "Using OpenAI backend");
        let mut client = OpenAIClient::new(&openai_config.api_key);
        if let Some(ref base_url) = openai_config.api_base {
            client = OpenAIClient::builder()
                .api_key(&openai_config.api_key)
                .base_url(base_url)
                .build();
        }
        return Ok(ModelProvider::OpenAI(client.completion_model(model_name)));
    }

    // Check environment variable for OpenAI
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        tracing::info!(model = %model_name, "Using OpenAI backend (from env)");
        return Ok(ModelProvider::OpenAI(
            OpenAIClient::new(&api_key).completion_model(model_name),
        ));
    }

    Err(BotError::config(
        "No LLM provider configured. Set up Ollama, or provide ANTHROPIC_API_KEY/OPENAI_API_KEY",
    ))
}

/// Print environment variable status.
fn print_env_status(name: &str) {
    let status = if std::env::var(name).is_ok() {
        "set"
    } else {
        "-"
    };
    println!("  {name}: {status}");
}
