//! Agent with tool calling example using OpenAI.
//!
//! Demonstrates how to define a custom tool and let the agent
//! invoke it autonomously during its reasoning loop.
//!
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! cargo run --example agent_openai_tools
//! ```

#![allow(clippy::print_stdout)]

use async_trait::async_trait;
use machi::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

/// A simple weather tool that returns mock data.
struct GetWeather;

#[derive(Deserialize)]
struct WeatherArgs {
    city: String,
}

#[async_trait]
impl Tool for GetWeather {
    const NAME: &'static str = "get_weather";
    type Args = WeatherArgs;
    type Output = Value;
    type Error = ToolError;

    fn description(&self) -> String {
        "Get the current weather for a city.".into()
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name"
                }
            },
            "required": ["city"],
            "additionalProperties": false
        })
    }

    async fn call(&self, args: WeatherArgs) -> std::result::Result<Value, ToolError> {
        // In a real application, this would call a weather API.
        Ok(json!({
            "city": args.city,
            "temperature": "22°C",
            "condition": "Sunny"
        }))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let provider: SharedChatProvider = Arc::new(OpenAI::from_env()?);

    let agent = Agent::new("weather-bot")
        .instructions("You help users check the weather. Use the get_weather tool to answer.")
        .model("gpt-4o-mini")
        .provider(provider)
        .tool(Box::new(GetWeather));

    let result = agent
        .run("What's the weather like in Tokyo?", RunConfig::default())
        .await?;

    println!("{}", result.output);
    println!("\n Usage: {}", result.usage);

    Ok(())
}
