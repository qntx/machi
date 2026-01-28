//! Core agent implementation.
//!
//! The [`Agent`] struct is the main entry point for creating AI agents
//! with embedded wallet capabilities.

use serde_json::Value;

use crate::backend::{Backend, BackendResponse, ToolResult as BackendToolResult};
use crate::chain::{Chain, TransactionRequest, TxHash};
use crate::error::{Error, Result};
use crate::policy::{AllowAll, Decision, Policy};
use crate::tools::Tool;
use crate::tools::wallet::{GetAddress, GetBalance, SendTransaction};
use crate::tools::{ToolContext, ToolDefinition, ToolOutput, builtin_tool_definitions};
use crate::wallet::AgentWallet;

/// Maximum number of tool call iterations to prevent infinite loops.
const MAX_TOOL_ITERATIONS: usize = 10;

/// An AI agent with embedded wallet identity.
///
/// The agent combines an LLM backend for reasoning with a wallet for
/// blockchain interactions, controlled by a policy.
pub struct Agent<B, C, P = AllowAll> {
    /// The agent's wallet identity.
    wallet: AgentWallet,
    /// The LLM backend for reasoning.
    backend: B,
    /// The blockchain adapter.
    chain: C,
    /// The execution policy.
    policy: P,
}

impl<B, C, P> Agent<B, C, P>
where
    B: Backend,
    C: Chain,
    P: Policy,
{
    /// Get a reference to the agent's wallet.
    #[inline]
    pub const fn wallet(&self) -> &AgentWallet {
        &self.wallet
    }

    /// Get a reference to the backend.
    #[inline]
    pub const fn backend(&self) -> &B {
        &self.backend
    }

    /// Get a reference to the chain adapter.
    #[inline]
    pub const fn chain(&self) -> &C {
        &self.chain
    }

    /// Get the agent's address on the configured chain.
    pub fn address(&self) -> Result<C::Address> {
        self.chain
            .derive_address(self.wallet.inner(), self.wallet.default_index())
    }

    /// Get the agent's balance on the configured chain.
    pub async fn balance(&self) -> Result<u128> {
        let address = self.address()?;
        self.chain.balance(address.as_ref()).await
    }

    /// Send a simple prompt to the LLM backend.
    pub async fn chat(&self, prompt: &str) -> Result<String> {
        self.backend.complete(prompt).await
    }

    /// Send a prompt with wallet tools available.
    pub async fn chat_with_tools(&self, prompt: &str) -> Result<BackendResponse>
    where
        C: 'static,
    {
        let tools = self.available_tools();
        self.backend.complete_with_tools(prompt, &tools).await
    }

    /// Get the list of available wallet tools.
    pub fn available_tools(&self) -> Vec<ToolDefinition> {
        builtin_tool_definitions()
    }

    /// Execute a tool by name with the given arguments.
    pub async fn execute_tool(&self, name: &str, args: Value) -> ToolOutput
    where
        C: 'static,
    {
        let ctx = ToolContext {
            wallet: &self.wallet,
            chain: &self.chain,
        };

        match name {
            "get_address" => GetAddress.call(&ctx, args).await,
            "get_balance" => GetBalance.call(&ctx, args).await,
            "send_transaction" => SendTransaction.call(&ctx, args).await,
            _ => ToolOutput::err(format!("Unknown tool: {name}")),
        }
    }

    /// Run the agent with a user prompt, handling tool calls automatically.
    ///
    /// This is the main entry point for interacting with the agent. It:
    /// 1. Sends the prompt to the LLM with available tools
    /// 2. If LLM requests tool calls, executes them
    /// 3. Sends results back to LLM
    /// 4. Repeats until LLM gives a final text response
    ///
    /// # Example
    ///
    /// ```ignore
    /// let response = agent.run("What's my wallet balance?").await?;
    /// println!("Agent: {}", response);
    /// ```
    pub async fn run(&self, prompt: &str) -> Result<String>
    where
        C: 'static,
    {
        let tools = self.available_tools();
        let mut response = self.backend.complete_with_tools(prompt, &tools).await?;
        let mut iterations = 0;

        loop {
            match response {
                BackendResponse::Text(text) => {
                    return Ok(text);
                }
                BackendResponse::ToolCalls(ref tool_calls) => {
                    iterations += 1;
                    if iterations > MAX_TOOL_ITERATIONS {
                        return Err(Error::Backend(
                            "Maximum tool call iterations exceeded".into(),
                        ));
                    }

                    // Execute all tool calls
                    let mut results = Vec::new();
                    for call in tool_calls {
                        let tool_output =
                            self.execute_tool(&call.name, call.arguments.clone()).await;
                        let content = tool_output.to_string();
                        results.push(BackendToolResult::new(&call.id, content));
                    }

                    // Continue conversation with tool results
                    response = self
                        .backend
                        .continue_with_tool_results(&results, &tools)
                        .await?;
                }
            }
        }
    }

    /// Execute a send transaction with policy check.
    pub async fn send(&self, to: &str, value: u128) -> Result<TxHash> {
        let tx = TransactionRequest {
            to: to.to_string(),
            value,
            data: None,
        };

        // Check policy
        match self.policy.check_transaction(self.chain.name(), &tx) {
            Decision::Allow => {}
            Decision::Deny(reason) => return Err(Error::Policy(reason)),
            Decision::RequireApproval(reason) => {
                return Err(Error::Policy(format!("approval required: {reason}")));
            }
        }

        self.chain
            .send_transaction(self.wallet.inner(), self.wallet.default_index(), tx)
            .await
    }
}

/// Builder for creating agents.
///
/// Use this to configure and construct an [`Agent`] with the desired
/// backend, chain, and policy.
pub struct AgentBuilder<B, C, P> {
    wallet: Option<AgentWallet>,
    backend: Option<B>,
    chain: Option<C>,
    policy: Option<P>,
}

impl AgentBuilder<(), (), ()> {
    /// Create a new agent builder.
    pub fn new() -> AgentBuilder<(), (), AllowAll> {
        AgentBuilder {
            wallet: None,
            backend: None,
            chain: None,
            policy: Some(AllowAll),
        }
    }
}

impl Default for AgentBuilder<(), (), AllowAll> {
    fn default() -> Self {
        AgentBuilder::new()
    }
}

impl<B, C, P> AgentBuilder<B, C, P> {
    /// Set the LLM backend.
    pub fn backend<NewB: Backend>(self, backend: NewB) -> AgentBuilder<NewB, C, P> {
        AgentBuilder {
            wallet: self.wallet,
            backend: Some(backend),
            chain: self.chain,
            policy: self.policy,
        }
    }

    /// Set the blockchain adapter.
    pub fn chain<NewC: Chain>(self, chain: NewC) -> AgentBuilder<B, NewC, P> {
        AgentBuilder {
            wallet: self.wallet,
            backend: self.backend,
            chain: Some(chain),
            policy: self.policy,
        }
    }

    /// Set the execution policy.
    pub fn policy<NewP: Policy>(self, policy: NewP) -> AgentBuilder<B, C, NewP> {
        AgentBuilder {
            wallet: self.wallet,
            backend: self.backend,
            chain: self.chain,
            policy: Some(policy),
        }
    }

    /// Use an existing wallet.
    pub fn wallet(mut self, wallet: AgentWallet) -> Self {
        self.wallet = Some(wallet);
        self
    }

    /// Generate a new wallet with the specified word count.
    pub fn generate_wallet(mut self, word_count: usize) -> Result<Self> {
        self.wallet = Some(AgentWallet::generate(word_count, None)?);
        Ok(self)
    }

    /// Generate a new wallet with a passphrase.
    pub fn generate_wallet_with_passphrase(
        mut self,
        word_count: usize,
        passphrase: &str,
    ) -> Result<Self> {
        self.wallet = Some(AgentWallet::generate(word_count, Some(passphrase))?);
        Ok(self)
    }

    /// Import a wallet from a mnemonic phrase.
    pub fn import_wallet(mut self, mnemonic: &str, passphrase: Option<&str>) -> Result<Self> {
        self.wallet = Some(AgentWallet::from_mnemonic(mnemonic, passphrase)?);
        Ok(self)
    }
}

impl<B: Backend, C: Chain, P: Policy> AgentBuilder<B, C, P> {
    /// Build the agent.
    ///
    /// # Errors
    ///
    /// Returns an error if required components are missing.
    pub fn build(self) -> Result<Agent<B, C, P>> {
        let wallet = self
            .wallet
            .ok_or_else(|| Error::Config("Wallet is required".into()))?;
        let backend = self
            .backend
            .ok_or_else(|| Error::Config("Backend is required".into()))?;
        let chain = self
            .chain
            .ok_or_else(|| Error::Config("Chain is required".into()))?;
        let policy = self
            .policy
            .ok_or_else(|| Error::Config("Policy is required".into()))?;

        Ok(Agent {
            wallet,
            backend,
            chain,
            policy,
        })
    }
}
