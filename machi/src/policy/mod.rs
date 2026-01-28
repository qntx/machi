//! Execution policies for agent actions.
//!
//! Policies control what actions an agent is allowed to perform autonomously
//! and what requires human approval.

use crate::chain::TransactionRequest;

/// A policy decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// Allow the action.
    Allow,
    /// Deny the action with a reason.
    Deny(String),
    /// Require human approval.
    RequireApproval(String),
}

/// Trait for execution policies.
///
/// Implement this trait to create custom policies that control
/// what actions agents can perform.
pub trait Policy: Send + Sync {
    /// Check if a transaction is allowed.
    fn check_transaction(&self, chain: &str, tx: &TransactionRequest) -> Decision;
}

/// A permissive policy that allows all actions.
#[derive(Debug, Clone, Default)]
pub struct AllowAll;

impl Policy for AllowAll {
    fn check_transaction(&self, _chain: &str, _tx: &TransactionRequest) -> Decision {
        Decision::Allow
    }
}

/// A policy that denies all transactions.
#[derive(Debug, Clone, Default)]
pub struct DenyAll;

impl Policy for DenyAll {
    fn check_transaction(&self, _chain: &str, _tx: &TransactionRequest) -> Decision {
        Decision::Deny("All transactions are denied by policy".into())
    }
}

/// A policy with spending limits.
#[derive(Debug, Clone)]
pub struct SpendingLimit {
    /// Maximum value per transaction (in smallest units).
    pub max_per_tx: u128,
    /// Allowed recipient addresses (empty = allow all).
    pub whitelist: Vec<String>,
}

impl SpendingLimit {
    /// Create a new spending limit policy.
    pub const fn new(max_per_tx: u128) -> Self {
        Self {
            max_per_tx,
            whitelist: Vec::new(),
        }
    }

    /// Add addresses to the whitelist.
    pub fn with_whitelist(mut self, addresses: Vec<String>) -> Self {
        self.whitelist = addresses;
        self
    }
}

impl Policy for SpendingLimit {
    fn check_transaction(&self, _chain: &str, tx: &TransactionRequest) -> Decision {
        if tx.value > self.max_per_tx {
            return Decision::RequireApproval(format!(
                "Transaction value {} exceeds limit {}",
                tx.value, self.max_per_tx
            ));
        }

        if !self.whitelist.is_empty() && !self.whitelist.contains(&tx.to) {
            return Decision::RequireApproval(format!("Recipient {} is not in whitelist", tx.to));
        }

        Decision::Allow
    }
}
