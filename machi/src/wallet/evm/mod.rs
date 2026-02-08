//! EVM wallet backed by [`kobe_eth`] derivation and [`alloy`] signing/RPC.
//!
//! See [`EvmWallet`] for construction and usage.

mod ens;
mod erc20;
pub(crate) mod tools;
mod wallet;
#[cfg(feature = "x402")]
pub mod x402;

pub use wallet::EvmWallet;
