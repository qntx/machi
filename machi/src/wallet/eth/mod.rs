//! EVM wallet backed by [`kobe_eth`] derivation and [`alloy`] signing/RPC.
//!
//! See [`EvmWallet`] for construction and usage.

pub(crate) mod tools;
mod wallet;

pub use wallet::EvmWallet;
