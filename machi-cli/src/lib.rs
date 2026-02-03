//! Machi CLI - Command line interface for the machi AI agents framework.
//!
//! This crate provides command-line tools for interacting with machi agents.

/// Placeholder function (to be implemented).
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
