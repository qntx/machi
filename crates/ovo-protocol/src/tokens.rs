//! Rough token estimation for preflight overflow checks.
//!
//!
//! Heuristic: text bytes / 4; images fixed at [`IMAGE_TOKEN_COST`].

/// Fixed token cost for one image part (provider-agnostic estimate).
pub const IMAGE_TOKEN_COST: u32 = 765;

/// Approximate tokens for a UTF-8 string: `ceil(bytes / 4)`.
#[must_use]
pub fn estimate_text_tokens(text: &str) -> u32 {
    let bytes = u32::try_from(text.len()).unwrap_or(u32::MAX);
    bytes.div_ceil(4)
}

/// Approximate tokens for one image attachment.
#[must_use]
pub const fn estimate_image_tokens() -> u32 {
    IMAGE_TOKEN_COST
}

/// Framing overhead tokens per message (role + separators).
pub const MESSAGE_FRAME_TOKENS: u32 = 4;

/// Preflight decision before sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreflightOverflow {
    /// Under threshold.
    Ok {
        /// Estimated tokens.
        estimated: u32,
    },
    /// Over threshold; caller should compact or fail.
    Overflow {
        /// Estimated tokens.
        estimated: u32,
        /// Context window (tokens).
        window: u32,
        /// Soft threshold used (`window * threshold_ratio`).
        limit: u32,
    },
}

/// Check whether `estimated` exceeds `window * threshold_ratio` (clamped).
///
/// `threshold_ratio` is typically `0.85`–`0.95`.
#[must_use]
pub fn check_context_overflow(
    estimated: u32,
    window: u32,
    threshold_ratio: f32,
) -> PreflightOverflow {
    if window == 0 {
        return PreflightOverflow::Ok { estimated };
    }
    let ratio = threshold_ratio.clamp(0.1, 1.0);
    // Integer math: ceil(window * ratio) = (window * numer + denom - 1) / denom
    // with ratio ≈ numer/1000 for millis precision.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "token limits are u32; ratio is clamped to 0.1..=1.0"
    )]
    let numer = (ratio * 1000.0).round() as u64;
    let numer = numer.clamp(100, 1000);
    let limit = (u64::from(window).saturating_mul(numer).saturating_add(999) / 1000)
        .min(u64::from(u32::MAX));
    let limit = u32::try_from(limit).unwrap_or(u32::MAX);
    if estimated > limit {
        PreflightOverflow::Overflow {
            estimated,
            window,
            limit,
        }
    } else {
        PreflightOverflow::Ok { estimated }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_estimate_bytes_over_four() {
        assert_eq!(estimate_text_tokens("abcd"), 1);
        assert_eq!(estimate_text_tokens("abcde"), 2);
    }

    #[test]
    fn overflow_threshold() {
        let est = 900;
        match check_context_overflow(est, 1000, 0.85) {
            PreflightOverflow::Overflow { limit, .. } => {
                assert_eq!(limit, 850);
            }
            PreflightOverflow::Ok { .. } => unreachable!("expected overflow"),
        }
        assert!(matches!(
            check_context_overflow(800, 1000, 0.85),
            PreflightOverflow::Ok { .. }
        ));
    }
}
