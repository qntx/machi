// per-keep_tail compaction contract (must compact + invariant).
#[cfg(test)]
#[allow(clippy::expect_used, clippy::missing_assert_message, clippy::panic, reason = "matrix")]
mod select_matrix {
    use ovo_types::{Message, Role, ToolCall, ToolCallId};
    use serde_json::json;
    use super::{apply_range, select_compaction_range, tool_pair_invariant_holds};

    fn conversation() -> Vec<Message> {
        let mut msgs = vec![Message::system("s")];
        for i in 0..20 {
            let id = ToolCallId::new(format!("c{i}")).expect("id");
            msgs.push(Message::user(format!("u{i}")));
            msgs.push(Message::assistant_tools(vec![ToolCall {
                id: id.clone(),
                name: "x".into(),
                arguments: json!({}),
            }]));
            msgs.push(Message::tool_result(id, "x", "ok"));
        }
        msgs
    }

    #[test]
    fn keep_tail_1_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 1).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_2_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 2).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_3_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 3).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_4_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 4).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_5_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 5).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_6_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 6).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_7_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 7).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_8_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 8).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_9_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 9).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_10_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 10).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_11_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 11).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_12_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 12).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_13_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 13).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_14_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 14).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_15_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 15).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_16_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 16).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_17_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 17).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_18_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 18).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_19_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 19).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_20_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 20).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_21_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 21).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_22_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 22).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_23_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 23).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_24_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 24).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_25_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 25).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_26_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 26).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_27_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 27).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_28_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 28).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_29_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 29).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_30_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 30).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_31_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 31).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_32_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 32).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_33_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 33).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_34_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 34).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_35_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 35).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_36_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 36).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_37_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 37).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_38_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 38).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_39_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 39).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_40_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 40).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_41_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 41).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_42_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 42).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_43_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 43).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_44_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 44).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_45_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 45).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_46_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 46).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_47_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 47).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_48_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 48).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_49_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 49).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_50_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 50).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_51_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 51).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_52_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 52).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_53_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 53).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_54_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 54).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_55_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 55).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_56_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 56).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_57_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 57).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_58_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 58).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_59_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 59).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn keep_tail_60_compacts_safely() {
        let msgs = conversation();
        let n = msgs.len();
        let range = select_compaction_range(&msgs, 60).expect("must compact");
        assert!(range.split_idx > 0 && range.split_idx <= n);
        let out = apply_range(msgs, range, None);
        assert!(tool_pair_invariant_holds(&out));
        assert_eq!(out.first().map(|m| m.role), Some(Role::System));
        if let Some(m) = out.get(1) {
            assert_ne!(m.role, Role::Tool);
        }
    }

    #[test]
    fn no_compact_at_or_above_len() {
        let msgs = conversation();
        let n = msgs.len();
        assert!(select_compaction_range(&msgs, n).is_none());
        assert!(select_compaction_range(&msgs, n + 1).is_none());
        assert!(select_compaction_range(&msgs, 0).is_none());
    }
}
