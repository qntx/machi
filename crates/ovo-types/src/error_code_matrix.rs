// ErrorCode contract matrix.
#[cfg(test)]
#[allow(non_snake_case, clippy::missing_assert_message, reason = "matrix names embed ErrorCode variants")]
mod error_code_matrix {
    use super::ErrorCode;

    #[test]
    fn code_TypesInvalidId_as_str_nonempty_and_domain() {
        let c = ErrorCode::TypesInvalidId;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_TypesValidation_as_str_nonempty_and_domain() {
        let c = ErrorCode::TypesValidation;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_TypesSerde_as_str_nonempty_and_domain() {
        let c = ErrorCode::TypesSerde;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_ToolNotFound_as_str_nonempty_and_domain() {
        let c = ErrorCode::ToolNotFound;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_ToolInvalidArgs_as_str_nonempty_and_domain() {
        let c = ErrorCode::ToolInvalidArgs;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_ToolExecution_as_str_nonempty_and_domain() {
        let c = ErrorCode::ToolExecution;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_ToolTimeout_as_str_nonempty_and_domain() {
        let c = ErrorCode::ToolTimeout;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_ToolCancelled_as_str_nonempty_and_domain() {
        let c = ErrorCode::ToolCancelled;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_ToolDenied_as_str_nonempty_and_domain() {
        let c = ErrorCode::ToolDenied;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_ToolApprovalDenied_as_str_nonempty_and_domain() {
        let c = ErrorCode::ToolApprovalDenied;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_ToolStreamProtocol_as_str_nonempty_and_domain() {
        let c = ErrorCode::ToolStreamProtocol;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_ToolRateLimited_as_str_nonempty_and_domain() {
        let c = ErrorCode::ToolRateLimited;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_ToolConcurrencyLimit_as_str_nonempty_and_domain() {
        let c = ErrorCode::ToolConcurrencyLimit;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_ToolNetwork_as_str_nonempty_and_domain() {
        let c = ErrorCode::ToolNetwork;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_ToolServiceUnavailable_as_str_nonempty_and_domain() {
        let c = ErrorCode::ToolServiceUnavailable;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_LlmProvider_as_str_nonempty_and_domain() {
        let c = ErrorCode::LlmProvider;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_LlmCancelled_as_str_nonempty_and_domain() {
        let c = ErrorCode::LlmCancelled;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_LlmInvalidResponse_as_str_nonempty_and_domain() {
        let c = ErrorCode::LlmInvalidResponse;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_LlmAuth_as_str_nonempty_and_domain() {
        let c = ErrorCode::LlmAuth;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_LlmRateLimit_as_str_nonempty_and_domain() {
        let c = ErrorCode::LlmRateLimit;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_LlmIdleTimeout_as_str_nonempty_and_domain() {
        let c = ErrorCode::LlmIdleTimeout;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_LlmEmptyResponse_as_str_nonempty_and_domain() {
        let c = ErrorCode::LlmEmptyResponse;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_LlmTruncated_as_str_nonempty_and_domain() {
        let c = ErrorCode::LlmTruncated;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_AgentInvalidDefinition_as_str_nonempty_and_domain() {
        let c = ErrorCode::AgentInvalidDefinition;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_AgentBuild_as_str_nonempty_and_domain() {
        let c = ErrorCode::AgentBuild;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_AgentNotFound_as_str_nonempty_and_domain() {
        let c = ErrorCode::AgentNotFound;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_RuntimeMaxSteps_as_str_nonempty_and_domain() {
        let c = ErrorCode::RuntimeMaxSteps;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_RuntimeCancelled_as_str_nonempty_and_domain() {
        let c = ErrorCode::RuntimeCancelled;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_RuntimeGate_as_str_nonempty_and_domain() {
        let c = ErrorCode::RuntimeGate;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_RuntimeStructuredOutput_as_str_nonempty_and_domain() {
        let c = ErrorCode::RuntimeStructuredOutput;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_RuntimeDeadline_as_str_nonempty_and_domain() {
        let c = ErrorCode::RuntimeDeadline;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_RuntimeStationarity_as_str_nonempty_and_domain() {
        let c = ErrorCode::RuntimeStationarity;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_HostSpawn_as_str_nonempty_and_domain() {
        let c = ErrorCode::HostSpawn;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_HostBudget_as_str_nonempty_and_domain() {
        let c = ErrorCode::HostBudget;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_HostDepth_as_str_nonempty_and_domain() {
        let c = ErrorCode::HostDepth;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_HostConcurrency_as_str_nonempty_and_domain() {
        let c = ErrorCode::HostConcurrency;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_HostUnsupported_as_str_nonempty_and_domain() {
        let c = ErrorCode::HostUnsupported;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_HostCancelled_as_str_nonempty_and_domain() {
        let c = ErrorCode::HostCancelled;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_HostIsolation_as_str_nonempty_and_domain() {
        let c = ErrorCode::HostIsolation;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_WorkflowScript_as_str_nonempty_and_domain() {
        let c = ErrorCode::WorkflowScript;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_WorkflowDivergence_as_str_nonempty_and_domain() {
        let c = ErrorCode::WorkflowDivergence;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_WorkflowJournal_as_str_nonempty_and_domain() {
        let c = ErrorCode::WorkflowJournal;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_WorkflowBudget_as_str_nonempty_and_domain() {
        let c = ErrorCode::WorkflowBudget;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_WorkflowCancelled_as_str_nonempty_and_domain() {
        let c = ErrorCode::WorkflowCancelled;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_WorkflowValidate_as_str_nonempty_and_domain() {
        let c = ErrorCode::WorkflowValidate;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_StateInvariant_as_str_nonempty_and_domain() {
        let c = ErrorCode::StateInvariant;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_StatePersistence_as_str_nonempty_and_domain() {
        let c = ErrorCode::StatePersistence;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_CompactionFailed_as_str_nonempty_and_domain() {
        let c = ErrorCode::CompactionFailed;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_CompactionOverflow_as_str_nonempty_and_domain() {
        let c = ErrorCode::CompactionOverflow;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn code_Internal_as_str_nonempty_and_domain() {
        let c = ErrorCode::Internal;
        let s = c.as_str();
        assert!(!s.is_empty());
        assert!(!s.contains(' '));
        let d = c.domain();
        assert!(!d.is_empty());
        if c != ErrorCode::Internal {
            assert!(
                s == d || s.starts_with(&format!("{d}.")),
                "{s} vs domain {d}"
            );
        }
        let _ = c.default_retry();
    }

    #[test]
    fn all_as_str_unique() {
        let all = [
            ErrorCode::TypesInvalidId,
            ErrorCode::TypesValidation,
            ErrorCode::TypesSerde,
            ErrorCode::ToolNotFound,
            ErrorCode::ToolInvalidArgs,
            ErrorCode::ToolExecution,
            ErrorCode::ToolTimeout,
            ErrorCode::ToolCancelled,
            ErrorCode::ToolDenied,
            ErrorCode::ToolApprovalDenied,
            ErrorCode::ToolStreamProtocol,
            ErrorCode::ToolRateLimited,
            ErrorCode::ToolConcurrencyLimit,
            ErrorCode::ToolNetwork,
            ErrorCode::ToolServiceUnavailable,
            ErrorCode::LlmProvider,
            ErrorCode::LlmCancelled,
            ErrorCode::LlmInvalidResponse,
            ErrorCode::LlmAuth,
            ErrorCode::LlmRateLimit,
            ErrorCode::LlmIdleTimeout,
            ErrorCode::LlmEmptyResponse,
            ErrorCode::LlmTruncated,
            ErrorCode::AgentInvalidDefinition,
            ErrorCode::AgentBuild,
            ErrorCode::AgentNotFound,
            ErrorCode::RuntimeMaxSteps,
            ErrorCode::RuntimeCancelled,
            ErrorCode::RuntimeGate,
            ErrorCode::RuntimeStructuredOutput,
            ErrorCode::RuntimeDeadline,
            ErrorCode::RuntimeStationarity,
            ErrorCode::HostSpawn,
            ErrorCode::HostBudget,
            ErrorCode::HostDepth,
            ErrorCode::HostConcurrency,
            ErrorCode::HostUnsupported,
            ErrorCode::HostCancelled,
            ErrorCode::HostIsolation,
            ErrorCode::WorkflowScript,
            ErrorCode::WorkflowDivergence,
            ErrorCode::WorkflowJournal,
            ErrorCode::WorkflowBudget,
            ErrorCode::WorkflowCancelled,
            ErrorCode::WorkflowValidate,
            ErrorCode::StateInvariant,
            ErrorCode::StatePersistence,
            ErrorCode::CompactionFailed,
            ErrorCode::CompactionOverflow,
            ErrorCode::Internal
        ];
        let mut set = std::collections::BTreeSet::new();
        for c in all {
            assert!(set.insert(c.as_str()), "dup {}", c.as_str());
        }
        assert_eq!(set.len(), all.len());
    }
}
