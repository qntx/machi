//! Agent definitions and built instances.
//!
//! An [`Agent`] does not run a `ReAct` loop. [`machi_runtime::TurnRuntime`]
//! (or a product host) executes turns against an agent.

#![forbid(unsafe_code)]

pub mod builder;
pub mod builtin;
pub mod definition;
pub mod discovery;
pub mod instance;
pub mod prompt;
pub mod registry;

pub use builder::AgentBuilder;
pub use builtin::{
    EXPLORE, GENERAL_PURPOSE, ORCHESTRATOR_DELEGATION_PROMPT, PLAN, builtin_definitions,
    builtin_names,
};
pub use definition::{
    AgentDefinition, AgentSource, CompletionRequirement, Instructions, ToolPolicy,
};
pub use discovery::{
    PROJECT_AGENTS_DIR, USER_AGENTS_DIR, by_name, by_name_in_dir, by_name_resolved,
    discover_in_dir, discover_project, discover_user, load_file, parse_definition_markdown,
    project_agent_dirs, resolve_agents, user_agents_dir,
};
pub use instance::Agent;
pub use prompt::{
    IdentityAssembler, PROJECT_AGENTS_MD, ProjectPromptAssembler, PromptAssembler, agents_md_path,
};
pub use registry::AgentRegistry;
