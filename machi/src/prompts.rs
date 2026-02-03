//! Prompt templates for agent system prompts.
//!
//! This module provides configurable prompt templates that define how
//! agents interact with language models.

use serde::{Deserialize, Serialize};

/// Complete prompt templates for an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplates {
    /// Main system prompt.
    pub system_prompt: String,
    /// Planning prompt templates.
    pub planning: PlanningPrompts,
    /// Managed agent prompt templates.
    pub managed_agent: ManagedAgentPrompts,
    /// Final answer prompt templates.
    pub final_answer: FinalAnswerPrompts,
}

/// Planning-related prompt templates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanningPrompts {
    /// Initial planning prompt.
    pub initial_plan: String,
    /// Pre-messages for plan updates.
    pub update_plan_pre_messages: String,
    /// Post-messages for plan updates.
    pub update_plan_post_messages: String,
}

/// Managed agent prompt templates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedAgentPrompts {
    /// Task prompt for managed agents.
    pub task: String,
    /// Report prompt for managed agent results.
    pub report: String,
}

/// Final answer prompt templates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalAnswerPrompts {
    /// Pre-messages before final answer.
    pub pre_messages: String,
    /// Post-messages after final answer.
    pub post_messages: String,
}

impl Default for PromptTemplates {
    fn default() -> Self {
        Self::tool_calling_agent()
    }
}

impl PromptTemplates {
    /// Create default prompts for a tool-calling agent.
    #[must_use]
    pub fn tool_calling_agent() -> Self {
        Self {
            system_prompt: TOOL_CALLING_SYSTEM_PROMPT.to_string(),
            planning: PlanningPrompts::default(),
            managed_agent: ManagedAgentPrompts::default(),
            final_answer: FinalAnswerPrompts::default(),
        }
    }

    /// Create default prompts for a code agent.
    #[must_use]
    pub fn code_agent() -> Self {
        Self {
            system_prompt: CODE_AGENT_SYSTEM_PROMPT.to_string(),
            planning: PlanningPrompts::default(),
            managed_agent: ManagedAgentPrompts::default(),
            final_answer: FinalAnswerPrompts::default(),
        }
    }
}

impl Default for PlanningPrompts {
    fn default() -> Self {
        Self {
            initial_plan: INITIAL_PLAN_PROMPT.to_string(),
            update_plan_pre_messages: UPDATE_PLAN_PRE_PROMPT.to_string(),
            update_plan_post_messages: UPDATE_PLAN_POST_PROMPT.to_string(),
        }
    }
}

impl Default for ManagedAgentPrompts {
    fn default() -> Self {
        Self {
            task: MANAGED_AGENT_TASK_PROMPT.to_string(),
            report: MANAGED_AGENT_REPORT_PROMPT.to_string(),
        }
    }
}

impl Default for FinalAnswerPrompts {
    fn default() -> Self {
        Self {
            pre_messages: FINAL_ANSWER_PRE_PROMPT.to_string(),
            post_messages: FINAL_ANSWER_POST_PROMPT.to_string(),
        }
    }
}

/// Default system prompt for tool-calling agents.
pub const TOOL_CALLING_SYSTEM_PROMPT: &str = r#"You are a helpful AI assistant that can use tools to accomplish tasks.

You have access to the following tools:
{{tools}}

When you need to use a tool, respond with a tool call in the appropriate format.
When you have the final answer, use the 'final_answer' tool to provide it.

Think step by step about what you need to do to accomplish the task.

{{#if custom_instructions}}
{{custom_instructions}}
{{/if}}"#;

/// Default system prompt for code agents.
pub const CODE_AGENT_SYSTEM_PROMPT: &str = r#"You are an expert assistant who can solve any task using code blobs.
You will be given a task to solve as best you can.
To do so, you have been given access to a list of tools: these tools are basically functions which you can call with code.

At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use.
Then in the Code sequence you should write the code. The code sequence must be opened with '```code', and closed with '```'.
During each intermediate step, you can use 'print()' to save whatever important information you will then need.
These print outputs will then appear in the 'Observation:' field, which will be available as input for the next step.
In the end you have to return a final answer using the `final_answer` tool.

Here are the available tools:
```
{{tools}}
```

Here are the rules you should always follow:
1. Always provide a 'Thought:' sequence, and a '```code' sequence ending with '```', else you will fail.
2. Use only variables that you have defined!
3. Always use the right arguments for the tools.
4. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
5. Don't name any new variable with the same name as a tool.
6. Don't give up! You're in charge of solving the task, not providing directions to solve it.

{{#if custom_instructions}}
{{custom_instructions}}
{{/if}}

Now Begin!"#;

/// Initial planning prompt.
pub const INITIAL_PLAN_PROMPT: &str = r#"You are a world expert at analyzing a situation to derive facts, and plan accordingly towards solving a task.
Below I will present you a task. You will need to 1. build a survey of facts known or needed to solve the task, then 2. make a plan of action to solve the task.

## 1. Facts survey
You will build a comprehensive preparatory survey of which facts we have at our disposal and which ones we still need.

### 1.1. Facts given in the task
List here the specific facts given in the task that could help you.

### 1.2. Facts to look up
List here any facts that we may need to look up.

### 1.3. Facts to derive
List here anything that we want to derive from the above by logical reasoning.

## 2. Plan
Then for the given task, develop a step-by-step high-level plan taking into account the above inputs and list of facts.
This plan should involve individual tasks based on the available tools, that if executed correctly will yield the correct answer.
Do not skip steps, do not add any superfluous steps. Only write the high-level plan, DO NOT DETAIL INDIVIDUAL TOOL CALLS.
After writing the final step of the plan, write the '<end_plan>' tag and stop there.

---
Now begin! Here is your task:
```
{{task}}
```
First in part 1, write the facts survey, then in part 2, write your plan."#;

/// Update plan pre-messages prompt.
pub const UPDATE_PLAN_PRE_PROMPT: &str = r#"You are a world expert at analyzing a situation, and plan accordingly towards solving a task.
You have been given the following task:
```
{{task}}
```

Below you will find a history of attempts made to solve this task.
You will first have to produce a survey of known and unknown facts, then propose a step-by-step high-level plan to solve the task.
If the previous tries so far have met some success, your updated plan can build on these results.
If you are stalled, you can make a completely new plan starting from scratch.

Find the task and history below:"#;

/// Update plan post-messages prompt.
pub const UPDATE_PLAN_POST_PROMPT: &str = r#"Now write your updated facts below, taking into account the above history:
## 1. Updated facts survey
### 1.1. Facts given in the task
### 1.2. Facts that we have learned
### 1.3. Facts still to look up
### 1.4. Facts still to derive

Then write a step-by-step high-level plan to solve the task above.
## 2. Plan
### 2.1. ...
Etc.

This plan should involve individual tasks based on the available tools, that if executed correctly will yield the correct answer.
Beware that you have {{remaining_steps}} steps remaining.
Do not skip steps, do not add any superfluous steps. Only write the high-level plan, DO NOT DETAIL INDIVIDUAL TOOL CALLS.
After writing the final step of the plan, write the '<end_plan>' tag and stop there.

Now write your updated facts survey below, then your new plan."#;

/// Managed agent task prompt.
pub const MANAGED_AGENT_TASK_PROMPT: &str = r#"You're a helpful agent named '{{name}}'.
You have been submitted this task by your manager.
---
Task:
{{task}}
---
You're helping your manager solve a wider task: so make sure to not provide a one-line answer, but give as much information as possible to give them a clear understanding of the answer.

Your final_answer WILL HAVE to contain these parts:
### 1. Task outcome (short version):
### 2. Task outcome (extremely detailed version):
### 3. Additional context (if relevant):

Put all these in your final_answer tool, everything that you do not pass as an argument to final_answer will be lost.
And even if your task resolution is not successful, please return as much context as possible, so that your manager can act upon this feedback."#;

/// Managed agent report prompt.
pub const MANAGED_AGENT_REPORT_PROMPT: &str = r#"Here is the final answer from your managed agent '{{name}}':
{{final_answer}}"#;

/// Final answer pre-messages prompt.
pub const FINAL_ANSWER_PRE_PROMPT: &str = r#"An agent tried to answer a user query but it got stuck and failed to do so. You are tasked with providing an answer instead. Here is the agent's memory:"#;

/// Final answer post-messages prompt.
pub const FINAL_ANSWER_POST_PROMPT: &str = r#"Based on the above, please provide an answer to the following user task:
{{task}}"#;

/// Render a template with variables.
pub fn render_template(
    template: &str,
    variables: &std::collections::HashMap<String, String>,
) -> String {
    let mut result = template.to_string();
    for (key, value) in variables {
        let placeholder = format!("{{{{{}}}}}", key);
        result = result.replace(&placeholder, value);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_template() {
        let mut vars = std::collections::HashMap::new();
        vars.insert("name".to_string(), "TestAgent".to_string());
        vars.insert("task".to_string(), "Do something".to_string());

        let template = "Agent {{name}} will {{task}}";
        let result = render_template(template, &vars);
        assert_eq!(result, "Agent TestAgent will Do something");
    }

    #[test]
    fn test_default_prompts() {
        let prompts = PromptTemplates::default();
        assert!(!prompts.system_prompt.is_empty());
        assert!(!prompts.planning.initial_plan.is_empty());
    }
}
