//! Simple arithmetic evaluation tool for demos and tests.

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::context::ToolCallContext;
use crate::error::{ToolError, codes};
use crate::metadata::ToolMetadata;
use crate::tool::{DynTool, ToolResult};

/// Evaluates a restricted arithmetic expression (`+ - * / ( )` and numbers).
#[derive(Debug, Default, Clone, Copy)]
pub struct CalcTool;

#[async_trait]
impl DynTool for CalcTool {
    fn name(&self) -> &'static str {
        "calc"
    }

    fn description(&self) -> &'static str {
        "Evaluate a basic arithmetic expression with + - * / and parentheses. \
         Example: expr=\"(2+3)*4\""
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "expr": {
                    "type": "string",
                    "description": "Arithmetic expression to evaluate"
                }
            },
            "required": ["expr"],
            "additionalProperties": false
        })
    }

    fn metadata(&self) -> ToolMetadata {
        ToolMetadata::read_only()
    }

    async fn call(&self, _ctx: ToolCallContext, arguments: Value) -> Result<ToolResult, ToolError> {
        let expr = arguments
            .get("expr")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| codes::invalid_args("calc requires non-empty expr"))?;
        let value = eval_expr(expr).map_err(codes::execution)?;
        Ok(ToolResult {
            content: value.to_string(),
            structured: Some(json!({ "expr": expr, "value": value })),
            is_error: false,
        })
    }
}

/// Recursive-descent evaluator for numbers and + - * / ( ).
fn eval_expr(input: &str) -> Result<f64, String> {
    let tokens = tokenize(input)?;
    let mut p = Parser { tokens, i: 0 };
    let v = p.parse_expr()?;
    if p.i != p.tokens.len() {
        return Err("unexpected trailing tokens".into());
    }
    Ok(v)
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Op(char),
    LParen,
    RParen,
}

fn tokenize(s: &str) -> Result<Vec<Tok>, String> {
    let mut out = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        let Some(&c) = chars.get(i) else {
            break;
        };
        if c.is_whitespace() {
            i = i.saturating_add(1);
            continue;
        }
        if c.is_ascii_digit() || c == '.' {
            let start = i;
            i = i.saturating_add(1);
            while i < chars.len()
                && chars
                    .get(i)
                    .is_some_and(|ch| ch.is_ascii_digit() || *ch == '.')
            {
                i = i.saturating_add(1);
            }
            let slice: String = chars
                .get(start..i)
                .ok_or_else(|| "bad number slice".to_owned())?
                .iter()
                .collect();
            let n: f64 = slice
                .parse()
                .map_err(|_| format!("invalid number: {slice}"))?;
            out.push(Tok::Num(n));
            continue;
        }
        match c {
            '+' | '-' | '*' | '/' => {
                out.push(Tok::Op(c));
                i = i.saturating_add(1);
            }
            '(' => {
                out.push(Tok::LParen);
                i = i.saturating_add(1);
            }
            ')' => {
                out.push(Tok::RParen);
                i = i.saturating_add(1);
            }
            other => return Err(format!("invalid character: {other}")),
        }
    }
    Ok(out)
}

struct Parser {
    tokens: Vec<Tok>,
    i: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.i)
    }

    fn bump(&mut self) -> Option<Tok> {
        let t = self.tokens.get(self.i).cloned();
        if t.is_some() {
            self.i = self.i.saturating_add(1);
        }
        t
    }

    fn parse_expr(&mut self) -> Result<f64, String> {
        let mut v = self.parse_term()?;
        loop {
            match self.peek() {
                Some(Tok::Op('+')) => {
                    self.bump();
                    v += self.parse_term()?;
                }
                Some(Tok::Op('-')) => {
                    self.bump();
                    v -= self.parse_term()?;
                }
                _ => break,
            }
        }
        Ok(v)
    }

    fn parse_term(&mut self) -> Result<f64, String> {
        let mut v = self.parse_factor()?;
        loop {
            match self.peek() {
                Some(Tok::Op('*')) => {
                    self.bump();
                    v *= self.parse_factor()?;
                }
                Some(Tok::Op('/')) => {
                    self.bump();
                    let d = self.parse_factor()?;
                    if d == 0.0 {
                        return Err("division by zero".into());
                    }
                    v /= d;
                }
                _ => break,
            }
        }
        Ok(v)
    }

    fn parse_factor(&mut self) -> Result<f64, String> {
        match self.bump() {
            Some(Tok::Num(n)) => Ok(n),
            Some(Tok::Op('-')) => Ok(-self.parse_factor()?),
            Some(Tok::Op('+')) => self.parse_factor(),
            Some(Tok::LParen) => {
                let v = self.parse_expr()?;
                match self.bump() {
                    Some(Tok::RParen) => Ok(v),
                    _ => Err("expected ')'".into()),
                }
            }
            other => Err(format!("unexpected token: {other:?}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn evaluates_expression() {
        let tool = CalcTool;
        let result = tool
            .call(ToolCallContext::default(), json!({"expr": "(2+3)*4"}))
            .await
            .expect("calc");
        assert_eq!(result.content, "20");
        assert_eq!(
            result.structured.as_ref().and_then(|v| v.get("value")),
            Some(&json!(20.0))
        );
    }

    #[test]
    fn rejects_letters() {
        assert!(eval_expr("1+foo").is_err());
    }
}
