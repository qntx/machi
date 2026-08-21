//! Drive real [`OpenAiCompatSampler::sample`] against a local HTTP fake.
#![cfg(feature = "openai")]
#![allow(
    unused_crate_dependencies,
    reason = "integration binary links provider deps"
)]

#[cfg(test)]
mod http_fake {
    #![allow(
        clippy::expect_used,
        clippy::unwrap_used,
        clippy::print_stdout,
        reason = "integration test harness"
    )]

    use ovo_llm::{LlmSampler, OpenAiCompatConfig, OpenAiCompatSampler, SampleRequest, ToolChoice};
    use ovo_types::Message;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio_util::sync::CancellationToken;

    async fn serve_one_json_response(listener: TcpListener, body: String) {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let mut buf = vec![0u8; 8192];
        let _n = socket.read(&mut buf).await.expect("read request");
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("write response");
    }

    #[tokio::test]
    async fn openai_compat_sample_parses_tool_calls_from_fake_http() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let fixture = serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "calc",
                            "arguments": "{\"expr\":\"2+3\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 11, "completion_tokens": 7 }
        });
        let body = fixture.to_string();
        let server = tokio::spawn(serve_one_json_response(listener, body));

        let config = OpenAiCompatConfig::new(format!("http://{addr}"), "test-key");
        let sampler = OpenAiCompatSampler::new(config).expect("client");
        let response = sampler
            .sample(SampleRequest {
                model: "gpt-test".into(),
                messages: vec![Message::user("compute 2+3")],
                tools: vec![],
                tool_choice: ToolChoice::Auto,
                response_format: None,
                max_output_tokens: None,
                temperature: None,
                cancel: CancellationToken::new(),
                deadline: None,
            })
            .await
            .expect("sample");

        server.await.expect("server join");

        assert_eq!(response.message.tool_calls.len(), 1);
        let tc = response.message.tool_calls.first().expect("tc");
        assert_eq!(tc.id.as_str(), "call_abc");
        assert_eq!(tc.name, "calc");
        assert_eq!(
            tc.arguments.get("expr").and_then(|v| v.as_str()),
            Some("2+3")
        );
        assert_eq!(response.usage.input_tokens, 11);
        assert_eq!(response.usage.output_tokens, 7);
        assert_eq!(response.stop_reason.as_deref(), Some("tool_calls"));
    }

    #[tokio::test]
    async fn openai_compat_sample_parses_assistant_text_from_fake_http() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let fixture = serde_json::json!({
            "choices": [{
                "message": { "role": "assistant", "content": "wire-ok-42" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 2, "completion_tokens": 4 }
        });
        let server = tokio::spawn(serve_one_json_response(listener, fixture.to_string()));

        let sampler =
            OpenAiCompatSampler::new(OpenAiCompatConfig::new(format!("http://{addr}"), ""))
                .expect("client");
        let response = sampler
            .sample(SampleRequest {
                model: "m".into(),
                messages: vec![Message::user("hi")],
                tools: vec![],
                tool_choice: ToolChoice::Auto,
                response_format: None,
                max_output_tokens: None,
                temperature: None,
                cancel: CancellationToken::new(),
                deadline: None,
            })
            .await
            .expect("sample");
        server.await.expect("join");
        assert_eq!(response.message.text(), "wire-ok-42");
        assert_eq!(response.usage.output_tokens, 4);
    }
}
