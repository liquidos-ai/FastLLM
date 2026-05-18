use crate::{
    AutoagentsProviderConfig, LlmGatewayError, LlmProvider, LlmRequest, LlmResponse,
    LlmStreamEvent, ModelInfo, ModelLoadRequest, build_autoagents_provider,
};
use autoagents_llm::ToolCall;
use autoagents_llm::chat::{
    ChatMessage, ChatRole, FunctionTool, MessageType, StructuredOutputFormat, Tool,
};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Default, Clone)]
pub struct LlmGateway {
    providers: BTreeMap<String, Arc<dyn LlmProvider>>,
}

impl LlmGateway {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_provider(
        mut self,
        name: impl Into<String>,
        provider: Arc<dyn LlmProvider>,
    ) -> Self {
        self.register_provider(name, provider);
        self
    }

    pub fn register_provider(&mut self, name: impl Into<String>, provider: Arc<dyn LlmProvider>) {
        self.providers.insert(name.into(), provider);
    }

    pub fn register_autoagents_provider(
        &mut self,
        config: AutoagentsProviderConfig,
    ) -> Result<(), LlmGatewayError> {
        let name = config.provider.clone();
        let provider = build_autoagents_provider(config)?;
        self.register_provider(name, provider);
        Ok(())
    }

    pub fn providers(&self) -> impl Iterator<Item = &str> {
        self.providers.keys().map(String::as_str)
    }

    pub async fn load_model(
        &self,
        request: ModelLoadRequest,
    ) -> Result<ModelInfo, LlmGatewayError> {
        self.provider(&request.route.provider)?;
        Ok(ModelInfo {
            route: request.route,
            loaded: true,
        })
    }

    pub async fn chat(&self, request: LlmRequest) -> Result<LlmResponse, LlmGatewayError> {
        let provider = self.provider(&request.route.provider)?;
        let provider_name = request.route.provider.clone();
        let messages = request
            .messages
            .iter()
            .map(to_autoagents_message)
            .collect::<Vec<_>>();
        let tools = request
            .tools
            .iter()
            .map(to_autoagents_tool)
            .collect::<Vec<_>>();
        let schema = request.output_schema.map(|schema| StructuredOutputFormat {
            name: "odyssey_output".to_string(),
            description: None,
            schema: Some(schema),
            strict: Some(true),
        });
        let response = provider
            .chat_with_tools(
                &messages,
                (!tools.is_empty()).then_some(tools.as_slice()),
                schema,
            )
            .await
            .map_err(|source| LlmGatewayError::Provider {
                provider: provider_name,
                message: source.to_string(),
            })?;
        Ok(LlmResponse {
            text: response.text().unwrap_or_default(),
            reasoning: response.thinking(),
            tool_calls: response
                .tool_calls()
                .unwrap_or_default()
                .into_iter()
                .map(from_autoagents_tool_call)
                .collect(),
            usage: response.usage().map(|usage| crate::TokenUsage {
                input_tokens: usage.prompt_tokens as u64,
                output_tokens: usage.completion_tokens as u64,
            }),
        })
    }

    pub async fn chat_stream(
        &self,
        request: LlmRequest,
    ) -> Result<Vec<LlmStreamEvent>, LlmGatewayError> {
        let response = self.chat(request).await?;
        let mut events = Vec::new();
        if !response.text.is_empty() {
            events.push(LlmStreamEvent::TextDelta {
                text: response.text.clone(),
            });
        }
        if let Some(reasoning) = &response.reasoning
            && !reasoning.is_empty()
        {
            events.push(LlmStreamEvent::ReasoningDelta {
                text: reasoning.clone(),
            });
        }
        events.extend(
            response
                .tool_calls
                .iter()
                .cloned()
                .map(|call| LlmStreamEvent::ToolCall { call }),
        );
        events.push(LlmStreamEvent::Done { response });
        Ok(events)
    }

    fn provider(&self, name: &str) -> Result<Arc<dyn LlmProvider>, LlmGatewayError> {
        self.providers
            .get(name)
            .cloned()
            .ok_or_else(|| LlmGatewayError::UnknownProvider(name.to_string()))
    }

    pub fn provider_handle(&self, name: &str) -> Result<Arc<dyn LlmProvider>, LlmGatewayError> {
        self.provider(name)
    }
}

fn to_autoagents_message(message: &crate::LlmMessage) -> ChatMessage {
    let role = match message.role.as_str() {
        "system" => ChatRole::System,
        "assistant" => ChatRole::Assistant,
        "tool" => ChatRole::Tool,
        _ => ChatRole::User,
    };
    ChatMessage {
        role,
        message_type: MessageType::Text,
        content: message.content.clone(),
    }
}

fn to_autoagents_tool(tool: &crate::LlmTool) -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: FunctionTool {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.input_schema.clone(),
        },
    }
}

fn from_autoagents_tool_call(call: ToolCall) -> crate::LlmToolCall {
    crate::LlmToolCall {
        name: call.function.name,
        arguments: serde_json::from_str(&call.function.arguments)
            .unwrap_or(serde_json::Value::Null),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EchoProvider, LlmMessage, ModelRoute};

    #[tokio::test]
    async fn gateway_routes_to_registered_provider() {
        let gateway = LlmGateway::new().with_provider("echo", Arc::new(EchoProvider));
        let response = gateway
            .chat(LlmRequest {
                route: ModelRoute::new("echo", "test"),
                messages: vec![LlmMessage::user("hello")],
                tools: Vec::new(),
                output_schema: None,
                parameters: BTreeMap::new(),
            })
            .await
            .expect("response");

        assert_eq!(response.text, "hello");
    }

    #[tokio::test]
    async fn gateway_denies_unknown_provider() {
        let err = LlmGateway::new()
            .chat(LlmRequest {
                route: ModelRoute::new("missing", "test"),
                messages: Vec::new(),
                tools: Vec::new(),
                output_schema: None,
                parameters: BTreeMap::new(),
            })
            .await
            .expect_err("unknown provider");

        assert!(err.to_string().contains("missing"));
    }

    #[tokio::test]
    async fn gateway_streams_full_response_as_events() {
        let gateway = LlmGateway::new().with_provider("echo", Arc::new(EchoProvider));
        let events = gateway
            .chat_stream(LlmRequest {
                route: ModelRoute::new("echo", "test"),
                messages: vec![LlmMessage::user("hello")],
                tools: Vec::new(),
                output_schema: None,
                parameters: BTreeMap::new(),
            })
            .await
            .expect("events");

        assert!(matches!(
            &events[0],
            LlmStreamEvent::TextDelta { text } if text == "hello"
        ));
        assert!(matches!(events.last(), Some(LlmStreamEvent::Done { .. })));
    }
}
