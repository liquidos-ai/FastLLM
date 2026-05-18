use crate::{
    CacheConfig, ExecutionScheduler, GatewayConfig, LlmGatewayError, LlmProvider, LlmRequest,
    LlmResponse, LlmStreamEvent, ModelConfig, ModelInfo, ModelLoadReason, ModelLoadRequest,
    ModelRoute, ModelState, PromptCache, ProviderConfig, RetryPipeline, Telemetry, build_provider,
    local::ModelRegistry, runtime::InferenceRuntime,
};
use autoagents_llm::ToolCall;
use autoagents_llm::chat::{
    ChatMessage, ChatRole, FunctionTool, MessageType, StructuredOutputFormat, Tool,
};
use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone)]
pub struct LlmGateway {
    inner: Arc<GatewayInner>,
}

struct GatewayInner {
    config: GatewayConfig,
    providers: RwLock<BTreeMap<String, Arc<dyn LlmProvider>>>,
    runtimes: RwLock<BTreeMap<String, Arc<dyn InferenceRuntime>>>,
    cache: PromptCache,
    scheduler: ExecutionScheduler,
    retry: RetryPipeline,
    registry: ModelRegistry,
    telemetry: Telemetry,
}

pub struct LlmGatewayBuilder {
    config: GatewayConfig,
}

impl LlmGatewayBuilder {
    pub fn new() -> Self {
        Self {
            config: GatewayConfig::default(),
        }
    }

    pub fn config(mut self, config: GatewayConfig) -> Self {
        self.config = config;
        self
    }

    pub fn cache(mut self, cache: CacheConfig) -> Self {
        self.config.cache = cache;
        self
    }

    pub fn build(self) -> LlmGateway {
        LlmGateway::with_config(self.config)
    }
}

impl Default for LlmGatewayBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for LlmGateway {
    fn default() -> Self {
        Self::new()
    }
}

impl LlmGateway {
    pub fn new() -> Self {
        Self::with_config(GatewayConfig::default())
    }

    pub fn builder() -> LlmGatewayBuilder {
        LlmGatewayBuilder::new()
    }

    pub fn with_config(config: GatewayConfig) -> Self {
        let telemetry = Telemetry::new();
        let scheduler = ExecutionScheduler::new(config.scheduler.clone(), telemetry.clone());
        let retry = RetryPipeline::new(config.retry.clone(), telemetry.clone());
        let registry = ModelRegistry::new(config.local_memory_budget_bytes, telemetry.clone());
        for model in config.models.values().cloned() {
            registry.register(model);
        }
        Self {
            inner: Arc::new(GatewayInner {
                cache: PromptCache::new(config.cache.clone()),
                config,
                providers: RwLock::new(BTreeMap::new()),
                runtimes: RwLock::new(BTreeMap::new()),
                scheduler,
                retry,
                registry,
                telemetry,
            }),
        }
    }

    pub fn with_provider(self, name: impl Into<String>, provider: Arc<dyn LlmProvider>) -> Self {
        self.register_provider(name, provider);
        self
    }

    pub fn register_provider(&self, name: impl Into<String>, provider: Arc<dyn LlmProvider>) {
        self.inner
            .providers
            .write()
            .expect("provider registry poisoned")
            .insert(name.into(), provider);
    }

    pub fn register_runtime(&self, route: ModelRoute, runtime: Arc<dyn InferenceRuntime>) {
        self.inner
            .runtimes
            .write()
            .expect("runtime registry poisoned")
            .insert(route.key(), runtime);
    }

    pub fn register_model(&self, config: ModelConfig) {
        self.inner.registry.register(config);
    }

    #[cfg(feature = "local")]
    pub fn register_llamacpp_model(&self, config: ModelConfig) {
        let route = config.route.clone();
        self.register_model(config.clone());
        self.register_runtime(
            route,
            Arc::new(crate::runtime::llamacpp::LlamaCppRuntime::new(config)),
        );
    }

    pub fn register_provider_config(&self, config: ProviderConfig) -> Result<(), LlmGatewayError> {
        let name = config.provider.clone();
        let provider = build_provider(config)?;
        self.register_provider(name, provider);
        Ok(())
    }

    pub fn providers(&self) -> Vec<String> {
        self.inner
            .providers
            .read()
            .expect("provider registry poisoned")
            .keys()
            .cloned()
            .collect()
    }

    pub fn config(&self) -> &GatewayConfig {
        &self.inner.config
    }

    pub fn telemetry(&self) -> Telemetry {
        self.inner.telemetry.clone()
    }

    pub async fn load_model(
        &self,
        request: ModelLoadRequest,
    ) -> Result<ModelInfo, LlmGatewayError> {
        if let Some(runtime) = self.runtime(&request.route) {
            let info = runtime.load_model(request.clone()).await?;
            self.inner.telemetry.record_model_load(&request.route);
            return Ok(info);
        }
        if self.inner.config.model(&request.route).is_some()
            || self.inner.registry.info(&request.route).is_some()
        {
            return self
                .inner
                .registry
                .load(&request.route, now_ms(), ModelLoadReason::Explicit);
        }
        self.provider(&request.route.provider)?;
        Ok(ModelInfo {
            route: request.route,
            loaded: true,
            state: ModelState::Loaded,
            memory_bytes: 0,
            kv_cache_bytes: 0,
            device: None,
            expires_at_ms: None,
            load_reason: ModelLoadReason::Explicit,
        })
    }

    pub async fn chat(&self, request: LlmRequest) -> Result<LlmResponse, LlmGatewayError> {
        let inner = Arc::clone(&self.inner);
        self.inner
            .retry
            .execute(request, move |request| {
                let inner = Arc::clone(&inner);
                async move { inner.execute_cached(request).await }
            })
            .await
    }

    pub async fn chat_stream(
        &self,
        request: LlmRequest,
    ) -> Result<Vec<LlmStreamEvent>, LlmGatewayError> {
        let response = self.chat(request).await?;
        Ok(response_to_events(response))
    }

    fn provider(&self, name: &str) -> Result<Arc<dyn LlmProvider>, LlmGatewayError> {
        self.inner.provider(name)
    }

    fn runtime(&self, route: &ModelRoute) -> Option<Arc<dyn InferenceRuntime>> {
        self.inner.runtime(route)
    }

    pub fn provider_handle(&self, name: &str) -> Result<Arc<dyn LlmProvider>, LlmGatewayError> {
        self.provider(name)
    }
}

impl GatewayInner {
    async fn execute_cached(
        self: Arc<Self>,
        request: LlmRequest,
    ) -> Result<LlmResponse, LlmGatewayError> {
        if let Some(response) = self.cache.get(&request) {
            self.telemetry.record_cache_hit(&request.route);
            return Ok(response);
        }
        self.telemetry.record_cache_miss(&request.route);
        let scheduled = Arc::clone(&self);
        let response = self
            .scheduler
            .execute(request.clone(), move |request| {
                let scheduled = Arc::clone(&scheduled);
                async move { scheduled.execute_uncached(request).await }
            })
            .await?;
        self.cache.insert(&request, response.clone());
        Ok(response)
    }

    async fn execute_uncached(
        self: Arc<Self>,
        request: LlmRequest,
    ) -> Result<LlmResponse, LlmGatewayError> {
        if let Some(runtime) = self.runtime(&request.route) {
            return runtime.chat(request).await;
        }
        if self.config.model(&request.route).is_some()
            || self.registry.info(&request.route).is_some()
        {
            self.registry
                .load(&request.route, now_ms(), ModelLoadReason::OnDemand)?;
        }
        let provider = self.provider(&request.route.provider)?;
        chat_with_provider(provider, request.route.provider.clone(), request).await
    }

    fn provider(&self, name: &str) -> Result<Arc<dyn LlmProvider>, LlmGatewayError> {
        self.providers
            .read()
            .expect("provider registry poisoned")
            .get(name)
            .cloned()
            .ok_or_else(|| LlmGatewayError::UnknownProvider(name.to_string()))
    }

    fn runtime(&self, route: &ModelRoute) -> Option<Arc<dyn InferenceRuntime>> {
        self.runtimes
            .read()
            .expect("runtime registry poisoned")
            .get(&route.key())
            .cloned()
    }
}

pub(crate) async fn chat_with_provider(
    provider: Arc<dyn LlmProvider>,
    provider_name: String,
    request: LlmRequest,
) -> Result<LlmResponse, LlmGatewayError> {
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
        name: "fastllm_output".to_string(),
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

fn response_to_events(response: LlmResponse) -> Vec<LlmStreamEvent> {
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
    events
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

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CacheConfig, EchoProvider, LlmMessage, RuntimeKind};

    #[tokio::test]
    async fn gateway_routes_to_registered_provider() {
        let gateway = LlmGateway::new().with_provider("echo", Arc::new(EchoProvider));
        let response = gateway
            .chat(LlmRequest::new(
                ModelRoute::new("echo", "test"),
                vec![LlmMessage::user("hello")],
            ))
            .await
            .expect("response");

        assert_eq!(response.text, "hello");
    }

    #[tokio::test]
    async fn gateway_denies_unknown_provider() {
        let err = LlmGateway::new()
            .chat(LlmRequest::new(
                ModelRoute::new("missing", "test"),
                Vec::new(),
            ))
            .await
            .expect_err("unknown provider");

        assert!(err.to_string().contains("missing"));
    }

    #[tokio::test]
    async fn gateway_streams_full_response_as_events() {
        let gateway = LlmGateway::new().with_provider("echo", Arc::new(EchoProvider));
        let events = gateway
            .chat_stream(LlmRequest::new(
                ModelRoute::new("echo", "test"),
                vec![LlmMessage::user("hello")],
            ))
            .await
            .expect("events");

        assert!(matches!(
            &events[0],
            LlmStreamEvent::TextDelta { text } if text == "hello"
        ));
        assert!(matches!(events.last(), Some(LlmStreamEvent::Done { .. })));
    }

    #[tokio::test]
    async fn gateway_caches_successful_chat_responses() {
        let gateway = LlmGateway::builder()
            .cache(CacheConfig {
                ttl_seconds: 60,
                ..CacheConfig::default()
            })
            .build()
            .with_provider("echo", Arc::new(EchoProvider));
        let request = LlmRequest::new(
            ModelRoute::new("echo", "test"),
            vec![LlmMessage::user("hello")],
        );

        let first = gateway.chat(request.clone()).await.expect("first");
        let second = gateway.chat(request).await.expect("second");

        assert_eq!(first, second);
        let metrics = gateway.telemetry().snapshot();
        assert_eq!(metrics.cache_misses, 1);
        assert_eq!(metrics.cache_hits, 1);
    }

    #[tokio::test]
    async fn gateway_loads_configured_local_model_metadata() {
        let route = ModelRoute::new("local", "tiny");
        let gateway = LlmGateway::builder()
            .config(
                GatewayConfig {
                    local_memory_budget_bytes: 2_000,
                    ..GatewayConfig::default()
                }
                .with_model(ModelConfig {
                    route: route.clone(),
                    runtime: RuntimeKind::Local,
                    memory_bytes: 1_000,
                    ttl_seconds: 60,
                    ..ModelConfig::default()
                }),
            )
            .build();

        let info = gateway
            .load_model(ModelLoadRequest {
                route,
                config: BTreeMap::new(),
            })
            .await
            .expect("load");

        assert!(info.loaded);
        assert_eq!(info.memory_bytes, 1_000);
    }
}
