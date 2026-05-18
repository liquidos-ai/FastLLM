use crate::{LlmGatewayError, LlmProvider, LlmRequest, LlmResponse, ModelInfo, ModelLoadRequest};
use async_trait::async_trait;
use std::sync::Arc;

#[async_trait]
pub trait InferenceRuntime: Send + Sync {
    async fn chat(&self, request: LlmRequest) -> Result<LlmResponse, LlmGatewayError>;
    async fn load_model(&self, request: ModelLoadRequest) -> Result<ModelInfo, LlmGatewayError>;
    async fn unload_model(&self, route: &crate::ModelRoute) -> Result<(), LlmGatewayError>;
}

#[async_trait]
pub trait LocalModelRuntime: InferenceRuntime {
    fn memory_bytes(&self) -> u64;
    fn kv_cache_bytes(&self) -> u64;
}

#[async_trait]
pub trait CloudRuntime: InferenceRuntime {}

#[derive(Clone)]
pub struct ProviderRuntime {
    provider_name: String,
    provider: Arc<dyn LlmProvider>,
}

impl ProviderRuntime {
    pub fn new(provider_name: impl Into<String>, provider: Arc<dyn LlmProvider>) -> Self {
        Self {
            provider_name: provider_name.into(),
            provider,
        }
    }

    pub fn provider(&self) -> Arc<dyn LlmProvider> {
        Arc::clone(&self.provider)
    }
}

#[async_trait]
impl InferenceRuntime for ProviderRuntime {
    async fn chat(&self, request: LlmRequest) -> Result<LlmResponse, LlmGatewayError> {
        crate::gateway::chat_with_provider(self.provider(), self.provider_name.clone(), request)
            .await
    }

    async fn load_model(&self, request: ModelLoadRequest) -> Result<ModelInfo, LlmGatewayError> {
        Ok(ModelInfo {
            route: request.route,
            loaded: true,
            state: crate::ModelState::Loaded,
            memory_bytes: 0,
            kv_cache_bytes: 0,
            device: None,
            expires_at_ms: None,
            load_reason: crate::ModelLoadReason::Explicit,
        })
    }

    async fn unload_model(&self, _route: &crate::ModelRoute) -> Result<(), LlmGatewayError> {
        Ok(())
    }
}

#[async_trait]
impl CloudRuntime for ProviderRuntime {}

#[cfg(feature = "local")]
pub mod llamacpp;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EchoProvider, LlmMessage, ModelRoute};

    #[tokio::test]
    async fn provider_runtime_delegates_chat() {
        let runtime = ProviderRuntime::new("echo", Arc::new(EchoProvider));
        let response = runtime
            .chat(LlmRequest::new(
                ModelRoute::new("echo", "test"),
                vec![LlmMessage::user("hi")],
            ))
            .await
            .expect("chat");

        assert_eq!(response.text, "hi");
    }
}
