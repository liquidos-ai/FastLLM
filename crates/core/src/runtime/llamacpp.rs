use crate::{
    LlmGatewayError, LlmRequest, LlmResponse, ModelConfig, ModelInfo, ModelLoadReason,
    ModelLoadRequest, ModelState,
    runtime::{InferenceRuntime, LocalModelRuntime},
};
use async_trait::async_trait;
use std::sync::{Arc, RwLock};

pub struct LlamaCppRuntime {
    config: ModelConfig,
    provider: RwLock<Option<Arc<autoagents_llamacpp::LlamaCppProvider>>>,
}

impl LlamaCppRuntime {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            provider: RwLock::new(None),
        }
    }

    async fn ensure_provider(
        &self,
    ) -> Result<Arc<autoagents_llamacpp::LlamaCppProvider>, LlmGatewayError> {
        if let Some(provider) = self
            .provider
            .read()
            .expect("llamacpp provider poisoned")
            .as_ref()
            .cloned()
        {
            return Ok(provider);
        }

        let mut builder = autoagents_llamacpp::LlamaCppProvider::builder()
            .model_source(self.model_source()?)
            .context_reuse(true)
            .n_ctx(self.config.context_tokens);
        if let Some(value) = self
            .config
            .parameters
            .get("n_gpu_layers")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
        {
            builder = builder.n_gpu_layers(value);
        }
        if let Some(value) = self.parameter_u32("max_tokens") {
            builder = builder.max_tokens(value);
        }
        if let Some(value) = self.parameter_f32("temperature") {
            builder = builder.temperature(value);
        }
        if let Some(value) = self.parameter_f32("top_p") {
            builder = builder.top_p(value);
        }
        if let Some(value) = self.parameter_u32("top_k") {
            builder = builder.top_k(value);
        }
        let provider =
            Arc::new(
                builder
                    .build()
                    .await
                    .map_err(|source| LlmGatewayError::Runtime {
                        runtime: "llamacpp".to_string(),
                        message: source.to_string(),
                    })?,
            );
        *self.provider.write().expect("llamacpp provider poisoned") = Some(Arc::clone(&provider));
        Ok(provider)
    }

    fn model_source(&self) -> Result<autoagents_llamacpp::ModelSource, LlmGatewayError> {
        if let Some(model_path) = self.config.model_path.clone() {
            return Ok(autoagents_llamacpp::ModelSource::Gguf { model_path });
        }

        let repo_id = self
            .parameter_str("huggingface_repo_id")
            .or_else(|| self.parameter_str("hf_repo_id"))
            .ok_or_else(|| LlmGatewayError::Runtime {
                runtime: "llamacpp".to_string(),
                message: format!(
                    "model `{}` requires model_path or huggingface_repo_id",
                    self.config.route.key()
                ),
            })?;

        Ok(autoagents_llamacpp::ModelSource::HuggingFace {
            repo_id,
            filename: self
                .parameter_str("huggingface_filename")
                .or_else(|| self.parameter_str("hf_filename")),
            mmproj_filename: self
                .parameter_str("huggingface_mmproj_filename")
                .or_else(|| self.parameter_str("hf_mmproj_filename")),
        })
    }

    fn parameter_str(&self, key: &str) -> Option<String> {
        self.config
            .parameters
            .get(key)
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
    }

    fn parameter_u32(&self, key: &str) -> Option<u32> {
        self.config
            .parameters
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
    }

    fn parameter_f32(&self, key: &str) -> Option<f32> {
        self.config
            .parameters
            .get(key)
            .and_then(serde_json::Value::as_f64)
            .map(|value| value as f32)
    }
}

#[async_trait]
impl InferenceRuntime for LlamaCppRuntime {
    async fn chat(&self, request: LlmRequest) -> Result<LlmResponse, LlmGatewayError> {
        let provider = self.ensure_provider().await?;
        crate::gateway::chat_with_provider(provider, "llamacpp".to_string(), request).await
    }

    async fn load_model(&self, request: ModelLoadRequest) -> Result<ModelInfo, LlmGatewayError> {
        let _provider = self.ensure_provider().await?;
        Ok(ModelInfo {
            route: request.route,
            loaded: true,
            state: ModelState::Loaded,
            memory_bytes: self.config.memory_bytes,
            kv_cache_bytes: self.config.kv_cache_bytes,
            device: self.config.device.clone(),
            expires_at_ms: None,
            load_reason: ModelLoadReason::Explicit,
        })
    }

    async fn unload_model(&self, _route: &crate::ModelRoute) -> Result<(), LlmGatewayError> {
        *self.provider.write().expect("llamacpp provider poisoned") = None;
        Ok(())
    }
}

#[async_trait]
impl LocalModelRuntime for LlamaCppRuntime {
    fn memory_bytes(&self) -> u64 {
        self.config.memory_bytes
    }

    fn kv_cache_bytes(&self) -> u64 {
        self.config.kv_cache_bytes
    }
}
