use crate::ModelRoute;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct GatewayConfig {
    pub providers: BTreeMap<String, ProviderSettings>,
    pub models: BTreeMap<String, ModelConfig>,
    pub scheduler: SchedulerConfig,
    pub cache: CacheConfig,
    pub retry: RetryConfig,
    pub local_memory_budget_bytes: u64,
}

impl GatewayConfig {
    pub fn model(&self, route: &ModelRoute) -> Option<&ModelConfig> {
        self.models
            .get(&route.key())
            .or_else(|| self.models.get(&route.model))
    }

    pub fn with_model(mut self, model: ModelConfig) -> Self {
        self.models.insert(model.route.key(), model);
        self
    }

    pub fn with_provider(mut self, name: impl Into<String>, provider: ProviderSettings) -> Self {
        self.providers.insert(name.into(), provider);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct ProviderSettings {
    pub provider_type: ProviderType,
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    pub timeout_seconds: u64,
    pub defaults: BTreeMap<String, Value>,
}

impl Default for ProviderSettings {
    fn default() -> Self {
        Self {
            provider_type: ProviderType::Cloud,
            base_url: None,
            api_key_env: None,
            timeout_seconds: 60,
            defaults: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProviderType {
    #[default]
    Cloud,
    Local,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct ModelConfig {
    pub route: ModelRoute,
    pub runtime: RuntimeKind,
    pub device: Option<String>,
    pub model_path: Option<String>,
    pub memory_bytes: u64,
    pub kv_cache_bytes: u64,
    pub context_tokens: u32,
    pub max_parallel_sequences: usize,
    pub ttl_seconds: u64,
    pub parameters: BTreeMap<String, Value>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            route: ModelRoute::default(),
            runtime: RuntimeKind::Cloud,
            device: None,
            model_path: None,
            memory_bytes: 0,
            kv_cache_bytes: 0,
            context_tokens: 4096,
            max_parallel_sequences: 1,
            ttl_seconds: 300,
            parameters: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeKind {
    #[default]
    Cloud,
    Local,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct SchedulerConfig {
    pub max_queue_depth: usize,
    pub max_concurrent_tasks: usize,
    pub per_route_concurrency: usize,
    pub default_deadline_ms: u64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_queue_depth: 1024,
            max_concurrent_tasks: 64,
            per_route_concurrency: 8,
            default_deadline_ms: 120_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct CacheConfig {
    pub enabled: bool,
    pub ttl_seconds: u64,
    pub max_entries: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ttl_seconds: 300,
            max_entries: 4096,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct RetryConfig {
    pub max_attempts: usize,
    pub initial_backoff_ms: u64,
    pub max_backoff_ms: u64,
    pub retry_provider_errors: bool,
    pub fallback_routes: BTreeMap<String, Vec<ModelRoute>>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 1,
            initial_backoff_ms: 25,
            max_backoff_ms: 1_000,
            retry_provider_errors: true,
            fallback_routes: BTreeMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_config_builds_defaults_and_models() {
        let config = GatewayConfig {
            cache: CacheConfig {
                ttl_seconds: 60,
                ..CacheConfig::default()
            },
            ..GatewayConfig::default()
        }
        .with_model(ModelConfig {
            route: ModelRoute::new("local", "llama"),
            runtime: RuntimeKind::Local,
            memory_bytes: 1024,
            ..ModelConfig::default()
        });

        assert_eq!(config.cache.ttl_seconds, 60);
        assert_eq!(config.scheduler.max_queue_depth, 1024);
        assert_eq!(
            config
                .model(&ModelRoute::new("local", "llama"))
                .expect("model")
                .runtime,
            RuntimeKind::Local
        );
    }

    #[test]
    fn provider_config_is_inserted_by_name() {
        let config = GatewayConfig::default().with_provider(
            "openai",
            ProviderSettings {
                timeout_seconds: 30,
                ..ProviderSettings::default()
            },
        );

        assert_eq!(config.providers["openai"].timeout_seconds, 30);
    }
}
