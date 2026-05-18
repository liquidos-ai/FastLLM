use crate::{LlmGatewayError, LlmProvider};
use autoagents_llm::builder::LLMBuilder;
use autoagents_llm::chat::ReasoningEffort;
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AutoagentsProviderConfig {
    pub provider: String,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub timeout_seconds: Option<u64>,
    pub reasoning: Option<bool>,
    pub reasoning_effort: Option<String>,
    pub reasoning_budget_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub normalize_response: Option<bool>,
    pub extra_body: Option<Value>,
    pub api_version: Option<String>,
    pub deployment_id: Option<String>,
}

impl AutoagentsProviderConfig {
    pub fn new(provider: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            ..Self::default()
        }
    }

    pub fn from_env(provider: impl Into<String>, model: impl Into<String>) -> Self {
        let provider = provider.into();
        let env_prefix = provider.to_ascii_uppercase().replace(['-', '.'], "_");
        Self {
            provider,
            model: Some(model.into()),
            api_key: std::env::var(format!("{env_prefix}_API_KEY")).ok(),
            base_url: std::env::var(format!("{env_prefix}_BASE_URL")).ok(),
            api_version: std::env::var(format!("{env_prefix}_API_VERSION")).ok(),
            deployment_id: std::env::var(format!("{env_prefix}_DEPLOYMENT_ID")).ok(),
            ..Self::default()
        }
    }

    pub fn with_model_config(mut self, config: Option<&Value>) -> Self {
        let Some(config) = config.and_then(Value::as_object) else {
            return self;
        };

        self.max_tokens = config
            .get("max_tokens")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .or(self.max_tokens);
        self.temperature = config
            .get("temperature")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .or(self.temperature);
        self.timeout_seconds = config
            .get("timeout_seconds")
            .and_then(Value::as_u64)
            .or(self.timeout_seconds);
        self.reasoning = config
            .get("reasoning")
            .and_then(Value::as_bool)
            .or(self.reasoning);
        self.reasoning_effort = config
            .get("reasoning_effort")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or(self.reasoning_effort);
        self.reasoning_budget_tokens = config
            .get("reasoning_budget_tokens")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .or(self.reasoning_budget_tokens);
        self.top_p = config
            .get("top_p")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .or(self.top_p);
        self.top_k = config
            .get("top_k")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .or(self.top_k);
        self.normalize_response = config
            .get("normalize_response")
            .and_then(Value::as_bool)
            .or(self.normalize_response);
        self.extra_body = config.get("extra_body").cloned().or(self.extra_body);
        self.api_version = config
            .get("api_version")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or(self.api_version);
        self.deployment_id = config
            .get("deployment_id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or(self.deployment_id);
        self
    }
}

pub fn build_autoagents_provider(
    config: AutoagentsProviderConfig,
) -> Result<Arc<dyn LlmProvider>, LlmGatewayError> {
    match config.provider.as_str() {
        "openai" => build::<autoagents_llm::backends::openai::OpenAI>(config),
        "anthropic" => build::<autoagents_llm::backends::anthropic::Anthropic>(config),
        "ollama" => build::<autoagents_llm::backends::ollama::Ollama>(config),
        "deepseek" => build::<autoagents_llm::backends::deepseek::DeepSeek>(config),
        "xai" => build::<autoagents_llm::backends::xai::XAI>(config),
        "phind" => build::<autoagents_llm::backends::phind::Phind>(config),
        "google" => build::<autoagents_llm::backends::google::Google>(config),
        "groq" => build::<autoagents_llm::backends::groq::Groq>(config),
        "azure-openai" => build::<autoagents_llm::backends::azure_openai::AzureOpenAI>(config),
        "openrouter" => build::<autoagents_llm::backends::openrouter::OpenRouter>(config),
        "minimax" => build::<autoagents_llm::backends::minimax::MiniMax>(config),
        other => Err(LlmGatewayError::UnknownProvider(other.to_string())),
    }
}

fn build<T>(config: AutoagentsProviderConfig) -> Result<Arc<dyn LlmProvider>, LlmGatewayError>
where
    T: autoagents_llm::LLMProvider + autoagents_llm::HasConfig,
    LLMBuilder<T>: BuildAutoagentsProvider<T>,
{
    let provider_name = config.provider.clone();
    BuildAutoagentsProvider::build_provider(apply_common::<T>(config)).map_err(|source| {
        LlmGatewayError::Provider {
            provider: provider_name,
            message: source.to_string(),
        }
    })
}

fn apply_common<T>(config: AutoagentsProviderConfig) -> LLMBuilder<T>
where
    T: autoagents_llm::LLMProvider + autoagents_llm::HasConfig,
{
    let mut builder = LLMBuilder::<T>::new();
    if let Some(api_key) = config.api_key {
        builder = builder.api_key(api_key);
    }
    if let Some(base_url) = config.base_url {
        builder = builder.base_url(base_url);
    }
    if let Some(model) = config.model {
        builder = builder.model(model);
    }
    if let Some(max_tokens) = config.max_tokens {
        builder = builder.max_tokens(max_tokens);
    }
    if let Some(temperature) = config.temperature {
        builder = builder.temperature(temperature);
    }
    if let Some(timeout_seconds) = config.timeout_seconds {
        builder = builder.timeout_seconds(timeout_seconds);
    }
    if let Some(reasoning) = config.reasoning {
        builder = builder.reasoning(reasoning);
    }
    if let Some(reasoning_effort) = config.reasoning_effort {
        builder = match reasoning_effort.as_str() {
            "low" => builder.reasoning_effort(ReasoningEffort::Low),
            "medium" => builder.reasoning_effort(ReasoningEffort::Medium),
            "high" => builder.reasoning_effort(ReasoningEffort::High),
            _ => builder,
        };
    }
    if let Some(reasoning_budget_tokens) = config.reasoning_budget_tokens {
        builder = builder.reasoning_budget_tokens(reasoning_budget_tokens);
    }
    if let Some(top_p) = config.top_p {
        builder = builder.top_p(top_p);
    }
    if let Some(top_k) = config.top_k {
        builder = builder.top_k(top_k);
    }
    if let Some(normalize_response) = config.normalize_response {
        builder = builder.normalize_response(normalize_response);
    }
    if let Some(extra_body) = config.extra_body {
        builder = builder.extra_body(extra_body);
    }
    if let Some(api_version) = config.api_version {
        builder = builder.api_version(api_version);
    }
    if let Some(deployment_id) = config.deployment_id {
        builder = builder.deployment_id(deployment_id);
    }
    builder
}

pub trait BuildAutoagentsProvider<T> {
    fn build_provider(self) -> Result<Arc<dyn LlmProvider>, autoagents_llm::error::LLMError>;
}

macro_rules! impl_build_provider {
    ($($ty:path),+ $(,)?) => {
        $(
            impl BuildAutoagentsProvider<$ty> for LLMBuilder<$ty> {
                fn build_provider(self) -> Result<Arc<dyn LlmProvider>, autoagents_llm::error::LLMError> {
                    Ok(self.build()?)
                }
            }
        )+
    };
}

impl_build_provider!(
    autoagents_llm::backends::openai::OpenAI,
    autoagents_llm::backends::anthropic::Anthropic,
    autoagents_llm::backends::ollama::Ollama,
    autoagents_llm::backends::deepseek::DeepSeek,
    autoagents_llm::backends::xai::XAI,
    autoagents_llm::backends::phind::Phind,
    autoagents_llm::backends::google::Google,
    autoagents_llm::backends::groq::Groq,
    autoagents_llm::backends::azure_openai::AzureOpenAI,
    autoagents_llm::backends::openrouter::OpenRouter,
    autoagents_llm::backends::minimax::MiniMax,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_provider_is_error() {
        let err = match build_autoagents_provider(AutoagentsProviderConfig::new("missing")) {
            Ok(_) => panic!("unknown provider should fail"),
            Err(err) => err,
        };

        assert!(matches!(err, LlmGatewayError::UnknownProvider(_)));
    }

    #[test]
    fn provider_build_errors_are_preserved() {
        let err = match build_autoagents_provider(AutoagentsProviderConfig::new("openai")) {
            Ok(_) => panic!("missing key should fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("OpenAI"));
    }

    #[test]
    fn configured_autoagents_providers_build_without_network() {
        let cases = [
            AutoagentsProviderConfig {
                provider: "openai".to_string(),
                api_key: Some("test".to_string()),
                model: Some("gpt-test".to_string()),
                ..AutoagentsProviderConfig::default()
            },
            AutoagentsProviderConfig {
                provider: "anthropic".to_string(),
                api_key: Some("test".to_string()),
                model: Some("claude-test".to_string()),
                ..AutoagentsProviderConfig::default()
            },
            AutoagentsProviderConfig {
                provider: "ollama".to_string(),
                base_url: Some("http://localhost:11434".to_string()),
                model: Some("llama-test".to_string()),
                ..AutoagentsProviderConfig::default()
            },
            AutoagentsProviderConfig {
                provider: "deepseek".to_string(),
                api_key: Some("test".to_string()),
                model: Some("deepseek-test".to_string()),
                ..AutoagentsProviderConfig::default()
            },
            AutoagentsProviderConfig {
                provider: "xai".to_string(),
                api_key: Some("test".to_string()),
                model: Some("grok-test".to_string()),
                ..AutoagentsProviderConfig::default()
            },
            AutoagentsProviderConfig {
                provider: "phind".to_string(),
                model: Some("phind-test".to_string()),
                ..AutoagentsProviderConfig::default()
            },
            AutoagentsProviderConfig {
                provider: "google".to_string(),
                api_key: Some("test".to_string()),
                model: Some("gemini-test".to_string()),
                ..AutoagentsProviderConfig::default()
            },
            AutoagentsProviderConfig {
                provider: "groq".to_string(),
                api_key: Some("test".to_string()),
                model: Some("llama-test".to_string()),
                ..AutoagentsProviderConfig::default()
            },
            AutoagentsProviderConfig {
                provider: "azure-openai".to_string(),
                api_key: Some("test".to_string()),
                base_url: Some("https://example.test".to_string()),
                api_version: Some("2024-02-01".to_string()),
                deployment_id: Some("deployment".to_string()),
                model: Some("gpt-test".to_string()),
                ..AutoagentsProviderConfig::default()
            },
            AutoagentsProviderConfig {
                provider: "openrouter".to_string(),
                api_key: Some("test".to_string()),
                model: Some("openrouter-test".to_string()),
                ..AutoagentsProviderConfig::default()
            },
            AutoagentsProviderConfig {
                provider: "minimax".to_string(),
                api_key: Some("test".to_string()),
                model: Some("minimax-test".to_string()),
                ..AutoagentsProviderConfig::default()
            },
        ];

        for config in cases {
            build_autoagents_provider(config).expect("provider builds");
        }
    }
}
