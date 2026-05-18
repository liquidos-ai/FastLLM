use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ModelRoute {
    pub provider: String,
    pub model: String,
}

impl ModelRoute {
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
        }
    }

    pub fn key(&self) -> String {
        format!("{}:{}", self.provider, self.model)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmRequest {
    pub route: ModelRoute,
    pub messages: Vec<LlmMessage>,
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub deadline_ms: Option<u64>,
    #[serde(default)]
    pub priority: u8,
    #[serde(default = "default_cacheable")]
    pub cache: bool,
    #[serde(default)]
    pub tools: Vec<LlmTool>,
    #[serde(default)]
    pub output_schema: Option<Value>,
    #[serde(default)]
    pub parameters: BTreeMap<String, Value>,
    #[serde(default)]
    pub provider_parameters: BTreeMap<String, Value>,
}

impl LlmRequest {
    pub fn new(route: ModelRoute, messages: Vec<LlmMessage>) -> Self {
        Self {
            route,
            messages,
            request_id: None,
            deadline_ms: None,
            priority: 0,
            cache: true,
            tools: Vec::new(),
            output_schema: None,
            parameters: BTreeMap::new(),
            provider_parameters: BTreeMap::new(),
        }
    }

    pub fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    pub fn with_deadline_ms(mut self, deadline_ms: u64) -> Self {
        self.deadline_ms = Some(deadline_ms);
        self
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }
}

fn default_cacheable() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LlmMessage {
    pub role: String,
    pub content: String,
}

impl LlmMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmResponse {
    pub text: String,
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<LlmToolCall>,
    #[serde(default)]
    pub usage: Option<TokenUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum LlmStreamEvent {
    TextDelta { text: String },
    ReasoningDelta { text: String },
    ToolCall { call: LlmToolCall },
    Done { response: LlmResponse },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmToolCall {
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelLoadRequest {
    pub route: ModelRoute,
    #[serde(default)]
    pub config: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelInfo {
    pub route: ModelRoute,
    pub loaded: bool,
    pub state: ModelState,
    pub memory_bytes: u64,
    pub kv_cache_bytes: u64,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub expires_at_ms: Option<u64>,
    pub load_reason: ModelLoadReason,
}

impl ModelInfo {
    pub fn unloaded(route: ModelRoute) -> Self {
        Self {
            route,
            loaded: false,
            state: ModelState::Unloaded,
            memory_bytes: 0,
            kv_cache_bytes: 0,
            device: None,
            expires_at_ms: None,
            load_reason: ModelLoadReason::Explicit,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelState {
    Unloaded,
    Loading,
    Loaded,
    Evicting,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelLoadReason {
    Explicit,
    OnDemand,
    Retry,
    MemoryPressure,
}
