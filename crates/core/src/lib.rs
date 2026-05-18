mod autoagents;
mod cache;
mod config;
mod echo;
mod error;
mod gateway;
pub mod local;
mod retry;
pub mod runtime;
mod scheduler;
mod telemetry;
mod types;

pub use autoagents::{ProviderConfig, build_provider};
pub use autoagents_llm as auto;
pub use autoagents_llm::LLMProvider as LlmProvider;
pub use cache::{CacheKey, PromptCache};
pub use config::{
    CacheConfig, GatewayConfig, ModelConfig, ProviderSettings, ProviderType, RetryConfig,
    RuntimeKind, SchedulerConfig,
};
pub use echo::EchoProvider;
pub use error::LlmGatewayError;
pub use gateway::{LlmGateway, LlmGatewayBuilder};
pub use retry::RetryPipeline;
pub use scheduler::ExecutionScheduler;
pub use telemetry::{GatewayMetrics, GatewayMetricsSnapshot, Telemetry};
pub use types::{
    LlmMessage, LlmRequest, LlmResponse, LlmStreamEvent, LlmTool, LlmToolCall, ModelInfo,
    ModelLoadReason, ModelLoadRequest, ModelRoute, ModelState, TokenUsage,
};
