//! LLM routing and model loading contracts for Odyssey.

mod autoagents;
mod echo;
mod error;
mod gateway;
mod types;

pub use autoagents::{AutoagentsProviderConfig, build_autoagents_provider};
pub use autoagents_llm as auto;
pub use autoagents_llm::LLMProvider as LlmProvider;
pub use echo::EchoProvider;
pub use error::LlmGatewayError;
pub use gateway::LlmGateway;
pub use types::{
    LlmMessage, LlmRequest, LlmResponse, LlmStreamEvent, LlmTool, LlmToolCall, ModelInfo,
    ModelLoadRequest, ModelRoute, TokenUsage,
};
