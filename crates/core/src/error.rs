#[derive(Debug, thiserror::Error)]
pub enum LlmGatewayError {
    #[error("no LLM provider registered for `{0}`")]
    UnknownProvider(String),
    #[error("provider `{provider}` failed: {message}")]
    Provider { provider: String, message: String },
    #[error("model route `{0}` is not allowed by the effective capability grant")]
    RouteDenied(String),
}
