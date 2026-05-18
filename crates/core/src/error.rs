#[derive(Debug, thiserror::Error)]
pub enum LlmGatewayError {
    #[error("no LLM provider registered for `{0}`")]
    UnknownProvider(String),
    #[error("provider `{provider}` failed: {message}")]
    Provider { provider: String, message: String },
    #[error("model route `{0}` is not allowed by the effective capability grant")]
    RouteDenied(String),
    #[error("gateway configuration failed: {message}")]
    Config { message: String },
    #[error("scheduler queue is full at capacity {capacity}")]
    QueueFull { capacity: usize },
    #[error("route `{route}` is already at its concurrency limit of {limit}")]
    RouteBusy { route: String, limit: usize },
    #[error("request `{request_id}` exceeded its deadline")]
    DeadlineExceeded { request_id: String },
    #[error("model `{0}` is not loaded")]
    ModelNotLoaded(String),
    #[error(
        "insufficient memory for `{route}`: requested {requested_bytes} bytes with {available_bytes} bytes available"
    )]
    InsufficientMemory {
        route: String,
        requested_bytes: u64,
        available_bytes: u64,
    },
    #[error("cache backend failed: {message}")]
    CacheBackend { message: String },
    #[error("retry exhausted for `{route}` after {attempts} attempts: {message}")]
    RetryExhausted {
        route: String,
        attempts: usize,
        message: String,
    },
    #[error("runtime `{runtime}` failed: {message}")]
    Runtime { runtime: String, message: String },
}
