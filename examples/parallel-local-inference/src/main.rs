use fastllm::{
    GatewayConfig, LlmGateway, LlmMessage, LlmRequest, ModelConfig, ModelRoute, RuntimeKind,
    SchedulerConfig,
};
use std::collections::BTreeMap;

const GIB: u64 = 1024 * 1024 * 1024;
const DEFAULT_HF_REPO: &str = "unsloth/Qwen3.5-9B-GGUF";
const DEFAULT_HF_FILE: &str = "Qwen3.5-9B-Q4_0.gguf";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let route_a = ModelRoute::new("local", "llama-a");
    let route_b = ModelRoute::new("local", "llama-b");
    let gateway = LlmGateway::builder()
        .config(GatewayConfig {
            scheduler: SchedulerConfig {
                max_concurrent_tasks: 2,
                per_route_concurrency: 1,
                ..SchedulerConfig::default()
            },
            local_memory_budget_bytes: 32 * GIB,
            ..GatewayConfig::default()
        })
        .build();

    gateway.register_llamacpp_model(local_model(
        route_a.clone(),
        std::env::var("FASTLLM_GGUF_MODEL_A")
            .or_else(|_| std::env::var("FASTLLM_GGUF_MODEL"))
            .ok(),
    ));
    gateway.register_llamacpp_model(local_model(
        route_b.clone(),
        std::env::var("FASTLLM_GGUF_MODEL_B")
            .or_else(|_| std::env::var("FASTLLM_GGUF_MODEL"))
            .ok(),
    ));

    let first = gateway.chat(LlmRequest::new(
        route_a,
        vec![LlmMessage::user("Give one benefit of small local models.")],
    ));
    let second = gateway.chat(LlmRequest::new(
        route_b,
        vec![LlmMessage::user(
            "Give one limitation of small local models.",
        )],
    ));

    let (first, second) = tokio::try_join!(first, second)?;
    println!("model A: {}", first.text);
    println!("model B: {}", second.text);
    println!("metrics: {:?}", gateway.telemetry().snapshot());

    Ok(())
}

fn local_model(route: ModelRoute, model_path: Option<String>) -> ModelConfig {
    let mut parameters = BTreeMap::from([
        (
            "huggingface_repo_id".to_string(),
            serde_json::json!(DEFAULT_HF_REPO),
        ),
        (
            "huggingface_filename".to_string(),
            serde_json::json!(DEFAULT_HF_FILE),
        ),
        ("max_tokens".to_string(), serde_json::json!(256)),
        ("temperature".to_string(), serde_json::json!(0.7)),
    ]);
    if model_path.is_some() {
        parameters.clear();
    }

    ModelConfig {
        route,
        runtime: RuntimeKind::Local,
        model_path,
        memory_bytes: 8 * GIB,
        kv_cache_bytes: 2 * GIB,
        context_tokens: 4096,
        max_parallel_sequences: 1,
        ttl_seconds: 600,
        parameters,
        ..ModelConfig::default()
    }
}
