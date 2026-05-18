use fastllm::{
    GatewayConfig, LlmGateway, LlmMessage, LlmRequest, ModelConfig, ModelLoadRequest, ModelRoute,
    RuntimeKind,
};
use std::collections::BTreeMap;

const GIB: u64 = 1024 * 1024 * 1024;
const DEFAULT_HF_REPO: &str = "unsloth/Qwen3.5-9B-GGUF";
const DEFAULT_HF_FILE: &str = "Qwen3.5-9B-Q4_0.gguf";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let route = ModelRoute::new("local", "llama");

    let gateway = LlmGateway::builder()
        .config(GatewayConfig {
            local_memory_budget_bytes: 16 * GIB,
            ..GatewayConfig::default()
        })
        .build();

    gateway.register_llamacpp_model(local_model(route.clone()));

    let info = gateway
        .load_model(ModelLoadRequest {
            route: route.clone(),
            config: BTreeMap::new(),
        })
        .await?;
    println!(
        "loaded {} with {} bytes",
        info.route.key(),
        info.memory_bytes
    );

    let response = gateway
        .chat(LlmRequest::new(
            route,
            vec![LlmMessage::user(
                "Write one sentence about local inference.",
            )],
        ))
        .await?;

    println!("{}", response.text);
    Ok(())
}

fn local_model(route: ModelRoute) -> ModelConfig {
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
    let model_path = std::env::var("FASTLLM_GGUF_MODEL").ok();
    if model_path.is_some() {
        parameters.clear();
    }

    ModelConfig {
        route: route.clone(),
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
