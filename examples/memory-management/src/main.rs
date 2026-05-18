use fastllm::{GatewayConfig, LlmGateway, ModelConfig, ModelLoadRequest, ModelRoute, RuntimeKind};
use std::collections::BTreeMap;

const GIB: u64 = 1024 * 1024 * 1024;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let small = ModelRoute::new("local", "small");
    let large = ModelRoute::new("local", "large");
    let config = GatewayConfig {
        local_memory_budget_bytes: 10 * GIB,
        ..GatewayConfig::default()
    }
    .with_model(local_metadata(small.clone(), 6 * GIB, 2 * GIB))
    .with_model(local_metadata(large.clone(), 8 * GIB, 4 * GIB));

    let gateway = LlmGateway::builder().config(config).build();

    let loaded = gateway
        .load_model(ModelLoadRequest {
            route: small,
            config: BTreeMap::new(),
        })
        .await?;
    println!(
        "loaded {} using {} GiB model + {} GiB KV",
        loaded.route.key(),
        loaded.memory_bytes / GIB,
        loaded.kv_cache_bytes / GIB
    );

    let rejected = gateway
        .load_model(ModelLoadRequest {
            route: large,
            config: BTreeMap::new(),
        })
        .await
        .expect_err("large model should exceed the remaining memory budget");
    println!("large model rejected: {rejected}");

    Ok(())
}

fn local_metadata(route: ModelRoute, memory_bytes: u64, kv_cache_bytes: u64) -> ModelConfig {
    ModelConfig {
        route,
        runtime: RuntimeKind::Local,
        memory_bytes,
        kv_cache_bytes,
        ttl_seconds: 600,
        ..ModelConfig::default()
    }
}
