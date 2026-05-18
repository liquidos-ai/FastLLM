use fastllm::{
    CacheConfig, EchoProvider, GatewayConfig, LlmGateway, LlmMessage, LlmRequest, ModelRoute,
    SchedulerConfig,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gateway = LlmGateway::builder()
        .config(GatewayConfig {
            scheduler: SchedulerConfig {
                max_queue_depth: 8,
                max_concurrent_tasks: 2,
                per_route_concurrency: 1,
                default_deadline_ms: 5_000,
            },
            cache: CacheConfig {
                enabled: true,
                ttl_seconds: 60,
                max_entries: 128,
            },
            ..GatewayConfig::default()
        })
        .build()
        .with_provider("echo", Arc::new(EchoProvider));

    let request = LlmRequest::new(
        ModelRoute::new("echo", "demo"),
        vec![LlmMessage::user("cached scheduler request")],
    )
    .with_request_id("scheduler-demo")
    .with_deadline_ms(5_000);

    let first = gateway.chat(request.clone()).await?;
    let second = gateway.chat(request).await?;

    println!("first response: {}", first.text);
    println!("second response from cache: {}", second.text);
    println!("metrics: {:?}", gateway.telemetry().snapshot());
    Ok(())
}
