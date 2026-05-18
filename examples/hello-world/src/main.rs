use fastllm_core::{AutoagentsProviderConfig, LlmGateway, LlmMessage, LlmRequest, ModelRoute};
use std::collections::BTreeMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

    if std::env::var("OPENAI_API_KEY").is_err() {
        return Err("set OPENAI_API_KEY before running this example".into());
    }

    let mut gateway = LlmGateway::new();
    gateway.register_autoagents_provider(AutoagentsProviderConfig::from_env("openai", &model))?;

    let response = gateway
        .chat(LlmRequest {
            route: ModelRoute::new("openai", model),
            messages: vec![LlmMessage::user("What is the capital of France?")],
            tools: Vec::new(),
            output_schema: None,
            parameters: BTreeMap::new(),
        })
        .await?;

    println!("{}", response.text);
    Ok(())
}
