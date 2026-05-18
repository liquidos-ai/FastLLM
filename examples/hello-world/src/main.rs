use fastllm::{LlmGateway, LlmMessage, LlmRequest, ModelRoute, ProviderConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

    if std::env::var("OPENAI_API_KEY").is_err() {
        return Err("set OPENAI_API_KEY before running this example".into());
    }

    let gateway = LlmGateway::new();
    gateway.register_provider_config(ProviderConfig::from_env("openai", &model))?;

    let response = gateway
        .chat(LlmRequest::new(
            ModelRoute::new("openai", model),
            vec![LlmMessage::user("What is the capital of France?")],
        ))
        .await?;

    println!("{}", response.text);
    Ok(())
}
