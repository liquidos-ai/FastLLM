use async_trait::async_trait;
use autoagents_llm::chat::{ChatMessage, ChatProvider, ChatResponse, StructuredOutputFormat, Tool};
use autoagents_llm::completion::{CompletionProvider, CompletionRequest, CompletionResponse};
use autoagents_llm::embedding::EmbeddingProvider;
use autoagents_llm::error::LLMError;
use autoagents_llm::models::ModelsProvider;
use autoagents_llm::{HasConfig, LLMProvider, NoConfig};
use std::fmt;

#[derive(Debug, Default)]
pub struct EchoProvider;

#[async_trait]
impl ChatProvider for EchoProvider {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        Ok(Box::new(EchoResponse {
            text: messages
                .last()
                .map(|message| message.content.clone())
                .unwrap_or_default(),
        }))
    }
}

#[async_trait]
impl CompletionProvider for EchoProvider {
    async fn complete(
        &self,
        request: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: request.prompt.clone(),
        })
    }
}

#[async_trait]
impl EmbeddingProvider for EchoProvider {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Ok(input
            .into_iter()
            .map(|text| vec![text.len() as f32])
            .collect())
    }
}

impl ModelsProvider for EchoProvider {}

impl LLMProvider for EchoProvider {}

impl HasConfig for EchoProvider {
    type Config = NoConfig;
}

#[derive(Debug)]
struct EchoResponse {
    text: String,
}

impl ChatResponse for EchoResponse {
    fn text(&self) -> Option<String> {
        Some(self.text.clone())
    }

    fn tool_calls(&self) -> Option<Vec<autoagents_llm::ToolCall>> {
        None
    }
}

impl fmt::Display for EchoResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn echo_completion_and_embedding_are_deterministic() {
        let provider = EchoProvider;
        let completion = provider
            .complete(&CompletionRequest::new("hello"), None)
            .await
            .expect("completion");
        let embeddings = provider
            .embed(vec!["abc".to_string(), "abcd".to_string()])
            .await
            .expect("embeddings");

        assert_eq!(completion.text, "hello");
        assert_eq!(embeddings, vec![vec![3.0], vec![4.0]]);
    }
}
