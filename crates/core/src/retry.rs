use crate::{LlmGatewayError, LlmRequest, LlmResponse, ModelRoute, RetryConfig, Telemetry};
use std::{future::Future, time::Duration};

#[derive(Clone)]
pub struct RetryPipeline {
    config: RetryConfig,
    telemetry: Telemetry,
}

impl RetryPipeline {
    pub fn new(config: RetryConfig, telemetry: Telemetry) -> Self {
        Self { config, telemetry }
    }

    pub async fn execute<F, Fut>(
        &self,
        request: LlmRequest,
        mut operation: F,
    ) -> Result<LlmResponse, LlmGatewayError>
    where
        F: FnMut(LlmRequest) -> Fut,
        Fut: Future<Output = Result<LlmResponse, LlmGatewayError>>,
    {
        let routes = self.routes_for(&request.route);
        let attempts_per_route = self.config.max_attempts.max(1);
        let mut total_attempts = 0;
        let mut last_error = None;

        for route in routes {
            let mut routed_request = request.clone();
            routed_request.route = route.clone();
            for attempt in 0..attempts_per_route {
                total_attempts += 1;
                if total_attempts > 1 {
                    self.telemetry.record_retry_attempt(&route);
                }
                match operation(routed_request.clone()).await {
                    Ok(response) => return Ok(response),
                    Err(err) if self.should_retry(&err) && attempt + 1 < attempts_per_route => {
                        last_error = Some(err);
                        self.sleep_before_retry(attempt).await;
                    }
                    Err(err) => {
                        last_error = Some(err);
                        break;
                    }
                }
            }
        }

        let message = last_error
            .map(|error| error.to_string())
            .unwrap_or_else(|| "no route attempted".to_string());
        Err(LlmGatewayError::RetryExhausted {
            route: request.route.key(),
            attempts: total_attempts,
            message,
        })
    }

    fn routes_for(&self, route: &ModelRoute) -> Vec<ModelRoute> {
        let mut routes = vec![route.clone()];
        if let Some(fallbacks) = self.config.fallback_routes.get(&route.key()) {
            routes.extend(fallbacks.iter().cloned());
        }
        routes
    }

    fn should_retry(&self, error: &LlmGatewayError) -> bool {
        matches!(error, LlmGatewayError::Provider { .. }) && self.config.retry_provider_errors
    }

    async fn sleep_before_retry(&self, attempt: usize) {
        let base = self.config.initial_backoff_ms;
        if base == 0 {
            return;
        }
        let factor = 1_u64.checked_shl(attempt as u32).unwrap_or(u64::MAX);
        let delay = base.saturating_mul(factor).min(self.config.max_backoff_ms);
        tokio::time::sleep(Duration::from_millis(delay)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LlmMessage, ModelRoute};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[tokio::test]
    async fn retry_pipeline_retries_provider_errors() {
        let pipeline = RetryPipeline::new(
            RetryConfig {
                max_attempts: 2,
                initial_backoff_ms: 0,
                ..RetryConfig::default()
            },
            Telemetry::new(),
        );
        let calls = Arc::new(AtomicUsize::new(0));
        let request = LlmRequest::new(ModelRoute::new("p", "m"), vec![LlmMessage::user("hi")]);

        let response = pipeline
            .execute(request, {
                let calls = Arc::clone(&calls);
                move |_| {
                    let calls = Arc::clone(&calls);
                    async move {
                        if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                            Err(LlmGatewayError::Provider {
                                provider: "p".to_string(),
                                message: "transient".to_string(),
                            })
                        } else {
                            Ok(LlmResponse {
                                text: "ok".to_string(),
                                reasoning: None,
                                tool_calls: Vec::new(),
                                usage: None,
                            })
                        }
                    }
                }
            })
            .await
            .expect("retry success");

        assert_eq!(response.text, "ok");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn retry_pipeline_uses_fallback_routes() {
        let mut fallback_routes = std::collections::BTreeMap::new();
        fallback_routes.insert(
            "primary:a".to_string(),
            vec![ModelRoute::new("fallback", "b")],
        );
        let pipeline = RetryPipeline::new(
            RetryConfig {
                fallback_routes,
                initial_backoff_ms: 0,
                ..RetryConfig::default()
            },
            Telemetry::new(),
        );
        let request = LlmRequest::new(ModelRoute::new("primary", "a"), Vec::new());

        let response = pipeline
            .execute(request, |request| async move {
                if request.route.provider == "primary" {
                    Err(LlmGatewayError::Provider {
                        provider: "primary".to_string(),
                        message: "down".to_string(),
                    })
                } else {
                    Ok(LlmResponse {
                        text: request.route.key(),
                        reasoning: None,
                        tool_calls: Vec::new(),
                        usage: None,
                    })
                }
            })
            .await
            .expect("fallback success");

        assert_eq!(response.text, "fallback:b");
    }
}
