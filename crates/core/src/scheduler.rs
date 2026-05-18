use crate::{LlmGatewayError, LlmRequest, LlmResponse, SchedulerConfig, Telemetry};
use std::{
    collections::BTreeMap,
    future::Future,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Clone)]
pub struct ExecutionScheduler {
    config: SchedulerConfig,
    semaphore: Arc<Semaphore>,
    queued: Arc<AtomicUsize>,
    route_counts: Arc<Mutex<BTreeMap<String, usize>>>,
    telemetry: Telemetry,
}

impl ExecutionScheduler {
    pub fn new(config: SchedulerConfig, telemetry: Telemetry) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(config.max_concurrent_tasks.max(1))),
            queued: Arc::new(AtomicUsize::new(0)),
            route_counts: Arc::new(Mutex::new(BTreeMap::new())),
            config,
            telemetry,
        }
    }

    pub async fn execute<F, Fut>(
        &self,
        request: LlmRequest,
        operation: F,
    ) -> Result<LlmResponse, LlmGatewayError>
    where
        F: FnOnce(LlmRequest) -> Fut,
        Fut: Future<Output = Result<LlmResponse, LlmGatewayError>>,
    {
        let queue_position = self.queued.fetch_add(1, Ordering::SeqCst);
        if queue_position >= self.config.max_queue_depth {
            self.queued.fetch_sub(1, Ordering::SeqCst);
            self.telemetry.record_queue_rejection();
            return Err(LlmGatewayError::QueueFull {
                capacity: self.config.max_queue_depth,
            });
        }

        let permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("scheduler semaphore closed");
        self.queued.fetch_sub(1, Ordering::SeqCst);

        let route_guard = self.enter_route(&request)?;
        self.telemetry.record_scheduled_task(&request.route);
        let request_id = request
            .request_id
            .clone()
            .unwrap_or_else(|| request.route.key());
        let deadline = request
            .deadline_ms
            .unwrap_or(self.config.default_deadline_ms)
            .max(1);
        let _permit = permit;
        let _route_guard = route_guard;

        match tokio::time::timeout(Duration::from_millis(deadline), operation(request)).await {
            Ok(result) => result,
            Err(_) => Err(LlmGatewayError::DeadlineExceeded { request_id }),
        }
    }

    fn enter_route(&self, request: &LlmRequest) -> Result<RouteGuard, LlmGatewayError> {
        let route = request.route.key();
        let limit = self.config.per_route_concurrency.max(1);
        let mut counts = self.route_counts.lock().expect("route counts poisoned");
        let count = counts.entry(route.clone()).or_default();
        if *count >= limit {
            return Err(LlmGatewayError::RouteBusy { route, limit });
        }
        *count += 1;
        Ok(RouteGuard {
            route,
            route_counts: Arc::clone(&self.route_counts),
        })
    }

    pub fn queued(&self) -> usize {
        self.queued.load(Ordering::Relaxed)
    }
}

struct RouteGuard {
    route: String,
    route_counts: Arc<Mutex<BTreeMap<String, usize>>>,
}

impl Drop for RouteGuard {
    fn drop(&mut self) {
        let mut counts = self.route_counts.lock().expect("route counts poisoned");
        if let Some(count) = counts.get_mut(&self.route) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                counts.remove(&self.route);
            }
        }
    }
}

#[allow(dead_code)]
struct ScheduledTask {
    request: LlmRequest,
    _permit: OwnedSemaphorePermit,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LlmMessage, ModelRoute};

    #[tokio::test]
    async fn scheduler_executes_operation() {
        let scheduler = ExecutionScheduler::new(SchedulerConfig::default(), Telemetry::new());
        let request = LlmRequest::new(ModelRoute::new("p", "m"), vec![LlmMessage::user("hi")]);

        let response = scheduler
            .execute(request, |_| async {
                Ok(LlmResponse {
                    text: "done".to_string(),
                    reasoning: None,
                    tool_calls: Vec::new(),
                    usage: None,
                })
            })
            .await
            .expect("scheduled");

        assert_eq!(response.text, "done");
    }

    #[tokio::test]
    async fn scheduler_enforces_deadline() {
        let scheduler = ExecutionScheduler::new(
            SchedulerConfig {
                default_deadline_ms: 1,
                ..SchedulerConfig::default()
            },
            Telemetry::new(),
        );
        let request =
            LlmRequest::new(ModelRoute::new("p", "m"), Vec::new()).with_request_id("deadline");

        let err = scheduler
            .execute(request, |_| async {
                tokio::time::sleep(Duration::from_millis(10)).await;
                Ok(LlmResponse {
                    text: String::new(),
                    reasoning: None,
                    tool_calls: Vec::new(),
                    usage: None,
                })
            })
            .await
            .expect_err("deadline");

        assert!(matches!(
            err,
            LlmGatewayError::DeadlineExceeded { request_id } if request_id == "deadline"
        ));
    }
}
