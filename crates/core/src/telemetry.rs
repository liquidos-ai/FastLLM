use crate::ModelRoute;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

#[derive(Debug, Default)]
pub struct GatewayMetrics {
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub scheduled_tasks: AtomicU64,
    pub queue_rejections: AtomicU64,
    pub retry_attempts: AtomicU64,
    pub model_loads: AtomicU64,
    pub model_unloads: AtomicU64,
}

impl GatewayMetrics {
    pub fn snapshot(&self) -> GatewayMetricsSnapshot {
        GatewayMetricsSnapshot {
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            scheduled_tasks: self.scheduled_tasks.load(Ordering::Relaxed),
            queue_rejections: self.queue_rejections.load(Ordering::Relaxed),
            retry_attempts: self.retry_attempts.load(Ordering::Relaxed),
            model_loads: self.model_loads.load(Ordering::Relaxed),
            model_unloads: self.model_unloads.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatewayMetricsSnapshot {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub scheduled_tasks: u64,
    pub queue_rejections: u64,
    pub retry_attempts: u64,
    pub model_loads: u64,
    pub model_unloads: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Telemetry {
    metrics: Arc<GatewayMetrics>,
}

impl Telemetry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn metrics(&self) -> Arc<GatewayMetrics> {
        Arc::clone(&self.metrics)
    }

    pub fn snapshot(&self) -> GatewayMetricsSnapshot {
        self.metrics.snapshot()
    }

    pub fn record_cache_hit(&self, _route: &ModelRoute) {
        self.metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cache_miss(&self, _route: &ModelRoute) {
        self.metrics.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_scheduled_task(&self, _route: &ModelRoute) {
        self.metrics.scheduled_tasks.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_queue_rejection(&self) {
        self.metrics
            .queue_rejections
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_retry_attempt(&self, _route: &ModelRoute) {
        self.metrics.retry_attempts.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_model_load(&self, _route: &ModelRoute) {
        self.metrics.model_loads.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_model_unload(&self, _route: &ModelRoute) {
        self.metrics.model_unloads.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_counts_events() {
        let telemetry = Telemetry::new();
        let route = ModelRoute::new("local", "model");

        telemetry.record_cache_hit(&route);
        telemetry.record_retry_attempt(&route);
        telemetry.record_model_load(&route);

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.cache_hits, 1);
        assert_eq!(snapshot.retry_attempts, 1);
        assert_eq!(snapshot.model_loads, 1);
    }
}
