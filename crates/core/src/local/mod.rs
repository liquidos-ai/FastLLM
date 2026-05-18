use crate::{
    LlmGatewayError, ModelConfig, ModelInfo, ModelLoadReason, ModelRoute, ModelState, Telemetry,
};
use std::{
    collections::{BTreeMap, VecDeque},
    sync::Mutex,
};

#[derive(Debug)]
pub struct ModelRegistry {
    models: Mutex<BTreeMap<String, ModelRecord>>,
    memory: MemoryManager,
    telemetry: Telemetry,
}

impl ModelRegistry {
    pub fn new(memory_budget_bytes: u64, telemetry: Telemetry) -> Self {
        Self {
            models: Mutex::new(BTreeMap::new()),
            memory: MemoryManager::new(memory_budget_bytes),
            telemetry,
        }
    }

    pub fn register(&self, config: ModelConfig) {
        let key = config.route.key();
        let info = ModelInfo::unloaded(config.route.clone());
        let record = ModelRecord {
            config,
            info,
            last_used_ms: 0,
        };
        self.models
            .lock()
            .expect("model registry poisoned")
            .insert(key, record);
    }

    pub fn load(
        &self,
        route: &ModelRoute,
        now_ms: u64,
        reason: ModelLoadReason,
    ) -> Result<ModelInfo, LlmGatewayError> {
        let mut models = self.models.lock().expect("model registry poisoned");
        let key = route.key();
        let record = models
            .get_mut(&key)
            .ok_or_else(|| LlmGatewayError::ModelNotLoaded(key.clone()))?;
        if !record.info.loaded {
            let requested = record.config.memory_bytes + record.config.kv_cache_bytes;
            self.memory.reserve(&key, requested)?;
            self.telemetry.record_model_load(route);
        }
        record.last_used_ms = now_ms;
        record.info = ModelInfo {
            route: route.clone(),
            loaded: true,
            state: ModelState::Loaded,
            memory_bytes: record.config.memory_bytes,
            kv_cache_bytes: record.config.kv_cache_bytes,
            device: record.config.device.clone(),
            expires_at_ms: Some(now_ms.saturating_add(record.config.ttl_seconds * 1_000)),
            load_reason: reason,
        };
        Ok(record.info.clone())
    }

    pub fn touch(&self, route: &ModelRoute, now_ms: u64) -> Result<ModelInfo, LlmGatewayError> {
        let mut models = self.models.lock().expect("model registry poisoned");
        let key = route.key();
        let record = models
            .get_mut(&key)
            .ok_or_else(|| LlmGatewayError::ModelNotLoaded(key.clone()))?;
        record.last_used_ms = now_ms;
        if record.info.loaded {
            record.info.expires_at_ms =
                Some(now_ms.saturating_add(record.config.ttl_seconds * 1_000));
        }
        Ok(record.info.clone())
    }

    pub fn info(&self, route: &ModelRoute) -> Option<ModelInfo> {
        self.models
            .lock()
            .expect("model registry poisoned")
            .get(&route.key())
            .map(|record| record.info.clone())
    }

    pub fn unload_expired(&self, now_ms: u64) -> Vec<ModelInfo> {
        let mut unloaded = Vec::new();
        let mut models = self.models.lock().expect("model registry poisoned");
        for (key, record) in models.iter_mut() {
            if record.info.loaded
                && record
                    .info
                    .expires_at_ms
                    .is_some_and(|expires_at| expires_at <= now_ms)
            {
                self.memory.release(key);
                self.telemetry.record_model_unload(&record.info.route);
                record.info.loaded = false;
                record.info.state = ModelState::Unloaded;
                record.info.expires_at_ms = None;
                unloaded.push(record.info.clone());
            }
        }
        unloaded
    }

    pub fn used_memory_bytes(&self) -> u64 {
        self.memory.used()
    }
}

#[derive(Debug, Clone)]
struct ModelRecord {
    config: ModelConfig,
    info: ModelInfo,
    last_used_ms: u64,
}

#[derive(Debug)]
pub struct MemoryManager {
    budget_bytes: u64,
    allocations: Mutex<BTreeMap<String, u64>>,
}

impl MemoryManager {
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            budget_bytes,
            allocations: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn reserve(&self, route: &str, bytes: u64) -> Result<(), LlmGatewayError> {
        let mut allocations = self.allocations.lock().expect("memory manager poisoned");
        let used = allocations.values().copied().sum::<u64>();
        let existing = allocations.get(route).copied().unwrap_or(0);
        let available = self
            .budget_bytes
            .saturating_sub(used.saturating_sub(existing));
        if bytes > available {
            return Err(LlmGatewayError::InsufficientMemory {
                route: route.to_string(),
                requested_bytes: bytes,
                available_bytes: available,
            });
        }
        allocations.insert(route.to_string(), bytes);
        Ok(())
    }

    pub fn release(&self, route: &str) {
        self.allocations
            .lock()
            .expect("memory manager poisoned")
            .remove(route);
    }

    pub fn used(&self) -> u64 {
        self.allocations
            .lock()
            .expect("memory manager poisoned")
            .values()
            .copied()
            .sum()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotState {
    Idle,
    Prefill,
    Decode,
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceSlot {
    pub id: usize,
    pub route: ModelRoute,
    pub state: SlotState,
}

#[derive(Debug)]
pub struct InferenceSlots {
    slots: Mutex<Vec<InferenceSlot>>,
}

impl InferenceSlots {
    pub fn new(route: ModelRoute, count: usize) -> Self {
        let slots = (0..count.max(1))
            .map(|id| InferenceSlot {
                id,
                route: route.clone(),
                state: SlotState::Idle,
            })
            .collect();
        Self {
            slots: Mutex::new(slots),
        }
    }

    pub fn acquire(&self, route: &ModelRoute) -> Option<usize> {
        let mut slots = self.slots.lock().expect("slots poisoned");
        let slot = slots
            .iter_mut()
            .find(|slot| slot.route == *route && slot.state == SlotState::Idle)?;
        slot.state = SlotState::Prefill;
        Some(slot.id)
    }

    pub fn set_state(&self, id: usize, state: SlotState) {
        if let Some(slot) = self
            .slots
            .lock()
            .expect("slots poisoned")
            .iter_mut()
            .find(|slot| slot.id == id)
        {
            slot.state = state;
        }
    }
}

#[derive(Debug, Default)]
pub struct KvCacheManager {
    prefixes: Mutex<BTreeMap<String, VecDeque<String>>>,
}

impl KvCacheManager {
    pub fn remember_prefix(&self, route: &ModelRoute, prefix_hash: impl Into<String>) {
        self.prefixes
            .lock()
            .expect("kv cache poisoned")
            .entry(route.key())
            .or_default()
            .push_back(prefix_hash.into());
    }

    pub fn has_prefix(&self, route: &ModelRoute, prefix_hash: &str) -> bool {
        self.prefixes
            .lock()
            .expect("kv cache poisoned")
            .get(&route.key())
            .is_some_and(|prefixes| prefixes.iter().any(|prefix| prefix == prefix_hash))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RuntimeKind;

    #[test]
    fn registry_loads_and_unloads_with_ttl() {
        let registry = ModelRegistry::new(2_000, Telemetry::new());
        registry.register(ModelConfig {
            route: ModelRoute::new("local", "tiny"),
            runtime: RuntimeKind::Local,
            memory_bytes: 1_000,
            ttl_seconds: 1,
            ..ModelConfig::default()
        });

        let info = registry
            .load(
                &ModelRoute::new("local", "tiny"),
                10,
                ModelLoadReason::Explicit,
            )
            .expect("load");
        assert!(info.loaded);
        assert_eq!(registry.used_memory_bytes(), 1_000);

        let unloaded = registry.unload_expired(1_010);
        assert_eq!(unloaded.len(), 1);
        assert_eq!(registry.used_memory_bytes(), 0);
    }

    #[test]
    fn memory_manager_rejects_oversized_models() {
        let memory = MemoryManager::new(10);
        let err = memory.reserve("local:large", 11).expect_err("oom");

        assert!(matches!(
            err,
            LlmGatewayError::InsufficientMemory {
                route,
                requested_bytes: 11,
                available_bytes: 10
            } if route == "local:large"
        ));
    }

    #[test]
    fn slots_only_acquire_matching_idle_route() {
        let route = ModelRoute::new("local", "a");
        let slots = InferenceSlots::new(route.clone(), 1);

        assert_eq!(slots.acquire(&route), Some(0));
        assert_eq!(slots.acquire(&route), None);
        slots.set_state(0, SlotState::Idle);
        assert_eq!(slots.acquire(&route), Some(0));
    }

    #[test]
    fn kv_cache_tracks_prefixes_by_route() {
        let kv = KvCacheManager::default();
        let route = ModelRoute::new("local", "a");

        kv.remember_prefix(&route, "abc");

        assert!(kv.has_prefix(&route, "abc"));
        assert!(!kv.has_prefix(&ModelRoute::new("local", "b"), "abc"));
    }
}
