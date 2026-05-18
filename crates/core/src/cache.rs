use crate::{CacheConfig, LlmRequest, LlmResponse};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CacheKey(String);

impl CacheKey {
    pub fn new(request: &LlmRequest) -> Self {
        #[derive(Serialize)]
        struct StableRequest<'a> {
            route: &'a crate::ModelRoute,
            messages: &'a [crate::LlmMessage],
            tools: &'a [crate::LlmTool],
            output_schema: &'a Option<serde_json::Value>,
            parameters: &'a BTreeMap<String, serde_json::Value>,
        }

        let stable = StableRequest {
            route: &request.route,
            messages: &request.messages,
            tools: &request.tools,
            output_schema: &request.output_schema,
            parameters: &request.parameters,
        };
        let encoded = serde_json::to_string(&stable).unwrap_or_else(|_| request.route.key());
        Self(encoded)
    }
}

#[derive(Debug, Clone)]
struct CacheEntry {
    response: LlmResponse,
    expires_at: Instant,
    sequence: u64,
}

#[derive(Debug)]
pub struct PromptCache {
    config: CacheConfig,
    entries: Mutex<BTreeMap<CacheKey, CacheEntry>>,
    sequence: AtomicU64,
}

impl PromptCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            entries: Mutex::new(BTreeMap::new()),
            sequence: AtomicU64::new(0),
        }
    }

    pub fn get(&self, request: &LlmRequest) -> Option<LlmResponse> {
        if !self.config.enabled || !request.cache {
            return None;
        }
        let key = CacheKey::new(request);
        let mut entries = self.entries.lock().expect("cache mutex poisoned");
        let entry = entries.get(&key)?;
        if Instant::now() >= entry.expires_at {
            entries.remove(&key);
            return None;
        }
        Some(entry.response.clone())
    }

    pub fn insert(&self, request: &LlmRequest, response: LlmResponse) {
        if !self.config.enabled || !request.cache || self.config.ttl_seconds == 0 {
            return;
        }
        let key = CacheKey::new(request);
        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed);
        let expires_at = Instant::now() + Duration::from_secs(self.config.ttl_seconds);
        let mut entries = self.entries.lock().expect("cache mutex poisoned");
        entries.insert(
            key,
            CacheEntry {
                response,
                expires_at,
                sequence,
            },
        );
        while entries.len() > self.config.max_entries {
            let Some(oldest_key) = entries
                .iter()
                .min_by_key(|(_, entry)| entry.sequence)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            entries.remove(&oldest_key);
        }
    }

    pub fn len(&self) -> usize {
        self.entries.lock().expect("cache mutex poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries
            .lock()
            .expect("cache mutex poisoned")
            .is_empty()
    }
}

impl Default for PromptCache {
    fn default() -> Self {
        Self::new(CacheConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LlmMessage, ModelRoute};

    #[test]
    fn cache_ignores_provider_only_parameters() {
        let cache = PromptCache::new(CacheConfig::default());
        let mut first = LlmRequest::new(
            ModelRoute::new("openai", "gpt"),
            vec![LlmMessage::user("hello")],
        );
        let mut second = first.clone();
        first
            .provider_parameters
            .insert("trace".to_string(), serde_json::json!("a"));
        second
            .provider_parameters
            .insert("trace".to_string(), serde_json::json!("b"));

        cache.insert(
            &first,
            LlmResponse {
                text: "cached".to_string(),
                reasoning: None,
                tool_calls: Vec::new(),
                usage: None,
            },
        );

        assert_eq!(cache.get(&second).expect("hit").text, "cached");
    }

    #[test]
    fn cache_respects_disabled_requests() {
        let cache = PromptCache::new(CacheConfig::default());
        let mut request = LlmRequest::new(ModelRoute::new("p", "m"), Vec::new());
        request.cache = false;

        cache.insert(
            &request,
            LlmResponse {
                text: "nope".to_string(),
                reasoning: None,
                tool_calls: Vec::new(),
                usage: None,
            },
        );

        assert!(cache.get(&request).is_none());
    }

    #[test]
    fn cache_respects_zero_ttl_and_global_disable() {
        let request = LlmRequest::new(ModelRoute::new("p", "m"), vec![LlmMessage::user("hello")]);
        let response = LlmResponse {
            text: "value".to_string(),
            reasoning: None,
            tool_calls: Vec::new(),
            usage: None,
        };

        let zero_ttl = PromptCache::new(CacheConfig {
            ttl_seconds: 0,
            ..CacheConfig::default()
        });
        zero_ttl.insert(&request, response.clone());
        assert!(zero_ttl.is_empty());

        let disabled = PromptCache::new(CacheConfig {
            enabled: false,
            ..CacheConfig::default()
        });
        disabled.insert(&request, response);
        assert!(disabled.get(&request).is_none());
    }

    #[test]
    fn cache_evicts_oldest_entry_when_full() {
        let cache = PromptCache::new(CacheConfig {
            max_entries: 1,
            ..CacheConfig::default()
        });
        let first = LlmRequest::new(ModelRoute::new("p", "a"), Vec::new());
        let second = LlmRequest::new(ModelRoute::new("p", "b"), Vec::new());

        cache.insert(
            &first,
            LlmResponse {
                text: "first".to_string(),
                reasoning: None,
                tool_calls: Vec::new(),
                usage: None,
            },
        );
        cache.insert(
            &second,
            LlmResponse {
                text: "second".to_string(),
                reasoning: None,
                tool_calls: Vec::new(),
                usage: None,
            },
        );

        assert_eq!(cache.len(), 1);
        assert!(cache.get(&first).is_none());
        assert_eq!(cache.get(&second).expect("second remains").text, "second");
    }
}
