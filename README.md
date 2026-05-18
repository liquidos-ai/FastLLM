# FastLLM

FastLLM is a Rust workspace for routing LLM requests through a unified
gateway API. It provides typed routing contracts, provider registration,
scheduling, prompt caching, retries, local model metadata, and adapters backed
by `autoagents-llm`.

Configuration is typed Rust API. FastLLM does not currently expose YAML or TOML
configuration loading.

The current workspace is intentionally compact:

| Path | Purpose |
| --- | --- |
| `crates/fastllm` | Public SDK facade exposed as `fastllm` |
| `crates/core` | Core gateway implementation exposed as `fastllm-core` |
| `examples/hello-world` | Minimal OpenAI chat example |
| `examples/local-model-inference` | Single local GGUF inference with `autoagents-llamacpp` |
| `examples/parallel-local-inference` | Two local model routes executed concurrently |
| `examples/scheduler-showcase` | Scheduler, prompt cache, and telemetry without external services |
| `examples/memory-management` | Local model memory-budget admission |

## Quick Start

Build and check the workspace:

```sh
cargo check
```

Run the OpenAI hello-world example:

```sh
OPENAI_API_KEY=sk-... cargo run -p fastllm-example-hello-world
```

The example defaults to `gpt-4o-mini`. Set `OPENAI_MODEL` to use a different
OpenAI chat model:

```sh
OPENAI_API_KEY=sk-... OPENAI_MODEL=gpt-4o-mini cargo run -p fastllm-example-hello-world
```

Run examples that do not need external services:

```sh
cargo run -p fastllm-example-scheduler-showcase
cargo run -p fastllm-example-memory-management
```

Run the default local GGUF model from Hugging Face
(`unsloth/Qwen3.5-9B-GGUF`, `Qwen3.5-9B-Q4_0.gguf`):

```sh
cargo run -p fastllm-example-local-model-inference
```

Override with a local GGUF file:

```sh
FASTLLM_GGUF_MODEL=/models/model.gguf cargo run -p fastllm-example-local-model-inference
```

Run two local routes concurrently. By default both routes use the same Hugging
Face model; set local paths to override:

```sh
FASTLLM_GGUF_MODEL_A=/models/a.gguf \
FASTLLM_GGUF_MODEL_B=/models/b.gguf \
cargo run -p fastllm-example-parallel-local-inference
```

## Cloud Provider Example

Register an `autoagents-llm` provider and send one chat request:

```rust
use fastllm::{LlmGateway, LlmMessage, LlmRequest, ModelRoute, ProviderConfig};

let gateway = LlmGateway::new();
gateway.register_provider_config(ProviderConfig::from_env("openai", "gpt-4o-mini"))?;

let response = gateway
    .chat(LlmRequest::new(
        ModelRoute::new("openai", "gpt-4o-mini"),
        vec![LlmMessage::user("What is the capital of France?")],
    ))
    .await?;
println!("{}", response.text);
```

## Typed Configuration

Compose scheduler, cache, retry, and local model policy with Rust structs:

```rust
use fastllm::{
    CacheConfig, GatewayConfig, LlmGateway, ModelConfig, ModelRoute, RetryConfig,
    RuntimeKind, SchedulerConfig,
};

let route = ModelRoute::new("local", "llama-3.2");
let config = GatewayConfig {
    scheduler: SchedulerConfig {
        max_queue_depth: 2048,
        max_concurrent_tasks: 64,
        per_route_concurrency: 4,
        default_deadline_ms: 120_000,
    },
    cache: CacheConfig {
        enabled: true,
        ttl_seconds: 300,
        max_entries: 4096,
    },
    retry: RetryConfig {
        max_attempts: 2,
        ..RetryConfig::default()
    },
    local_memory_budget_bytes: 24 * 1024 * 1024 * 1024,
    ..GatewayConfig::default()
}
.with_model(ModelConfig {
    route,
    runtime: RuntimeKind::Local,
    model_path: Some("/models/llama.gguf".to_string()),
    memory_bytes: 8 * 1024 * 1024 * 1024,
    kv_cache_bytes: 2 * 1024 * 1024 * 1024,
    max_parallel_sequences: 4,
    ttl_seconds: 600,
    ..ModelConfig::default()
});

let gateway = LlmGateway::builder().config(config).build();
```

With the `local` feature enabled, `register_llamacpp_model` attaches a
lazy-loading local runtime backed by `autoagents-llamacpp`.

## Runtime Features

- `ExecutionScheduler` applies bounded admission, per-route concurrency, and
  request deadlines before dispatch.
- `PromptCache` uses canonical request keys, TTL, and entry-limit eviction while
  excluding provider-only parameters from cache identity.
- `RetryPipeline` applies typed retry and fallback policies around scheduler
  execution.
- `ModelRegistry`, `MemoryManager`, `InferenceSlots`, and `KvCacheManager`
  track local residency, memory pressure, parallel slots, and KV-prefix
  metadata.
- `Telemetry` exposes lightweight counters for cache hits/misses, scheduling,
  retries, and model load/unload events.

Required environment variables:

| Variable | Description |
| --- | --- |
| `OPENAI_API_KEY` | API key used by the OpenAI provider |
| `OPENAI_MODEL` | Optional model override for the example |

## Development

Format and check before sending changes:

```sh
cargo fmt --all --check
cargo check
```
