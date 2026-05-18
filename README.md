<div align="center">
  <img src="assets/logo.png" alt="LiquidOS Logo" width="200" height="200">

# FastLLM 

**A Unified LLM Gateway with local Models in Rust**

[![Crates.io](https://img.shields.io/crates/v/fastllm.svg)](https://crates.io/crates/fastllm)
[![Documentation](https://docs.rs/fastllm/badge.svg)](https://liquidos-ai.github.io/FastLLM)
[![License](https://img.shields.io/crates/l/fastllm.svg)](https://github.com/liquidos-ai/FastLLM#license)
[![codecov](https://codecov.io/gh/liquidos-ai/FastLLM/graph/badge.svg)](https://codecov.io/gh/liquidos-ai/FastLLM)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/liquidos-ai/FastLLM)

<br />
<strong>Like this project?</strong> <a href="https://github.com/liquidos-ai/AutoAgents">Star us on GitHub</a>
</div>

---

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

## License

AutoAgents is dual-licensed under:

- **MIT License** ([MIT_LICENSE](MIT_LICENSE))
- **Apache License 2.0** ([APACHE_LICENSE](APACHE_LICENSE))

You may choose either license for your use case.

---

## Acknowledgments

Built by the [Liquidos AI](https://liquidos.ai) team and wonderful community of researchers and engineers.

<a href="https://github.com/liquidos-ai/FastLLM/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=liquidos-ai/FastLLM" />
</a>

Special thanks to:

- The Rust community for the excellent ecosystem
- LLM providers for enabling high-quality model APIs
- All contributors who help improve AutoAgents
