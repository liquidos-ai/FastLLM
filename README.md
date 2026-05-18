# FastLLM

FastLLM is a Rust workspace for routing LLM requests through a unified
gateway API. It provides the provider registry, model route types, request and
response contracts, and adapters backed by `autoagents-llm`.

The current workspace is intentionally compact:

| Path | Purpose |
| --- | --- |
| `crates/fastllm` | Public SDK facade exposed as `fastllm` |
| `crates/core` | Core gateway implementation exposed as `fastllm-core` |
| `examples/hello-world` | Minimal OpenAI chat example |

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

## Gateway Example

The example creates a gateway, registers the OpenAI provider from environment
configuration, sends one user message, and prints the model response:

```rust
let mut gateway = LlmGateway::new();
gateway.register_autoagents_provider(AutoagentsProviderConfig::from_env(
    "openai",
    "gpt-4o-mini",
))?;

let response = gateway.chat(request).await?;
println!("{}", response.text);
```

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
