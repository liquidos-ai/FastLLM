//! Public FastLLM SDK facade.
//!
//! This crate re-exports the core gateway API so applications can depend on
//! `fastllm` while the workspace keeps the implementation in `fastllm-core`.

pub use fastllm_core::*;
