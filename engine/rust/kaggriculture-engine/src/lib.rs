//! High-throughput, parity-checked Kaggriculture research engine.
//!
//! The official Python environment remains the correctness oracle. This crate
//! is intentionally optimized for batched local rollouts and does not replace
//! Kaggle's judge, timeout, logging, or replay layers.

pub mod batch;
pub mod constants;
pub mod init;
pub mod model;
pub mod rules;

pub use batch::{BatchResult, run_batch};
pub use init::new_game;
pub use model::{GameState, JointAction, MarketOrderTrace, UnitActionTrace};
pub use rules::{
    FailureCounts, step, step_counting_failures, step_with_market_trace, step_with_trace,
};
