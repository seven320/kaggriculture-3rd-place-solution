//! Measure an official action trace with the Rust batch kernel.

use std::env;
use std::error::Error;
use std::fs::File;
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

use kaggriculture_engine::model::{GameState, JointAction};
use kaggriculture_engine::run_batch;
use rayon::ThreadPoolBuilder;
use serde::Deserialize;

const BATCH_SIZES: [usize; 4] = [1, 8, 64, 256];

#[derive(Debug, Deserialize)]
struct ActionStream {
    joint_actions: Vec<JointAction>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    initial: GameState,
    action_stream: ActionStream,
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = env::args()
        .nth(1)
        .ok_or("usage: bench_fixture FIXTURE.jsonl [REPEATS]")?;
    let repeats = env::args()
        .nth(2)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(20);
    let line = BufReader::new(File::open(path)?)
        .lines()
        .find(|line| line.as_ref().is_ok_and(|value| !value.trim().is_empty()))
        .ok_or("fixture file is empty")??;
    let fixture: Fixture = serde_json::from_str(&line)?;
    let pool = ThreadPoolBuilder::new().num_threads(8).build()?;

    println!(
        "threads=8 transitions_per_episode={} repeats={repeats}",
        fixture.action_stream.joint_actions.len()
    );
    for batch_size in BATCH_SIZES {
        let streams = vec![fixture.action_stream.joint_actions.clone(); batch_size];
        let mut elapsed = Duration::ZERO;
        for _ in 0..repeats {
            let states = vec![fixture.initial.clone(); batch_size];
            let started = Instant::now();
            let result = pool.install(|| run_batch(states, &streams));
            elapsed += started.elapsed();
            black_box(result);
        }
        let episodes = (batch_size * repeats) as f64;
        let episodes_per_second = episodes / elapsed.as_secs_f64();
        let transitions_per_second =
            episodes_per_second * fixture.action_stream.joint_actions.len() as f64;
        println!(
            "batch={batch_size:>3} elapsed={:.6}s episodes/s={episodes_per_second:.1} transitions/s={transitions_per_second:.0}",
            elapsed.as_secs_f64(),
        );
    }
    Ok(())
}
