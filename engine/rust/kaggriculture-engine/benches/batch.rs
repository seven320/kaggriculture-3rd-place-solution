use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use kaggriculture_engine::model::{Config, PlayerAction};
use kaggriculture_engine::{JointAction, new_game, run_batch};
use rayon::ThreadPoolBuilder;

const BATCHES: [usize; 4] = [1, 8, 64, 256];

fn pass_stream(steps: usize) -> Vec<JointAction> {
    let pass = PlayerAction::default();
    vec![[pass.clone(), pass]; steps]
}

fn benchmark_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("pass_episode_batch");
    let config = Config::default();
    let transitions_per_episode = (config.episode_steps - 1) as usize;
    let pool = ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .expect("eight-thread benchmark pool");

    for batch_size in BATCHES {
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |bencher, &size| {
                bencher.iter_batched(
                    || {
                        let states = (0..size)
                            .map(|seed| new_game(seed as i64, config.clone()))
                            .collect();
                        let streams = (0..size)
                            .map(|_| pass_stream(transitions_per_episode))
                            .collect::<Vec<_>>();
                        (states, streams)
                    },
                    |(states, streams)| pool.install(|| run_batch(states, &streams)),
                    BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, benchmark_batch);
criterion_main!(benches);
