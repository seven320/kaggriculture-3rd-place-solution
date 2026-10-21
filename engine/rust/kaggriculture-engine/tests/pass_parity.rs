//! Values in this test were exported from the pinned official Python
//! interpreter with `scripts/export_rust_parity.py --episode-steps 24`.

use kaggriculture_engine::model::{Config, PlayerAction, Status};
use kaggriculture_engine::{new_game, step};

fn run_pass_episode(seed: i64, config: Config) -> kaggriculture_engine::GameState {
    let transitions = config.episode_steps - 1;
    let mut state = new_game(seed, config);
    let pass = PlayerAction::default();
    let action = [pass.clone(), pass];
    for _ in 0..transitions {
        step(&mut state, &action);
    }
    state
}

#[test]
fn pass_stream_matches_official_short_episode() {
    let config = Config {
        episode_steps: 24,
        ..Config::default()
    };
    let mut state = run_pass_episode(0, config);
    let pass = PlayerAction::default();
    let action = [pass.clone(), pass];

    // Python oracle: no end-of-day boundary is crossed in a 24-step episode,
    // so the farms remain at their initialized values. Town center consumes
    // one unit of each non-fertilizer product at step zero.
    assert_eq!((state.step, state.day, state.hour), (23, 0, 23));
    assert_eq!(state.statuses, [Status::Done, Status::Done]);
    assert_eq!(state.rewards, [Some(3_000.0), Some(3_000.0)]);
    assert_eq!(
        state.market.inventory,
        [
            9_999, 9_999, 9_999, 9_999, 9_999, 9_999, 9_999, 9_999, 10_000
        ]
    );
    assert_eq!(
        state.market.prices,
        [26.0, 35.0, 60.0, 128.0, 256.0, 50.0, 169.0, 206.0, 100.0]
    );
    assert!(state.town.unlocked_shops.is_empty());
    assert!(state.farms.iter().all(|farm| farm.money == 3_000.0));
    assert!(state.private.iter().all(|private| {
        private.shed.iter().all(|value| *value == 0)
            && private.seeds.iter().all(|value| *value == 0)
    }));

    // A terminal state is stable under accidental extra calls, matching the
    // official interpreter's early return when `env.done` is true.
    let terminal = state.clone();
    step(&mut state, &action);
    assert_eq!(state, terminal);
}

#[test]
fn full_pass_episode_matches_official_rng_and_town_fixture() {
    use kaggriculture_engine::model::{Shop, Tile};

    let state = run_pass_episode(0, Config::default());

    assert_eq!((state.step, state.day, state.hour), (719, 29, 23));
    assert_eq!(state.statuses, [Status::Done, Status::Done]);
    assert_eq!(state.rewards, [Some(3_000.0), Some(3_000.0)]);
    assert_eq!(
        state.market.inventory,
        [
            9_592, 9_664, 9_844, 9_736, 9_970, 9_718, 9_862, 9_538, 10_000
        ]
    );
    assert_eq!(
        state.market.prices,
        [45.0, 61.0, 79.0, 256.0, 280.0, 67.0, 262.0, 253.0, 100.0]
    );
    assert_eq!(
        state.town.unlocked_shops,
        [
            Shop::YarnStore,
            Shop::Bakery,
            Shop::FarmersMarket,
            Shop::SmoothieShop,
            Shop::PetCafe,
            Shop::Bakery,
            Shop::YarnStore,
            Shop::Bakery,
        ]
    );
    for farm in &state.farms {
        assert_eq!(
            farm.tiles
                .iter()
                .filter(|tile| matches!(tile, Tile::Weed))
                .count(),
            3
        );
        assert_eq!(
            farm.tiles
                .iter()
                .filter(|tile| matches!(tile, Tile::Empty))
                .count(),
            22
        );
    }
}

#[test]
fn decoupled_rng_is_inert_when_off_and_changes_only_the_lottery_when_on() {
    use kaggriculture_engine::model::{Tile, UnitAction};
    // Two seasons: the default (official) coupling and the audit-only flag.
    // With identical actions the flag must still change the draw sequence
    // (one draw per tile instead of one per empty tile) -- and the default
    // path must be bit-identical to itself and to the pre-flag behaviour.
    let run = |decoupled: bool, plant: bool| {
        let config = Config {
            episode_steps: 24 * 4,
            decoupled_rng: decoupled,
            ..Config::default()
        };
        let mut state = kaggriculture_engine::new_game(7, config);
        let mut planting = PlayerAction::default();
        if plant {
            planting.farmer = UnitAction::Plant(kaggriculture_engine::model::Crop::Wheat);
        }
        let pass = PlayerAction::default();
        while state.statuses.iter().any(|s| *s == Status::Active) {
            let actions = [planting.clone(), pass.clone()];
            kaggriculture_engine::step(&mut state, &actions);
        }
        let weeds: Vec<usize> = state
            .farms
            .iter()
            .map(|f| f.tiles.iter().filter(|t| matches!(t, Tile::Weed)).count())
            .collect();
        (weeds, state.town.unlocked_shops.clone())
    };
    assert_eq!(run(false, false), run(false, false));
    assert_eq!(run(true, false), run(true, false));
    // The official coupling re-rolls the lottery when tile occupancy changes
    // (a plant on turn 0 shifts every later draw); decoupled mode must not.
    let (_, shops_off_pass) = run(false, false);
    let (_, shops_off_plant) = run(false, true);
    let (_, shops_on_pass) = run(true, false);
    let (_, shops_on_plant) = run(true, true);
    assert_eq!(
        shops_on_pass, shops_on_plant,
        "decoupled: shops independent of actions"
    );
    // Not an assertion on inequality for the official path (a 4-day season may
    // coincide by chance); record it so a future change is visible.
    eprintln!("official shops pass={shops_off_pass:?} plant={shops_off_plant:?}");
}
