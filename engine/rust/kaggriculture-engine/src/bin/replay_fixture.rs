//! Replay JSONL fixtures exported by `scripts/export_rust_parity.py`.

use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

use kaggriculture_engine::model::{GameState, JointAction};
use kaggriculture_engine::step;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ActionStream {
    encoding: String,
    joint_actions: Vec<JointAction>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    schema: String,
    seed: i64,
    initial: GameState,
    final_state: GameState,
    action_stream: ActionStream,
    /// Optional v1 extension. Element zero is the initial state and element
    /// `n + 1` is the state after applying joint action `n`.
    #[serde(default)]
    step_states: Option<Vec<GameState>>,
}

fn difference_summary(actual: &GameState, expected: &GameState) -> String {
    if actual.config != expected.config {
        return "config differs".to_owned();
    }
    if actual.seed != expected.seed {
        return format!("seed: rust={} python={}", actual.seed, expected.seed);
    }
    if (actual.step, actual.day, actual.hour) != (expected.step, expected.day, expected.hour) {
        return format!(
            "clock: rust=({},{},{}) python=({},{},{})",
            actual.step, actual.day, actual.hour, expected.step, expected.day, expected.hour
        );
    }
    for player in 0..actual.farms.len() {
        let rust = &actual.farms[player];
        let python = &expected.farms[player];
        if rust.money != python.money {
            return format!(
                "farms[{player}].money: rust={} python={}",
                rust.money, python.money
            );
        }
        if rust.farmer != python.farmer {
            return format!(
                "farms[{player}].farmer: rust={:?} python={:?}",
                rust.farmer, python.farmer
            );
        }
        if rust.hands != python.hands {
            return format!(
                "farms[{player}].hands: rust={:?} python={:?}",
                rust.hands, python.hands
            );
        }
        if rust.unlocked_quadrants != python.unlocked_quadrants {
            return format!(
                "farms[{player}].unlocked_quadrants: rust={} python={}",
                rust.unlocked_quadrants, python.unlocked_quadrants
            );
        }
        if rust.hires_today != python.hires_today {
            return format!(
                "farms[{player}].hires_today: rust={} python={}",
                rust.hires_today, python.hires_today
            );
        }
        if rust.tiles.len() != python.tiles.len() {
            return format!(
                "farms[{player}].tiles length: rust={} python={}",
                rust.tiles.len(),
                python.tiles.len()
            );
        }
        if let Some((tile, (rust_tile, python_tile))) = rust
            .tiles
            .iter()
            .zip(&python.tiles)
            .enumerate()
            .find(|(_, (rust_tile, python_tile))| rust_tile != python_tile)
        {
            return format!(
                "farms[{player}].tiles[{tile}]: rust={rust_tile:?} python={python_tile:?}"
            );
        }
    }
    for player in 0..actual.private.len() {
        let rust = &actual.private[player];
        let python = &expected.private[player];
        if let Some((item, (rust_count, python_count))) = rust
            .shed
            .iter()
            .zip(&python.shed)
            .enumerate()
            .find(|(_, (rust_count, python_count))| rust_count != python_count)
        {
            return format!(
                "private[{player}].shed[{item}]: rust={rust_count} python={python_count}"
            );
        }
        if let Some((crop, (rust_count, python_count))) = rust
            .seeds
            .iter()
            .zip(&python.seeds)
            .enumerate()
            .find(|(_, (rust_count, python_count))| rust_count != python_count)
        {
            return format!(
                "private[{player}].seeds[{crop}]: rust={rust_count} python={python_count}"
            );
        }
        if rust.inventories != python.inventories {
            return format!(
                "private[{player}].inventories: rust={:?} python={:?}",
                rust.inventories, python.inventories
            );
        }
        if rust.inventory_orders != python.inventory_orders {
            return format!(
                "private[{player}].inventory_orders: rust={:?} python={:?}",
                rust.inventory_orders, python.inventory_orders
            );
        }
    }
    if let Some((product, (rust_inventory, python_inventory))) = actual
        .market
        .inventory
        .iter()
        .zip(&expected.market.inventory)
        .enumerate()
        .find(|(_, (rust_inventory, python_inventory))| rust_inventory != python_inventory)
    {
        return format!(
            "market.inventory[{product}]: rust={rust_inventory} python={python_inventory}"
        );
    }
    if let Some((product, (rust_price, python_price))) = actual
        .market
        .prices
        .iter()
        .zip(&expected.market.prices)
        .enumerate()
        .find(|(_, (rust_price, python_price))| rust_price != python_price)
    {
        return format!("market.prices[{product}]: rust={rust_price} python={python_price}");
    }
    if actual.market.params != expected.market.params {
        return "market.params differs".to_owned();
    }
    if actual.town != expected.town {
        return format!("town: rust={:?} python={:?}", actual.town, expected.town);
    }
    if actual.statuses != expected.statuses {
        return format!(
            "statuses: rust={:?} python={:?}",
            actual.statuses, expected.statuses
        );
    }
    if actual.rewards != expected.rewards {
        return format!(
            "rewards: rust={:?} python={:?}",
            actual.rewards, expected.rewards
        );
    }
    "states differ at an unclassified field".to_owned()
}

fn parity_error(
    fixture_number: usize,
    seed: i64,
    state_index: usize,
    actual: &GameState,
    expected: &GameState,
) -> String {
    format!(
        "fixture {fixture_number} seed {seed} diverged at state {state_index} (after {state_index} transitions): {}",
        difference_summary(actual, expected)
    )
}

fn replay_fixture(mut fixture: Fixture, fixture_number: usize) -> Result<usize, String> {
    if fixture.schema != "kaggriculture-rust-parity-v1" {
        return Err(format!(
            "fixture {fixture_number} seed {} uses unsupported schema: {}",
            fixture.seed, fixture.schema
        ));
    }
    if fixture.action_stream.encoding != "explicit" {
        return Err(format!(
            "fixture {fixture_number} seed {} uses unsupported action encoding: {}",
            fixture.seed, fixture.action_stream.encoding
        ));
    }

    if let Some(states) = &fixture.step_states {
        let required = fixture.action_stream.joint_actions.len() + 1;
        if states.len() != required {
            return Err(format!(
                "fixture {fixture_number} seed {} has {} step_states; expected {required} (initial plus one per action)",
                fixture.seed,
                states.len()
            ));
        }
        if fixture.initial != states[0] {
            return Err(parity_error(
                fixture_number,
                fixture.seed,
                0,
                &fixture.initial,
                &states[0],
            ));
        }
    }

    for (action_index, action) in fixture.action_stream.joint_actions.iter().enumerate() {
        step(&mut fixture.initial, action);
        if let Some(states) = &fixture.step_states {
            let state_index = action_index + 1;
            if fixture.initial != states[state_index] {
                return Err(parity_error(
                    fixture_number,
                    fixture.seed,
                    state_index,
                    &fixture.initial,
                    &states[state_index],
                ));
            }
        }
    }
    if fixture.initial != fixture.final_state {
        return Err(format!(
            "fixture {fixture_number} seed {} final_state mismatch after {} transitions: {}",
            fixture.seed,
            fixture.action_stream.joint_actions.len(),
            difference_summary(&fixture.initial, &fixture.final_state)
        ));
    }
    Ok(fixture.action_stream.joint_actions.len())
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = env::args()
        .nth(1)
        .ok_or("usage: replay_fixture FIXTURE.jsonl")?;
    let reader = BufReader::new(File::open(path)?);
    let mut fixture_count = 0_u64;
    let mut transition_count = 0_u64;

    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let fixture: Fixture = serde_json::from_str(&line)
            .map_err(|error| format!("fixture line {} is invalid JSON: {error}", line_index + 1))?;
        transition_count += replay_fixture(fixture, line_index + 1)? as u64;
        fixture_count += 1;
    }

    println!("fixtures={fixture_count} transitions={transition_count} parity=ok");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kaggriculture_engine::model::{Config, PlayerAction, Product};
    use kaggriculture_engine::new_game;

    fn fixture_with_states() -> Fixture {
        let initial = new_game(7, Config::default());
        let action = PlayerAction::default();
        let joint_action = [action.clone(), action];
        let mut after = initial.clone();
        step(&mut after, &joint_action);
        Fixture {
            schema: "kaggriculture-rust-parity-v1".to_owned(),
            seed: 7,
            initial,
            final_state: after.clone(),
            action_stream: ActionStream {
                encoding: "explicit".to_owned(),
                joint_actions: vec![joint_action],
            },
            step_states: Some(vec![new_game(7, Config::default()), after]),
        }
    }

    #[test]
    fn checks_every_expected_state() {
        assert_eq!(replay_fixture(fixture_with_states(), 3), Ok(1));
    }

    #[test]
    fn reports_first_transition_and_field() {
        let mut fixture = fixture_with_states();
        fixture.step_states.as_mut().unwrap()[1].market.prices[Product::Wheat.index()] += 1.0;
        let error = replay_fixture(fixture, 4).unwrap_err();
        assert!(error.contains("fixture 4 seed 7"), "{error}");
        assert!(error.contains("state 1 (after 1 transitions)"), "{error}");
        assert!(error.contains("market.prices[0]"), "{error}");
    }

    #[test]
    fn old_fixture_without_step_states_still_works() {
        let mut fixture = fixture_with_states();
        fixture.step_states = None;
        assert_eq!(replay_fixture(fixture, 1), Ok(1));
    }

    #[test]
    fn rejects_wrong_step_state_count() {
        let mut fixture = fixture_with_states();
        fixture.step_states.as_mut().unwrap().pop();
        let error = replay_fixture(fixture, 2).unwrap_err();
        assert!(error.contains("has 1 step_states; expected 2"), "{error}");
    }
}
