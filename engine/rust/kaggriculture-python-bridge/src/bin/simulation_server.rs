use std::io::{self, BufRead, Write};

use anyhow::{Context, Result};
use kaggriculture_engine::{new_game, step};
use kaggriculture_engine::model::{Config, GameState, Status};
use kaggriculture_python_bridge::observation_value;
use serde_json::{Value, json};

fn snapshot(state: &GameState) -> Value {
    json!({
        "observations": [observation_value(state, 0), observation_value(state, 1)],
        "rewards": state.rewards.iter().map(|reward| reward.or(Some(0.0))).collect::<Vec<_>>(),
        "statuses": state.statuses.iter().map(|s| match s { Status::Active => "ACTIVE", Status::Done => "DONE" }).collect::<Vec<_>>(),
        "done": state.statuses.iter().all(|s| *s == Status::Done),
    })
}

fn main() -> Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    let mut state: Option<GameState> = None;
    for line in stdin.lock().lines() {
        let request: Value = serde_json::from_str(&line?)?;
        let response = match request["command"].as_str() {
            Some("start") => {
                let seed = request["seed"].as_i64().context("start.seed must be i64")?;
                let mut config = Config::default();
                if let Some(n) = request["configuration"]["episodeSteps"].as_u64() {
                    config.episode_steps = u32::try_from(n)?;
                }
                if let Some(n) = request["configuration"]["boardSize"].as_u64() {
                    config.board_size = u16::try_from(n)?;
                }
                macro_rules! u32_config { ($json:literal, $field:ident) => { if let Some(n) = request["configuration"][$json].as_u64() { config.$field = u32::try_from(n)?; } }; }
                u32_config!("startingMoney", starting_money);
                u32_config!("turnsPerDay", turns_per_day);
                u32_config!("shedCapacity", shed_capacity);
                u32_config!("townShopUnlockInterval", town_shop_unlock_interval);
                u32_config!("townShopSellInterval", town_shop_sell_interval);
                u32_config!("townCenterSellInterval", town_center_sell_interval);
                u32_config!("farmHandCostMult", farm_hand_cost_multiplier);
                if let Some(n) = request["configuration"]["maxMarketOrdersPerTurn"].as_u64() { config.max_market_orders_per_turn = usize::try_from(n)?; }
                if let Some(n) = request["configuration"]["weedSpawnChance"].as_f64() { config.weed_spawn_chance = n; }
                if let Some(flag) = request["configuration"]["decoupledRng"].as_bool() { config.decoupled_rng = flag; }
                state = Some(new_game(seed, config));
                snapshot(state.as_ref().unwrap())
            }
            Some("step") => {
                let current = state.as_mut().context("step before start")?;
                let actions = request["actions"].as_array().context("step.actions must be an array")?;
                anyhow::ensure!(actions.len() == 2, "expected two player actions");
                let parsed = kaggriculture_trace_policy::parse_trace(&serde_json::to_string(actions)?)
                    .map_err(anyhow::Error::msg)?;
                let joint = [parsed[0].clone(), parsed[1].clone()];
                step(current, &joint);
                snapshot(current)
            }
            other => anyhow::bail!("unknown command {other:?}"),
        };
        serde_json::to_writer(&mut stdout, &response)?;
        stdout.write_all(b"\n")?;
        stdout.flush()?;
    }
    Ok(())
}

