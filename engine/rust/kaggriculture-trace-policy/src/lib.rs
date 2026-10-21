//! Shared parser, executor, and parity runners for public route snapshots.

use kaggriculture_engine::model::{
    Animal, Crop, Direction, GameState, Item, MarketOrder, PlayerAction, Product, UnitAction,
};
pub mod replay;
pub use replay::{RepairState, RouteReplayAgent, WEED_REPLAY_STEPS, action_words, trace_json};

use serde::Deserialize;
use serde_json::Value;
use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

#[derive(Deserialize)]
struct RawAction {
    farmer: Vec<Value>,
    #[serde(default)]
    hands: Vec<Vec<Value>>,
    #[serde(default)]
    market: Vec<Vec<Value>>,
}

fn word(value: &Value) -> Result<&str, String> {
    value
        .as_str()
        .ok_or_else(|| format!("expected string, got {value}"))
}

fn quantity(value: Option<&Value>) -> Result<u32, String> {
    value
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| format!("expected u32 quantity, got {value:?}"))
}

fn optional_unit_quantity(value: Option<&Value>) -> Result<u32, String> {
    value.map_or(Ok(1), |value| quantity(Some(value)))
}

fn crop(value: &Value) -> Result<Crop, String> {
    match word(value)? {
        "WHEAT" => Ok(Crop::Wheat),
        "CARROT" => Ok(Crop::Carrot),
        "TOMATO" => Ok(Crop::Tomato),
        "STRAWBERRY" => Ok(Crop::Strawberry),
        "MELON" => Ok(Crop::Melon),
        other => Err(format!("unknown crop {other}")),
    }
}

fn product(value: &Value) -> Result<Product, String> {
    match word(value)? {
        "WHEAT" => Ok(Product::Wheat),
        "CARROT" => Ok(Product::Carrot),
        "TOMATO" => Ok(Product::Tomato),
        "STRAWBERRY" => Ok(Product::Strawberry),
        "MELON" => Ok(Product::Melon),
        "EGG" => Ok(Product::Egg),
        "MILK" => Ok(Product::Milk),
        "WOOL" => Ok(Product::Wool),
        "FERTILIZER" => Ok(Product::Fertilizer),
        other => Err(format!("unknown product {other}")),
    }
}

fn animal(value: &Value) -> Result<Animal, String> {
    match word(value)? {
        "GOOSE" => Ok(Animal::Goose),
        "COW" => Ok(Animal::Cow),
        "SHEEP" => Ok(Animal::Sheep),
        other => Err(format!("unknown animal {other}")),
    }
}

fn item(value: &Value) -> Result<Item, String> {
    animal(value)
        .map(Item::Animal)
        .or_else(|_| product(value).map(Item::Product))
}

fn unit(values: &[Value]) -> Result<UnitAction, String> {
    let op = values.first().ok_or("empty unit action")?;
    match word(op)? {
        "NORTH" => Ok(UnitAction::Move(Direction::North)),
        "SOUTH" => Ok(UnitAction::Move(Direction::South)),
        "EAST" => Ok(UnitAction::Move(Direction::East)),
        "WEST" => Ok(UnitAction::Move(Direction::West)),
        "PASS" => Ok(UnitAction::Pass),
        "DROP" => Ok(UnitAction::Drop),
        "PICKUP" => Ok(UnitAction::Pickup(
            item(values.get(1).ok_or("PICKUP item missing")?)?,
            optional_unit_quantity(values.get(2))?,
        )),
        "PLANT" => Ok(UnitAction::Plant(crop(
            values.get(1).ok_or("PLANT crop missing")?,
        )?)),
        "WATER" => Ok(UnitAction::Water),
        "HARVEST" => Ok(UnitAction::Harvest),
        "FERTILIZE" => Ok(UnitAction::Fertilize),
        "BUILD_COOP" => Ok(UnitAction::BuildCoop),
        "BUILD_PASTURE" => Ok(UnitAction::BuildPasture),
        "DIG" => Ok(UnitAction::Dig),
        "PLACE" => Ok(UnitAction::Place(
            item(values.get(1).ok_or("PLACE item missing")?)?,
            optional_unit_quantity(values.get(2))?,
        )),
        "FEED" => Ok(UnitAction::Feed),
        "COLLECT_FERTILIZER" => Ok(UnitAction::CollectFertilizer),
        "CARE" => Ok(UnitAction::Care),
        other => Err(format!("unknown unit action {other}")),
    }
}

fn market(values: &[Value]) -> Result<MarketOrder, String> {
    let op = values.first().ok_or("empty market order")?;
    match word(op)? {
        "BUY_SEED" => Ok(MarketOrder::BuySeed(
            crop(values.get(1).ok_or("BUY_SEED crop missing")?)?,
            quantity(values.get(2))?,
        )),
        "BUY_PRODUCT" => Ok(MarketOrder::BuyProduct(
            product(values.get(1).ok_or("BUY_PRODUCT item missing")?)?,
            quantity(values.get(2))?,
        )),
        "BUY_ANIMAL" => Ok(MarketOrder::BuyAnimal(
            animal(values.get(1).ok_or("BUY_ANIMAL item missing")?)?,
            quantity(values.get(2))?,
        )),
        "SELL" => Ok(MarketOrder::Sell(
            item(values.get(1).ok_or("SELL item missing")?)?,
            quantity(values.get(2))?,
        )),
        "HIRE" => Ok(MarketOrder::Hire),
        "BUY_LAND" => Ok(MarketOrder::BuyLand),
        other => Err(format!("unknown market order {other}")),
    }
}

pub fn parse_trace(source: &str) -> Result<Vec<PlayerAction>, String> {
    let rows: Vec<RawAction> = serde_json::from_str(source).map_err(|error| error.to_string())?;
    rows.into_iter()
        .map(|row| {
            Ok(PlayerAction {
                farmer: unit(&row.farmer)?,
                hands: row
                    .hands
                    .iter()
                    .map(|action| unit(action))
                    .collect::<Result<_, _>>()?,
                market: row
                    .market
                    .iter()
                    .map(|order| market(order))
                    .collect::<Result<_, _>>()?,
            })
        })
        .collect()
}

/// Parse a trace containing only each turn's market-order list.
pub fn parse_market_trace(source: &str) -> Result<Vec<Vec<MarketOrder>>, String> {
    let rows: Vec<Vec<Vec<Value>>> =
        serde_json::from_str(source).map_err(|error| error.to_string())?;
    rows.into_iter()
        .map(|orders| {
            orders
                .iter()
                .map(|order| market(order))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect()
}

#[must_use]
pub fn trace_action(state: &GameState, player: usize, trace: &[PlayerAction]) -> PlayerAction {
    let mut action = PlayerAction::default();
    trace_action_into(state, player, trace, &mut action);
    action
}

/// `trace_action` into an existing action, reusing its two vector allocations
/// (`PlayerAction::clone_from`).  A replay rebuilds one action per step, so the
/// fresh vectors of `trace_action` are two allocations per step per player.
pub fn trace_action_into(
    state: &GameState,
    player: usize,
    trace: &[PlayerAction],
    action: &mut PlayerAction,
) {
    let Some(template) = trace.get(state.step as usize).or_else(|| trace.last()) else {
        *action = PlayerAction::default();
        return;
    };
    action.clone_from(template);
    // `resize` also truncates, so this is the same row length as before.
    action
        .hands
        .resize(state.farms[player].hands.len(), UnitAction::Pass);
}

type Agent = fn(&GameState, usize) -> PlayerAction;

#[derive(Deserialize)]
struct Fixture {
    schema: String,
    policy: String,
    seed: i64,
    step: u32,
    player: usize,
    state: GameState,
    expected: PlayerAction,
}

pub fn verify_fixture(path: &Path, policy_id: &str, agent: Agent) -> Result<(), Box<dyn Error>> {
    verify_fixture_mut(path, policy_id, agent)
}

pub fn verify_fixture_mut(
    path: &Path,
    policy_id: &str,
    mut agent: impl FnMut(&GameState, usize) -> PlayerAction,
) -> Result<(), Box<dyn Error>> {
    let mut checked = 0;
    for (line_index, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let fixture: Fixture = serde_json::from_str(&line?)?;
        if fixture.schema != "kaggriculture-policy-parity-v1" || fixture.policy != policy_id {
            return Err(format!("line {}: wrong schema or policy", line_index + 1).into());
        }
        let actual = agent(&fixture.state, fixture.player);
        if actual != fixture.expected {
            return Err(format!(
                "line {} seed={} step={} player={} action mismatch\nexpected={:?}\nactual={:?}",
                line_index + 1,
                fixture.seed,
                fixture.step,
                fixture.player,
                fixture.expected,
                actual
            )
            .into());
        }
        checked += 1;
    }
    println!("policy_parity=ok fixtures={checked}");
    Ok(())
}

#[derive(Deserialize)]
struct Episode {
    schema: String,
    policy: String,
    seed: i64,
    initial: GameState,
    joint_actions: Vec<[PlayerAction; 2]>,
    step_states: Vec<GameState>,
}

pub fn verify_e2e(path: &Path, policy_id: &str, agent: Agent) -> Result<(), Box<dyn Error>> {
    verify_e2e_stateful(path, policy_id, || agent)
}

pub fn verify_e2e_stateful<F, A>(
    path: &Path,
    policy_id: &str,
    factory: F,
) -> Result<(), Box<dyn Error>>
where
    F: Fn() -> A,
    A: FnMut(&GameState, usize) -> PlayerAction,
{
    use kaggriculture_engine::step;

    let mut episodes = 0;
    let mut transitions = 0;
    for (line_index, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let fixture: Episode = serde_json::from_str(&line?)?;
        if fixture.schema != "kaggriculture-policy-e2e-matrix-v1" || fixture.policy != policy_id {
            return Err(format!("line {}: wrong schema or policy", line_index + 1).into());
        }
        if fixture.step_states.len() != fixture.joint_actions.len() + 1
            || fixture.initial != fixture.step_states[0]
        {
            return Err(format!("line {}: state/action length mismatch", line_index + 1).into());
        }
        let mut rust_python = fixture.initial.clone();
        let mut rust_rust = fixture.initial.clone();
        let mut official_agents = [factory(), factory()];
        let mut rust_agents = [factory(), factory()];
        for (step_index, python_actions) in fixture.joint_actions.iter().enumerate() {
            let official = &fixture.step_states[step_index];
            if [
                official_agents[0](official, 0),
                official_agents[1](official, 1),
            ] != *python_actions
            {
                return Err(format!(
                    "seed={} transition={step_index}: official policy mismatch",
                    fixture.seed
                )
                .into());
            }
            let rust_actions = [rust_agents[0](&rust_rust, 0), rust_agents[1](&rust_rust, 1)];
            step(&mut rust_python, python_actions);
            step(&mut rust_rust, &rust_actions);
            let expected = &fixture.step_states[step_index + 1];
            if &rust_python != expected || &rust_rust != expected {
                return Err(format!(
                    "seed={} transition={step_index}: game path diverged",
                    fixture.seed
                )
                .into());
            }
            transitions += 1;
        }
        episodes += 1;
    }
    println!(
        "e2e_matrix=ok episodes={episodes} transitions={transitions} paths=official+python,official+rust,rust+python,rust+rust"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_action_family() {
        let source = r#"[{"farmer":["BUILD_PASTURE"],"hands":[["PICKUP","SHEEP",1],["PLACE","WHEAT",2],["DROP"]],"market":[["HIRE"],["BUY_LAND"],["BUY_ANIMAL","COW",2],["SELL","MILK",3]]}]"#;
        let trace = parse_trace(source).unwrap();
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].farmer, UnitAction::BuildPasture);
        assert_eq!(trace[0].hands.len(), 3);
        assert_eq!(trace[0].market.len(), 4);
    }
}
