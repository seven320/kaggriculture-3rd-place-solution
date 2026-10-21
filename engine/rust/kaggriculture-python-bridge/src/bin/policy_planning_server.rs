//! Continue several real policy checkpoints from one observed public state.
//!
//! The hidden opponent state is generated from an explicitly independent seed
//! and a hypothetical joint action history.  Public farms, market, town, and
//! the acting player's private state are then replaced by the live observation.

use std::collections::HashMap;
use std::io::{self, BufRead, Write};

use anyhow::{bail, ensure, Context, Result};
use kaggriculture_engine::model::{
    Animal, AnimalState, Config, Crop, Farm, GameState, Item, PlayerAction, Position, PrivateState,
    Product, Shop, Status, StructureKind, StructureState, Tile,
};
use kaggriculture_engine::{new_game, step};
use kaggriculture_python_bridge::observation_value;
use serde_json::{json, Map, Value};

const FORBIDDEN: [&str; 5] = [
    "seed",
    "world_seed",
    "opponent_private",
    "internal_state",
    "current_state",
];
const PRODUCTS: [&str; 9] = [
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
];
const CROPS: [&str; 5] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"];
const ANIMALS: [&str; 3] = ["GOOSE", "COW", "SHEEP"];

fn reject_hidden(value: &Value) -> Result<()> {
    let object = value.as_object().context("request must be an object")?;
    for key in FORBIDDEN {
        ensure!(!object.contains_key(key), "forbidden hidden input: {key}");
    }
    Ok(())
}

fn index(names: &[&str], value: &str) -> Result<usize> {
    names
        .iter()
        .position(|item| *item == value)
        .with_context(|| format!("unknown enum value {value:?}"))
}

fn number(value: &Value) -> Result<f64> {
    value
        .as_f64()
        .or_else(|| value.as_i64().map(|item| item as f64))
        .context("number expected")
}

fn position(value: &Value) -> Result<Position> {
    let values = value.as_array().context("position must be an array")?;
    ensure!(values.len() == 2, "position must contain x and y");
    Ok(Position {
        x: u16::try_from(values[0].as_u64().context("position x")?)?,
        y: u16::try_from(values[1].as_u64().context("position y")?)?,
    })
}

fn item(name: &str) -> Result<Item> {
    if let Some(index) = PRODUCTS.iter().position(|value| *value == name) {
        return Ok(Item::Product(Product::ALL[index]));
    }
    if let Some(index) = ANIMALS.iter().position(|value| *value == name) {
        return Ok(Item::Animal(Animal::ALL[index]));
    }
    bail!("unknown item {name:?}")
}

fn item_name(value: Item) -> &'static str {
    match value {
        Item::Product(product) => PRODUCTS[product.index()],
        Item::Animal(animal) => ANIMALS[animal.index()],
    }
}

fn tile(value: &Value) -> Result<Tile> {
    if value.is_null() {
        return Ok(Tile::Empty);
    }
    if value.as_str() == Some("LOCKED") {
        return Ok(Tile::Locked);
    }
    let object = value.as_object().context("tile must be an object")?;
    match object
        .get("kind")
        .and_then(Value::as_str)
        .context("tile.kind")?
    {
        "WEED" => Ok(Tile::Weed),
        "PLANT" => Ok(Tile::Plant(kaggriculture_engine::model::PlantState {
            crop: Crop::ALL[index(&CROPS, object["crop"].as_str().context("plant.crop")?)?],
            planted_day: u32::try_from(object["planted_day"].as_u64().unwrap_or(0))?,
            watered_today: object["watered_today"].as_bool().unwrap_or(false),
            consecutive_unwatered: u32::try_from(
                object["consecutive_unwatered"].as_u64().unwrap_or(0),
            )?,
            yield_units: u32::try_from(object["yield_units"].as_u64().unwrap_or(0))?,
            max_lifespan_step: i32::try_from(object["max_lifespan_step"].as_i64().unwrap_or(0))?,
            fertilized_until_day: i32::try_from(
                object["fertilized_until_day"].as_i64().unwrap_or(-1),
            )?,
        })),
        kind @ ("COOP" | "PASTURE") => {
            let animal_state = match object.get("animal").and_then(Value::as_str) {
                Some(name) => {
                    let animal = Animal::ALL[index(&ANIMALS, name)?];
                    Some(AnimalState {
                        animal,
                        placed_day: u32::try_from(object["placed_day"].as_u64().unwrap_or(0))?,
                        yield_units: u32::try_from(object["yield_units"].as_u64().unwrap_or(0))?,
                        consecutive_unfed: u32::try_from(
                            object["consecutive_unfed"].as_u64().unwrap_or(0),
                        )?,
                        fed_today: object["fed_today"].as_bool().unwrap_or(false),
                        cared_today: object["cared_today"].as_bool().unwrap_or(false),
                        fertilizer_available: object["fertilizer_available"]
                            .as_bool()
                            .unwrap_or(false),
                        pending_care_bonus: u32::try_from(
                            object["pending_care_bonus"].as_u64().unwrap_or(0),
                        )?,
                    })
                }
                None => None,
            };
            Ok(Tile::Structure(StructureState {
                kind: if kind == "COOP" {
                    StructureKind::Coop
                } else {
                    StructureKind::Pasture
                },
                animal: animal_state,
            }))
        }
        kind => bail!("unknown tile kind {kind:?}"),
    }
}

fn farm(value: &Value, board_size: usize) -> Result<Farm> {
    let object = value.as_object().context("farm must be an object")?;
    let rows = object["tiles"].as_array().context("farm.tiles")?;
    ensure!(
        rows.len() == board_size,
        "farm row count differs from boardSize"
    );
    let mut tiles = Vec::with_capacity(board_size * board_size);
    for row in rows {
        let values = row.as_array().context("tile row")?;
        ensure!(
            values.len() == board_size,
            "farm column count differs from boardSize"
        );
        for value in values {
            tiles.push(tile(value)?);
        }
    }
    let unlocked_quadrants = object["unlocked_quadrants"]
        .as_array()
        .map(|values| {
            values.iter().fold(0_u8, |mask, value| {
                mask | match value.as_str().unwrap_or("") {
                    "NW" => 1,
                    "NE" => 2,
                    "SW" => 4,
                    "SE" => 8,
                    _ => 0,
                }
            })
        })
        .unwrap_or(1);
    Ok(Farm {
        money: number(&object["money"])?,
        tiles,
        farmer: position(&object["farmer"])?,
        hands: object["hands"]
            .as_array()
            .context("farm.hands")?
            .iter()
            .map(position)
            .collect::<Result<_>>()?,
        unlocked_quadrants,
        hires_today: u32::try_from(object["hires_today"].as_u64().unwrap_or(0))?,
        // This is a derived scan cache. Zero means unknown and is the correct
        // value after rebuilding a farm from an observation.
        decay_from: 0,
    })
}

fn private(value: &Value, inventory_orders: &Value) -> Result<PrivateState> {
    let object = value.as_object().context("private must be an object")?;
    let mut shed = [0; 12];
    let mut seeds = [0; 5];
    if let Some(values) = object["shed"].as_object() {
        for (name, value) in values {
            shed[item(name)?.index()] = u32::try_from(value.as_u64().unwrap_or(0))?;
        }
    }
    if let Some(values) = object["seeds"].as_object() {
        for (name, value) in values {
            seeds[index(&CROPS, name)?] = u32::try_from(value.as_u64().unwrap_or(0))?;
        }
    }
    let inventory_values = object["inventories"]
        .as_array()
        .context("private.inventories")?;
    let explicit_orders = inventory_orders
        .as_array()
        .context("own_inventory_orders must be an array")?;
    ensure!(
        explicit_orders.len() == inventory_values.len(),
        "inventory order count mismatch"
    );
    let mut inventories = Vec::with_capacity(inventory_values.len());
    let mut orders = Vec::with_capacity(inventory_values.len());
    for (inventory, order) in inventory_values.iter().zip(explicit_orders) {
        let values = inventory
            .as_object()
            .context("inventory must be an object")?;
        let mut counts = [0; 12];
        for (name, value) in values {
            counts[item(name)?.index()] = u32::try_from(value.as_u64().unwrap_or(0))?;
        }
        let mut parsed_order = Vec::new();
        for name in order
            .as_array()
            .context("inventory order must be an array")?
        {
            let parsed = item(name.as_str().context("inventory order item")?)?;
            if counts[parsed.index()] > 0 && !parsed_order.contains(&parsed) {
                parsed_order.push(parsed);
            }
        }
        ensure!(
            parsed_order.len()
                == values
                    .values()
                    .filter(|value| value.as_u64().unwrap_or(0) > 0)
                    .count(),
            "inventory order omitted a positive item"
        );
        inventories.push(counts);
        orders.push(parsed_order);
    }
    Ok(PrivateState {
        shed,
        seeds,
        inventories,
        inventory_orders: orders,
    })
}

fn configured(request: &Value) -> Result<Config> {
    let value = request.get("configuration").unwrap_or(&Value::Null);
    let mut config = Config::default();
    if let Some(number) = value["episodeSteps"].as_u64() {
        config.episode_steps = u32::try_from(number)?;
    }
    if let Some(number) = value["boardSize"].as_u64() {
        config.board_size = u16::try_from(number)?;
    }
    macro_rules! u32_config {
        ($json:literal, $field:ident) => {
            if let Some(number) = value[$json].as_u64() {
                config.$field = u32::try_from(number)?;
            }
        };
    }
    u32_config!("startingMoney", starting_money);
    u32_config!("turnsPerDay", turns_per_day);
    u32_config!("shedCapacity", shed_capacity);
    u32_config!("townShopUnlockInterval", town_shop_unlock_interval);
    u32_config!("townShopSellInterval", town_shop_sell_interval);
    u32_config!("townCenterSellInterval", town_center_sell_interval);
    u32_config!("farmHandCostMult", farm_hand_cost_multiplier);
    if let Some(number) = value["maxMarketOrdersPerTurn"].as_u64() {
        config.max_market_orders_per_turn = usize::try_from(number)?;
    }
    if let Some(number) = value["weedSpawnChance"].as_f64() {
        config.weed_spawn_chance = number;
    }
    if let Some(flag) = value["decoupledRng"].as_bool() {
        config.decoupled_rng = flag;
    }
    Ok(config)
}

fn parse_action(value: &Value) -> Result<PlayerAction> {
    serde_json::from_value(value.clone()).context("invalid Rust action")
}

fn reconstruct(request: &Value) -> Result<(GameState, usize, Value)> {
    reject_hidden(request)?;
    let observation = request.get("observation").context("observation required")?;
    reject_hidden(observation)?;
    let player = usize::try_from(observation["player"].as_u64().unwrap_or(0))?;
    ensure!(player < 2, "player must be 0 or 1");
    let simulation_seed = request["simulation_seed"]
        .as_i64()
        .context("simulation_seed must be independent")?;
    let config = configured(request)?;
    let turns_per_day = config.turns_per_day.max(1);
    let board_size = usize::from(config.board_size);
    let mut state = new_game(simulation_seed, config);

    let history = request["hypothetical_history"]
        .as_array()
        .context("hypothetical_history must be an array")?;
    let observation_step = u32::try_from(observation["step"].as_u64().unwrap_or(0))?;
    ensure!(
        history.len() == observation_step as usize,
        "history length must equal observation.step"
    );
    for value in history {
        let actions = value
            .as_array()
            .context("history entry must be a joint action")?;
        ensure!(actions.len() == 2, "history entry must contain two actions");
        step(
            &mut state,
            &[parse_action(&actions[0])?, parse_action(&actions[1])?],
        );
    }
    ensure!(
        state.step == observation_step,
        "hypothetical replay ended at wrong step"
    );

    let farms = observation["farms"]
        .as_array()
        .context("observation.farms")?;
    ensure!(farms.len() == 2, "observation must contain two farms");
    state.farms = [farm(&farms[0], board_size)?, farm(&farms[1], board_size)?];
    state.private[player] = private(
        &observation["private"],
        request
            .get("own_inventory_orders")
            .context("own_inventory_orders required")?,
    )?;
    ensure!(
        state.private[player].inventories.len() == state.farms[player].hands.len() + 1,
        "own inventory count must match farmer plus hands"
    );
    let opponent = 1 - player;
    let opponent_inventories_before = state.private[opponent].inventories.len();
    let opponent_inventories_required = state.farms[opponent].hands.len() + 1;
    state.private[opponent]
        .inventories
        .resize(opponent_inventories_required, [0; 12]);
    state.private[opponent]
        .inventory_orders
        .resize(opponent_inventories_required, Vec::new());
    if let Some(values) = observation["market"]["inventory"].as_object() {
        for (name, value) in values {
            state.market.inventory[index(&PRODUCTS, name)?] =
                i32::try_from(value.as_i64().context("market inventory")?)?;
        }
    }
    if let Some(values) = observation["market"]["prices"].as_object() {
        for (name, value) in values {
            state.market.prices[index(&PRODUCTS, name)?] = number(value)?;
        }
    }
    state.town.unlocked_shops = observation["town"]["unlocked_shops"]
        .as_array()
        .context("town.unlocked_shops")?
        .iter()
        .map(|value| match value.as_str().unwrap_or("") {
            "BAKERY" => Ok(Shop::Bakery),
            "PIZZA_SHOP" => Ok(Shop::PizzaShop),
            "BRUNCH_SPOT" => Ok(Shop::BrunchSpot),
            "YARN_STORE" => Ok(Shop::YarnStore),
            "ICE_CREAM_SHOP" => Ok(Shop::IceCreamShop),
            "PET_CAFE" => Ok(Shop::PetCafe),
            "SMOOTHIE_SHOP" => Ok(Shop::SmoothieShop),
            "FARMERS_MARKET" => Ok(Shop::FarmersMarket),
            name => bail!("unknown shop {name:?}"),
        })
        .collect::<Result<_>>()?;
    state.step = observation_step;
    state.day = u32::try_from(
        observation["day"]
            .as_u64()
            .unwrap_or(u64::from(observation_step / turns_per_day)),
    )?;
    state.hour = u32::try_from(
        observation["hour"]
            .as_u64()
            .unwrap_or(u64::from(observation_step % turns_per_day)),
    )?;
    ensure!(
        state.day == state.step / turns_per_day,
        "observation day disagrees with step"
    );
    ensure!(
        state.hour == state.step % turns_per_day,
        "observation hour disagrees with step"
    );
    state.statuses = [Status::Active; 2];
    state.rewards = [None; 2];
    let estimation_adjustments = json!({
        "opponent_inventory_slots_before": opponent_inventories_before,
        "opponent_inventory_slots_after": opponent_inventories_required,
        "padded_empty_slots": opponent_inventories_required.saturating_sub(opponent_inventories_before),
        "truncated_slots": opponent_inventories_before.saturating_sub(opponent_inventories_required),
        "reason": "align independent-seed opponent private slots to live public worker count",
    });
    Ok((state, player, estimation_adjustments))
}

struct Session {
    player: usize,
    candidates: HashMap<String, GameState>,
    replayed_steps: usize,
    estimation_adjustments: Value,
}

fn inventory_orders(state: &GameState, player: usize) -> Value {
    Value::Array(
        state.private[player]
            .inventory_orders
            .iter()
            .map(|order| Value::Array(order.iter().map(|value| json!(item_name(*value))).collect()))
            .collect(),
    )
}

fn snapshot(session: &Session) -> Value {
    let observations = session
        .candidates
        .iter()
        .map(|(name, state)| (name.clone(), observation_value(state, session.player)))
        .collect::<Map<_, _>>();
    let opponent_observations = session
        .candidates
        .iter()
        .map(|(name, state)| (name.clone(), observation_value(state, 1 - session.player)))
        .collect::<Map<_, _>>();
    let own_inventory_orders = session
        .candidates
        .iter()
        .map(|(name, state)| (name.clone(), inventory_orders(state, session.player)))
        .collect::<Map<_, _>>();
    let opponent_inventory_orders = session
        .candidates
        .iter()
        .map(|(name, state)| (name.clone(), inventory_orders(state, 1 - session.player)))
        .collect::<Map<_, _>>();
    let scores = session
        .candidates
        .iter()
        .map(|(name, state)| {
            (
                name.clone(),
                json!({
                    "own_money": state.farms[session.player].money,
                    "rival_money": state.farms[1 - session.player].money,
                    "margin": state.farms[session.player].money - state.farms[1 - session.player].money,
                    "step": state.step,
                    "done": state.statuses.iter().all(|status| *status == Status::Done),
                }),
            )
        })
        .collect::<Map<_, _>>();
    json!({
        "observations": observations,
        "opponent_observations": opponent_observations,
        "own_inventory_orders": own_inventory_orders,
        "opponent_inventory_orders": opponent_inventory_orders,
        "scores": scores,
        "candidate_states": session.candidates.len(),
        "state_isolated": true,
        "replayed_hypothetical_steps": session.replayed_steps,
        "opponent_private_source": "independent_seed_hypothetical_history",
        "estimation_adjustments": session.estimation_adjustments,
    })
}

fn step_session(session: &mut Session, request: &Value) -> Result<Value> {
    reject_hidden(request)?;
    let actions = request["actions"].as_object().context("actions map")?;
    let opponent_actions = request["opponent_actions"]
        .as_object()
        .context("opponent_actions map")?;
    for (name, state) in &mut session.candidates {
        let own = parse_action(actions.get(name).context("missing candidate action")?)?;
        let opponent = parse_action(
            opponent_actions
                .get(name)
                .context("missing candidate-specific opponent action")?,
        )?;
        let joint = if session.player == 0 {
            [own, opponent]
        } else {
            [opponent, own]
        };
        step(state, &joint);
    }
    Ok(snapshot(session))
}

fn main() -> Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    let mut session: Option<Session> = None;
    for line in stdin.lock().lines() {
        let request: Value = serde_json::from_str(&line?)?;
        let response = match request["command"].as_str() {
            Some("start") => {
                let (state, player, estimation_adjustments) = reconstruct(&request)?;
                let names = request["candidates"].as_array().context("candidates")?;
                ensure!(!names.is_empty(), "at least one candidate required");
                let candidates = names
                    .iter()
                    .map(|value| {
                        let name = value.as_str().context("candidate name")?;
                        Ok((name.to_owned(), state.clone()))
                    })
                    .collect::<Result<HashMap<_, _>>>()?;
                let replayed_steps = state.step as usize;
                session = Some(Session {
                    player,
                    candidates,
                    replayed_steps,
                    estimation_adjustments,
                });
                snapshot(session.as_ref().unwrap())
            }
            Some("step") => step_session(
                session.as_mut().context("start must precede step")?,
                &request,
            )?,
            other => bail!("unknown command {other:?}"),
        };
        serde_json::to_writer(&mut stdout, &response)?;
        stdout.write_all(b"\n")?;
        stdout.flush()?;
    }
    Ok(())
}
