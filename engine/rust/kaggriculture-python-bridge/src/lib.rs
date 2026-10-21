//! Bridge a Python Kaggle-style Kaggriculture agent into the Rust engine.
//!
//! [`observation_value`] rebuilds the exact observation dictionary the
//! official `kaggle_environments` interpreter exposes (schema 1.32.7), so an
//! unmodified public-notebook `main.py` can play inside the fast Rust league.
//! [`PythonAgent`] hosts one such agent in a `worker.py` subprocess speaking
//! JSON lines. Serializer fidelity is proven by `verify_obs_parity` in the
//! consumer crate: a Python twin of a local policy must emit byte-identical
//! actions to its Rust port on live league states.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use anyhow::Context;
use kaggriculture_engine::model::{
    Animal, Crop, Farm, GameState, PlayerAction, PrivateState, Product, Shop, Tile,
};
use serde_json::{Map, Value, json};

const PRODUCT_NAMES: [&str; 9] = [
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
const ANIMAL_NAMES: [&str; 3] = ["GOOSE", "COW", "SHEEP"];
const CROP_NAMES: [&str; 5] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"];

fn item_name(index: usize) -> &'static str {
    if index < PRODUCT_NAMES.len() {
        PRODUCT_NAMES[index]
    } else {
        ANIMAL_NAMES[index - PRODUCT_NAMES.len()]
    }
}

fn crop_name(crop: Crop) -> &'static str {
    CROP_NAMES[crop.index()]
}

fn animal_name(animal: Animal) -> &'static str {
    ANIMAL_NAMES[animal.index()]
}

const fn shop_name(shop: Shop) -> &'static str {
    match shop {
        Shop::Bakery => "BAKERY",
        Shop::PizzaShop => "PIZZA_SHOP",
        Shop::BrunchSpot => "BRUNCH_SPOT",
        Shop::YarnStore => "YARN_STORE",
        Shop::IceCreamShop => "ICE_CREAM_SHOP",
        Shop::PetCafe => "PET_CAFE",
        Shop::SmoothieShop => "SMOOTHIE_SHOP",
        Shop::FarmersMarket => "FARMERS_MARKET",
    }
}

/// Integer-valued floats print as ints in Python; keep that shape.
fn number(value: f64) -> Value {
    if value.fract() == 0.0 && value.abs() < 9.0e15 {
        json!(value as i64)
    } else {
        json!(value)
    }
}

fn tile_value(tile: &Tile) -> Value {
    match tile {
        Tile::Empty => Value::Null,
        Tile::Locked => json!("LOCKED"),
        Tile::Weed => json!({"kind": "WEED"}),
        Tile::Plant(plant) => json!({
            "kind": "PLANT",
            "crop": crop_name(plant.crop),
            "planted_day": plant.planted_day,
            "watered_today": plant.watered_today,
            "consecutive_unwatered": plant.consecutive_unwatered,
            "yield_units": plant.yield_units,
            "max_lifespan_step": plant.max_lifespan_step,
            "fertilized_until_day": plant.fertilized_until_day,
        }),
        Tile::Structure(structure) => {
            let kind = match structure.kind {
                kaggriculture_engine::model::StructureKind::Coop => "COOP",
                kaggriculture_engine::model::StructureKind::Pasture => "PASTURE",
            };
            match &structure.animal {
                None => json!({"kind": kind}),
                Some(animal) => json!({
                    "kind": kind,
                    "animal": animal_name(animal.animal),
                    "placed_day": animal.placed_day,
                    "yield_units": animal.yield_units,
                    "consecutive_unfed": animal.consecutive_unfed,
                    "fed_today": animal.fed_today,
                    "cared_today": animal.cared_today,
                    "fertilizer_available": animal.fertilizer_available,
                    "pending_care_bonus": animal.pending_care_bonus,
                }),
            }
        }
    }
}

fn farm_value(farm: &Farm, board_size: usize) -> Value {
    let tiles: Vec<Value> = (0..board_size)
        .map(|y| {
            Value::Array(
                (0..board_size)
                    .map(|x| tile_value(&farm.tiles[y * board_size + x]))
                    .collect(),
            )
        })
        .collect();
    let quadrants: Vec<Value> = [
        (kaggriculture_engine::init::Quadrant::NorthWest.mask(), "NW"),
        (kaggriculture_engine::init::Quadrant::NorthEast.mask(), "NE"),
        (kaggriculture_engine::init::Quadrant::SouthWest.mask(), "SW"),
        (kaggriculture_engine::init::Quadrant::SouthEast.mask(), "SE"),
    ]
    .iter()
    .filter(|(mask, _)| farm.unlocked_quadrants & mask != 0)
    .map(|(_, name)| json!(name))
    .collect();
    json!({
        "money": number(farm.money),
        "tiles": tiles,
        "farmer": [farm.farmer.x, farm.farmer.y],
        "hands": farm.hands.iter().map(|p| json!([p.x, p.y])).collect::<Vec<_>>(),
        "unlocked_quadrants": quadrants,
        "hires_today": farm.hires_today,
    })
}

fn private_value(private: &PrivateState) -> Value {
    let mut shed = Map::new();
    for (index, count) in private.shed.iter().enumerate() {
        shed.insert(item_name(index).to_owned(), json!(count));
    }
    let mut seeds = Map::new();
    for (index, count) in private.seeds.iter().enumerate() {
        seeds.insert(CROP_NAMES[index].to_owned(), json!(count));
    }
    // Python inventories are dicts holding only positive counts, in insertion
    // order; `inventory_orders` preserves that order on the Rust side.
    let inventories: Vec<Value> = private
        .inventories
        .iter()
        .enumerate()
        .map(|(unit, counts)| {
            let mut inventory = Map::new();
            if let Some(order) = private.inventory_orders.get(unit) {
                for item in order {
                    let count = counts[item.index()];
                    if count > 0 {
                        inventory.insert(item_name(item.index()).to_owned(), json!(count));
                    }
                }
            }
            for (index, count) in counts.iter().enumerate() {
                if *count > 0 && !inventory.contains_key(item_name(index)) {
                    inventory.insert(item_name(index).to_owned(), json!(count));
                }
            }
            Value::Object(inventory)
        })
        .collect();
    json!({"shed": shed, "seeds": seeds, "inventories": inventories})
}

/// The observation the official interpreter would hand to `player` at the
/// current pre-action point of `state`.
#[must_use]
pub fn observation_value(state: &GameState, player: usize) -> Value {
    let board_size = state.config.board_size as usize;
    let mut market = Map::new();
    let mut inventory = Map::new();
    let mut prices = Map::new();
    for index in 0..PRODUCT_NAMES.len() {
        inventory.insert(
            PRODUCT_NAMES[index].to_owned(),
            json!(state.market.inventory[index]),
        );
        prices.insert(
            PRODUCT_NAMES[index].to_owned(),
            number(state.market.prices[index]),
        );
    }
    market.insert("inventory".to_owned(), Value::Object(inventory));
    market.insert("prices".to_owned(), Value::Object(prices));
    json!({
        "step": state.step,
        "day": state.day,
        "hour": state.hour,
        "player": player,
        "remainingOverageTime": 60,
        "farms": state
            .farms
            .iter()
            .map(|farm| farm_value(farm, board_size))
            .collect::<Vec<_>>(),
        "market": Value::Object(market),
        "town": json!({
            "unlocked_shops": state
                .town
                .unlocked_shops
                .iter()
                .map(|shop| json!(shop_name(*shop)))
                .collect::<Vec<_>>(),
        }),
        "private": private_value(&state.private[player]),
    })
}

fn product_name_of(product: Product) -> &'static str {
    PRODUCT_NAMES[product.index()]
}

/// One Python agent hosted in a worker subprocess, one JSON line per turn.
pub struct PythonAgent {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl PythonAgent {
    pub fn spawn(python: &Path, worker: &Path, agent_file: &Path) -> anyhow::Result<Self> {
        let mut child = Command::new(python)
            .arg(worker)
            .arg(agent_file)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("spawn {} {}", python.display(), worker.display()))?;
        let stdin = child.stdin.take().context("worker stdin")?;
        let stdout = BufReader::new(child.stdout.take().context("worker stdout")?);
        Ok(Self {
            child,
            stdin,
            stdout,
        })
    }

    pub fn act(&mut self, state: &GameState, player: usize) -> anyhow::Result<PlayerAction> {
        let observation = observation_value(state, player);
        serde_json::to_writer(&mut self.stdin, &observation)?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        let mut line = String::new();
        self.stdout.read_line(&mut line)?;
        anyhow::ensure!(!line.trim().is_empty(), "python worker closed its stdout");
        kaggriculture_trace_policy::parse_trace(&format!("[{}]", line.trim()))
            .map_err(anyhow::Error::msg)?
            .pop()
            .context("worker returned an empty action list")
    }
}

impl Drop for PythonAgent {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kaggriculture_engine::new_game;

    #[test]
    fn initial_observation_matches_python_shapes() {
        let state = new_game(7, Default::default());
        let obs = observation_value(&state, 1);
        assert_eq!(obs["step"], 0);
        assert_eq!(obs["player"], 1);
        assert_eq!(obs["farms"].as_array().unwrap().len(), 2);
        let farm = &obs["farms"][0];
        assert_eq!(farm["money"], 3000);
        assert_eq!(farm["farmer"], json!([4, 4]));
        assert_eq!(farm["unlocked_quadrants"], json!(["NW"]));
        // NW quadrant empty (null), everything else locked.
        assert_eq!(farm["tiles"][0][0], Value::Null);
        assert_eq!(farm["tiles"][0][9], json!("LOCKED"));
        assert_eq!(farm["tiles"][9][9], json!("LOCKED"));
        assert_eq!(obs["private"]["shed"]["COW"], 0);
        assert_eq!(obs["private"]["seeds"]["MELON"], 0);
        assert_eq!(obs["private"]["inventories"], json!([{}]));
        assert!(obs["market"]["prices"]["WHEAT"].is_i64());
        assert_eq!(obs["town"]["unlocked_shops"], json!([]));
        let _ = product_name_of(Product::Fertilizer);
    }
}
