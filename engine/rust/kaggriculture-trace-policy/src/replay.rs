//! Route replay with weed repair (moved here from `sol/evo_farm/open_ended`
//! on 2026-08-30 so the route-search crate can use it without the evolutionary
//! crate's policy dependencies), plus the inverse of `parse_trace`.
use std::collections::BTreeMap;

use kaggriculture_engine::model::{
    Animal, Crop, Direction, GameState, Item, MarketOrder, PlayerAction, Product, Tile, UnitAction,
};
use serde_json::{Value, json};


/// After a weed forced a `Dig`, replay the actor's own lane shifted back by
/// one step for this many steps so its personal schedule slides by one hour
/// instead of desynchronising.
pub const WEED_REPLAY_STEPS: u32 = 8;

/// Replays a fixed route (one `PlayerAction` per step); when a scheduled
/// `Plant` / `BuildPasture` / `BuildCoop` lands on a weed tile the actor digs
/// first and its lane is delayed by one step (boatlee-style repair).
#[derive(Clone)]
pub struct RouteReplayAgent<'a> {
    trace: &'a [PlayerAction],
    active: BTreeMap<usize, (u32, UnitAction)>,
}

/// Repair state carried across a snapshot (which actors are mid-repair).
pub type RepairState = BTreeMap<usize, (u32, UnitAction)>;

impl<'a> RouteReplayAgent<'a> {
    #[must_use]
    pub fn new(trace: &'a [PlayerAction]) -> Self {
        Self {
            trace,
            active: BTreeMap::new(),
        }
    }

    /// Resume replay of `trace` with a saved repair state (prefix snapshots).
    #[must_use]
    pub fn with_state(trace: &'a [PlayerAction], active: RepairState) -> Self {
        Self { trace, active }
    }

    #[must_use]
    pub fn trace(&self) -> &'a [PlayerAction] {
        self.trace
    }

    #[must_use]
    pub fn repair_state(&self) -> &RepairState {
        &self.active
    }

    pub fn act(&mut self, state: &GameState, player: usize) -> PlayerAction {
        let mut action = PlayerAction::default();
        self.act_into(state, player, &mut action);
        action
    }

    /// `act` into an existing action, reusing its allocations.  Same result as
    /// `act`; a replay calls it once per step, and `act` allocated four vectors
    /// per call (the row's two, plus the positions and unit lanes).
    pub fn act_into(&mut self, state: &GameState, player: usize, action: &mut PlayerAction) {
        crate::trace_action_into(state, player, self.trace, action);
        let farm = &state.farms[player];
        let board = state.config.board_size;
        let step = state.step;
        let trace = self.trace;
        let units = 1 + action.hands.len();
        self.active.retain(|actor, (start, intended)| {
            let actor = *actor;
            if actor >= units {
                return false;
            }
            let age = step.saturating_sub(*start);
            if age == 1 {
                *unit_mut(action, actor) = *intended;
                true
            } else if age >= 2 && age <= 1 + WEED_REPLAY_STEPS {
                *unit_mut(action, actor) = actor_trace_action(trace, step as usize - 1, actor);
                true
            } else {
                false
            }
        });
        for actor in 0..units {
            if self.active.contains_key(&actor) {
                continue;
            }
            let slot = unit_mut(action, actor);
            if !matches!(
                slot,
                UnitAction::Plant(_) | UnitAction::BuildPasture | UnitAction::BuildCoop
            ) {
                continue;
            }
            let intended = *slot;
            let position = if actor == 0 {
                farm.farmer
            } else {
                farm.hands[actor - 1]
            };
            let index = position.y as usize * board as usize + position.x as usize;
            if matches!(farm.tiles.get(index), Some(Tile::Weed)) {
                self.active.insert(actor, (step, intended));
                *unit_mut(action, actor) = UnitAction::Dig;
            }
        }
    }
}

/// Lane `actor` of a row: 0 is the farmer, `i` is hand `i - 1`.
fn unit_mut(action: &mut PlayerAction, actor: usize) -> &mut UnitAction {
    if actor == 0 {
        &mut action.farmer
    } else {
        &mut action.hands[actor - 1]
    }
}

fn actor_trace_action(trace: &[PlayerAction], step: usize, actor: usize) -> UnitAction {
    let template = &trace[step.min(trace.len() - 1)];
    if actor == 0 {
        template.farmer
    } else {
        template
            .hands
            .get(actor - 1)
            .copied()
            .unwrap_or(UnitAction::Pass)
    }
}

// ---- words: the inverse of `parse_trace` -----------------------------------

#[must_use]
pub fn crop_word(crop: Crop) -> &'static str {
    match crop {
        Crop::Wheat => "WHEAT",
        Crop::Carrot => "CARROT",
        Crop::Tomato => "TOMATO",
        Crop::Strawberry => "STRAWBERRY",
        Crop::Melon => "MELON",
    }
}

#[must_use]
pub fn product_word(product: Product) -> &'static str {
    match product {
        Product::Wheat => "WHEAT",
        Product::Carrot => "CARROT",
        Product::Tomato => "TOMATO",
        Product::Strawberry => "STRAWBERRY",
        Product::Melon => "MELON",
        Product::Egg => "EGG",
        Product::Milk => "MILK",
        Product::Wool => "WOOL",
        Product::Fertilizer => "FERTILIZER",
    }
}

#[must_use]
pub fn animal_word(animal: Animal) -> &'static str {
    match animal {
        Animal::Goose => "GOOSE",
        Animal::Cow => "COW",
        Animal::Sheep => "SHEEP",
    }
}

#[must_use]
pub fn item_word(item: Item) -> &'static str {
    match item {
        Item::Product(p) => product_word(p),
        Item::Animal(a) => animal_word(a),
    }
}

#[must_use]
pub fn unit_words(action: &UnitAction) -> Value {
    match action {
        UnitAction::Move(Direction::North) => json!(["NORTH"]),
        UnitAction::Move(Direction::South) => json!(["SOUTH"]),
        UnitAction::Move(Direction::East) => json!(["EAST"]),
        UnitAction::Move(Direction::West) => json!(["WEST"]),
        UnitAction::Move(Direction::Pass) | UnitAction::Pass => json!(["PASS"]),
        UnitAction::Drop => json!(["DROP"]),
        UnitAction::Pickup(item, qty) => json!(["PICKUP", item_word(*item), qty]),
        UnitAction::Plant(crop) => json!(["PLANT", crop_word(*crop)]),
        UnitAction::Water => json!(["WATER"]),
        UnitAction::Harvest => json!(["HARVEST"]),
        UnitAction::Fertilize => json!(["FERTILIZE"]),
        UnitAction::BuildCoop => json!(["BUILD_COOP"]),
        UnitAction::BuildPasture => json!(["BUILD_PASTURE"]),
        UnitAction::Dig => json!(["DIG"]),
        UnitAction::Place(item, qty) => json!(["PLACE", item_word(*item), qty]),
        UnitAction::Feed => json!(["FEED"]),
        UnitAction::CollectFertilizer => json!(["COLLECT_FERTILIZER"]),
        UnitAction::Care => json!(["CARE"]),
    }
}

#[must_use]
pub fn market_words(order: &MarketOrder) -> Value {
    match order {
        MarketOrder::BuySeed(crop, qty) => json!(["BUY_SEED", crop_word(*crop), qty]),
        MarketOrder::BuyProduct(product, qty) => {
            json!(["BUY_PRODUCT", product_word(*product), qty])
        }
        MarketOrder::BuyAnimal(animal, qty) => json!(["BUY_ANIMAL", animal_word(*animal), qty]),
        MarketOrder::Sell(item, qty) => json!(["SELL", item_word(*item), qty]),
        MarketOrder::Hire => json!(["HIRE"]),
        MarketOrder::BuyLand => json!(["BUY_LAND"]),
    }
}

/// One trace row in the `parse_trace` format.
#[must_use]
pub fn action_words(action: &PlayerAction) -> Value {
    json!({
        "farmer": unit_words(&action.farmer),
        "hands": action.hands.iter().map(unit_words).collect::<Vec<_>>(),
        "market": action.market.iter().map(market_words).collect::<Vec<_>>(),
    })
}

/// Serialise a whole trace; `parse_trace(&trace_json(&t)) == t`.
#[must_use]
pub fn trace_json(trace: &[PlayerAction]) -> String {
    let rows: Vec<Value> = trace.iter().map(action_words).collect();
    serde_json::to_string(&rows).expect("trace serialises")
}
