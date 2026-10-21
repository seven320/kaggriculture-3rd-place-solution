// Derived from kaggle-environments 1.32.7, kaggle_environments/envs/kaggriculture/kaggriculture.py
// (published by Kaggle under the Apache License 2.0; see LICENSES/Apache-2.0.txt and NOTICE at the repository root):
// game data model (state fields and enumerations) ported from the Python interpreter and modified (translated to Rust, restructured for batched rollouts).
use serde::{Deserialize, Serialize};

pub const PLAYER_COUNT: usize = 2;
pub const CROP_COUNT: usize = 5;
pub const PRODUCT_COUNT: usize = 9;
pub const ANIMAL_COUNT: usize = 3;
pub const ITEM_COUNT: usize = PRODUCT_COUNT + ANIMAL_COUNT;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Crop {
    #[default]
    Wheat,
    Carrot,
    Tomato,
    Strawberry,
    Melon,
}

impl Crop {
    pub const ALL: [Self; CROP_COUNT] = [
        Self::Wheat,
        Self::Carrot,
        Self::Tomato,
        Self::Strawberry,
        Self::Melon,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Product {
    #[default]
    Wheat,
    Carrot,
    Tomato,
    Strawberry,
    Melon,
    Egg,
    Milk,
    Wool,
    Fertilizer,
}

impl Product {
    pub const ALL: [Self; PRODUCT_COUNT] = [
        Self::Wheat,
        Self::Carrot,
        Self::Tomato,
        Self::Strawberry,
        Self::Melon,
        Self::Egg,
        Self::Milk,
        Self::Wool,
        Self::Fertilizer,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Animal {
    #[default]
    Goose,
    Cow,
    Sheep,
}

impl Animal {
    pub const ALL: [Self; ANIMAL_COUNT] = [Self::Goose, Self::Cow, Self::Sheep];

    pub const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Item {
    Product(Product),
    Animal(Animal),
}

impl Item {
    pub const fn index(self) -> usize {
        match self {
            Self::Product(product) => product.index(),
            Self::Animal(animal) => PRODUCT_COUNT + animal.index(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Position {
    pub x: u16,
    pub y: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StructureKind {
    Coop,
    Pasture,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlantState {
    pub crop: Crop,
    pub planted_day: u32,
    pub watered_today: bool,
    pub consecutive_unwatered: u32,
    pub yield_units: u32,
    pub max_lifespan_step: i32,
    pub fertilized_until_day: i32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnimalState {
    pub animal: Animal,
    pub placed_day: u32,
    pub yield_units: u32,
    pub consecutive_unfed: u32,
    pub fed_today: bool,
    pub cared_today: bool,
    pub fertilizer_available: bool,
    pub pending_care_bonus: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructureState {
    pub kind: StructureKind,
    pub animal: Option<AnimalState>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tile {
    #[default]
    Empty,
    Locked,
    Weed,
    Plant(PlantState),
    Structure(StructureState),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Farm {
    pub money: f64,
    pub tiles: Vec<Tile>,
    pub farmer: Position,
    pub hands: Vec<Position>,
    pub unlocked_quadrants: u8,
    pub hires_today: u32,
    /// Engine cache, not game state: the lowest step at which a plant on this
    /// farm can still decay.  `decay_plants` skips its scan below it and
    /// recomputes it from the surviving plants on every scan, so it never
    /// outlives its evidence.  `0` means "unknown, scan".
    ///
    /// It lives inside `Farm` so that it can never be separated from the board
    /// it describes: replacing a whole farm (`rollout.rs` rebuilding a state
    /// from an observation) replaces the mark with it, cloning a farm carries
    /// the matching mark along, and `#[serde(skip)]` makes a deserialized farm
    /// start from a scan.  Excluded from `PartialEq` below: it is a cache, and
    /// two farms with the same board are the same farm whatever it holds.
    #[serde(skip)]
    pub decay_from: u32,
}

impl PartialEq for Farm {
    fn eq(&self, other: &Self) -> bool {
        self.money == other.money
            && self.tiles == other.tiles
            && self.farmer == other.farmer
            && self.hands == other.hands
            && self.unlocked_quadrants == other.unlocked_quadrants
            && self.hires_today == other.hires_today
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrivateState {
    pub shed: [u32; ITEM_COUNT],
    pub seeds: [u32; CROP_COUNT],
    pub inventories: Vec<[u32; ITEM_COUNT]>,
    /// Python dictionaries preserve insertion order. DROP and the end-of-day
    /// transfer consume each inventory in that order when the shed overflows.
    #[serde(default)]
    pub inventory_orders: Vec<Vec<Item>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum PriceShape {
    Linear,
    Square,
    SquareRoot,
    Log,
    Log10,
    Hinge,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MarketParam {
    pub base: f64,
    pub initial_inventory: i32,
    pub scale: f64,
    pub below_shape: PriceShape,
    pub below_target: f64,
    pub above_shape: PriceShape,
    pub above_target: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Market {
    pub inventory: [i32; PRODUCT_COUNT],
    /// Initial custom bases may be fractional; refreshed prices are integer-valued f64.
    pub prices: [f64; PRODUCT_COUNT],
    pub params: [MarketParam; PRODUCT_COUNT],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Shop {
    Bakery,
    PizzaShop,
    BrunchSpot,
    YarnStore,
    IceCreamShop,
    PetCafe,
    SmoothieShop,
    FarmersMarket,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Town {
    pub unlocked_shops: Vec<Shop>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub episode_steps: u32,
    pub board_size: u16,
    pub starting_money: u32,
    pub max_market_orders_per_turn: usize,
    pub turns_per_day: u32,
    pub shed_capacity: u32,
    pub weed_spawn_chance: f64,
    pub town_shop_unlock_interval: u32,
    pub town_shop_sell_interval: u32,
    pub town_center_sell_interval: u32,
    pub farm_hand_cost_multiplier: u32,
    pub market_params: [MarketParam; PRODUCT_COUNT],
    /// Local-audit only (never the official engine): draw the end-of-day weed
    /// RNG once per tile regardless of occupancy, so that a player's tile
    /// count no longer re-rolls the weed placements and the town shop-unlock
    /// sequence for the rest of the game.  `false` (the default) is the
    /// official, bit-parity behaviour.
    #[serde(default)]
    pub decoupled_rng: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Status {
    #[default]
    Active,
    Done,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GameState {
    pub config: Config,
    pub seed: i64,
    pub step: u32,
    pub day: u32,
    pub hour: u32,
    pub farms: [Farm; PLAYER_COUNT],
    pub private: [PrivateState; PLAYER_COUNT],
    pub market: Market,
    pub town: Town,
    pub statuses: [Status; PLAYER_COUNT],
    pub rewards: [Option<f64>; PLAYER_COUNT],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    North,
    South,
    East,
    West,
    #[default]
    Pass,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnitAction {
    Move(Direction),
    #[default]
    Pass,
    Drop,
    Pickup(Item, u32),
    Plant(Crop),
    Water,
    Harvest,
    Fertilize,
    BuildCoop,
    BuildPasture,
    Dig,
    Place(Item, u32),
    Feed,
    CollectFertilizer,
    Care,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketOrder {
    BuySeed(Crop, u32),
    BuyProduct(Product, u32),
    BuyAnimal(Animal, u32),
    Sell(Item, u32),
    Hire,
    BuyLand,
}

#[derive(Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlayerAction {
    pub farmer: UnitAction,
    pub hands: Vec<UnitAction>,
    pub market: Vec<MarketOrder>,
}

impl Clone for PlayerAction {
    fn clone(&self) -> Self {
        Self {
            farmer: self.farmer,
            hands: self.hands.clone(),
            market: self.market.clone(),
        }
    }

    /// Reuses the existing `hands` / `market` allocations: a derived `Clone`
    /// allocates two fresh vectors on every call, and a replay agent rebuilds
    /// one `PlayerAction` per unit per step.
    fn clone_from(&mut self, source: &Self) {
        self.farmer = source.farmer;
        self.hands.clone_from(&source.hands);
        self.market.clone_from(&source.market);
    }
}

pub type JointAction = [PlayerAction; PLAYER_COUNT];

/// Realized effect of one market-order slot.
///
/// This is an evaluator/training diagnostic. It contains no hidden future
/// information and does not alter normal `step` semantics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MarketOrderTrace {
    pub step: u32,
    pub player: usize,
    pub order_index: usize,
    pub order: MarketOrder,
    pub money_delta: f64,
    pub shed_delta: [i64; ITEM_COUNT],
    pub seed_delta: [i64; CROP_COUNT],
    pub hands_delta: i64,
    pub quadrants_delta: i64,
}

/// Realized effect of one unit action before market processing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnitActionTrace {
    pub step: u32,
    pub player: usize,
    pub unit_index: usize,
    pub action: UnitAction,
    pub position_before: Option<Position>,
    pub position_after: Option<Position>,
    pub tile_before: Option<Tile>,
    pub tile_after: Option<Tile>,
    pub inventory_delta: [i64; ITEM_COUNT],
    pub seed_delta: [i64; CROP_COUNT],
    pub success: bool,
}
