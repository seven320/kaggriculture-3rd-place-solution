// Derived from kaggle-environments 1.32.7, kaggle_environments/envs/kaggriculture/kaggriculture.py
// (published by Kaggle under the Apache License 2.0; see LICENSES/Apache-2.0.txt and NOTICE at the repository root):
// game transition rules ported from the Python interpreter and modified (translated to Rust, restructured for batched rollouts).
//! Game transition rules ported from the official Python interpreter.
//!
//! The public [`step`] function intentionally accepts typed actions. JSON/schema
//! validation, replay construction, and agent callbacks belong outside this
//! research kernel.

use crate::model::{
    ANIMAL_COUNT, Animal, AnimalState, CROP_COUNT, Crop, Direction, Farm, GameState, ITEM_COUNT,
    Item, JointAction, MarketOrder, MarketOrderTrace, PLAYER_COUNT, PRODUCT_COUNT, PlantState,
    PriceShape, PrivateState, Product, Shop, Status, StructureKind, StructureState, Tile,
    UnitAction, UnitActionTrace,
};

const PRICE_FLOOR: f64 = 1.0;
const HINGE_GAIN: f64 = 8.0;
const LAND_PRICES: [u32; 3] = [1_000, 2_000, 4_000];
const MAX_SHOP_INSTANCES: usize = 8;

#[derive(Clone, Copy)]
struct CropData {
    seed: u32,
    first_yield_day: u32,
    max_yield_day: u32,
    interval: u32,
    max_yield: u32,
    ongoing: bool,
}

const CROPS: [CropData; CROP_COUNT] = [
    CropData {
        seed: 10,
        first_yield_day: 2,
        max_yield_day: 4,
        interval: 0,
        max_yield: 6,
        ongoing: false,
    },
    CropData {
        seed: 20,
        first_yield_day: 2,
        max_yield_day: 3,
        interval: 0,
        max_yield: 4,
        ongoing: false,
    },
    CropData {
        seed: 50,
        first_yield_day: 8,
        max_yield_day: 8,
        interval: 1,
        max_yield: 4,
        ongoing: true,
    },
    CropData {
        seed: 100,
        first_yield_day: 10,
        max_yield_day: 10,
        interval: 2,
        max_yield: 4,
        ongoing: true,
    },
    CropData {
        seed: 80,
        first_yield_day: 10,
        max_yield_day: 12,
        interval: 0,
        max_yield: 6,
        ongoing: false,
    },
];

#[derive(Clone, Copy)]
struct AnimalData {
    cost: u32,
    first_yield_day: u32,
    interval: u32,
    max_held: u32,
    product: Product,
}

const ANIMALS: [AnimalData; ANIMAL_COUNT] = [
    AnimalData {
        cost: 300,
        first_yield_day: 4,
        interval: 1,
        max_held: 4,
        product: Product::Egg,
    },
    AnimalData {
        cost: 400,
        first_yield_day: 8,
        interval: 2,
        max_held: 6,
        product: Product::Milk,
    },
    AnimalData {
        cost: 500,
        first_yield_day: 6,
        interval: 3,
        max_held: 6,
        product: Product::Wool,
    },
];

fn animal_structure(animal: Animal) -> StructureKind {
    match animal {
        Animal::Goose => StructureKind::Coop,
        Animal::Cow | Animal::Sheep => StructureKind::Pasture,
    }
}

fn shape(shape: PriceShape, x: f64, scale: f64) -> f64 {
    let x = x.max(0.0);
    match shape {
        PriceShape::Linear => x,
        PriceShape::Square => x * x,
        PriceShape::SquareRoot => x.sqrt(),
        // Match Python's math.log(1.0 + x), including its rounding behavior.
        PriceShape::Log => (1.0 + x).ln(),
        PriceShape::Log10 => (1.0 + x).log10(),
        PriceShape::Hinge if scale > 0.0 => {
            let u = x / scale;
            u + HINGE_GAIN * (u - 1.0).max(0.0).powi(2)
        }
        PriceShape::Hinge => x,
    }
}

// Python round() uses ties-to-even. Prices are non-negative, so this compact
// implementation also matches int(round(price)) at exact half-integers.
fn py_round_nonnegative(value: f64) -> f64 {
    let floor = value.floor();
    let fraction = value - floor;
    let rounded = if fraction < 0.5 {
        floor
    } else if fraction > 0.5 || (floor as u64) % 2 == 1 {
        floor + 1.0
    } else {
        floor
    };
    rounded.max(PRICE_FLOOR)
}

fn market_price(state: &GameState, item: Product, inventory: i32) -> f64 {
    let p = state.market.params[item.index()];
    let distance = (inventory - p.initial_inventory).unsigned_abs() as f64;
    let (curve, target, sign) = if inventory < p.initial_inventory {
        (p.below_shape, p.below_target, 1.0)
    } else {
        (p.above_shape, p.above_target, -1.0)
    };
    let denominator = shape(curve, p.scale, p.scale);
    let amplitude = if denominator == 0.0 {
        0.0
    } else {
        target * p.base / denominator
    };
    py_round_nonnegative(p.base + sign * amplitude * shape(curve, distance, p.scale))
}

fn refresh_prices(state: &mut GameState) {
    for product in Product::ALL {
        state.market.prices[product.index()] =
            market_price(state, product, state.market.inventory[product.index()]);
    }
}

/// Recompute only the products whose inventory may have changed since the last
/// refresh.  `market_price` depends on `(params, inventory[item])` alone, so an
/// untouched product already holds `market_price` of its current inventory and
/// recomputing it would write the same value.
///
/// Invariant relied upon: `prices[i] == market_price(inventory[i])` for every
/// product on entry.  Only used inside one `process_market` order slot, where
/// the only writer of `market.inventory` is `commit_unit` (SELL <product>,
/// BUY_PRODUCT WHEAT/FERTILIZER) and the slot's touched set is a superset of
/// what it can move.  `town_consume` runs after every slot loop and refreshes
/// every product, so a state assembled outside the engine (`rollout.rs` injects
/// observed inventory and prices together) is repaired on its first step
/// instead of carrying the injected prices forward.
fn refresh_touched_prices(state: &mut GameState, touched: [bool; PRODUCT_COUNT]) {
    for product in Product::ALL {
        if touched[product.index()] {
            state.market.prices[product.index()] =
                market_price(state, product, state.market.inventory[product.index()]);
        }
    }
}

fn shed_tiles(board_size: u16) -> [(u16, u16); 4] {
    let half = board_size / 2;
    [
        (half - 1, half - 1),
        (half, half - 1),
        (half - 1, half),
        (half, half),
    ]
}

fn is_shed_adjacent(x: u16, y: u16, board_size: u16) -> bool {
    shed_tiles(board_size).contains(&(x, y))
}

fn tile_index(x: u16, y: u16, board_size: u16) -> Option<usize> {
    if x < board_size && y < board_size {
        Some(y as usize * board_size as usize + x as usize)
    } else {
        None
    }
}

fn inventory_total(values: &[u32; ITEM_COUNT]) -> u32 {
    values.iter().copied().sum()
}

fn ensure_inventory(private: &mut PrivateState, unit_index: usize) {
    while private.inventories.len() <= unit_index {
        private.inventories.push([0; ITEM_COUNT]);
    }
    while private.inventory_orders.len() <= unit_index {
        private.inventory_orders.push(Vec::new());
    }
}

fn inventory_add(private: &mut PrivateState, unit_index: usize, item: Item, amount: u32) {
    if amount == 0 {
        return;
    }
    ensure_inventory(private, unit_index);
    let item_index = item.index();
    if private.inventories[unit_index][item_index] == 0 {
        private.inventory_orders[unit_index].push(item);
    }
    private.inventories[unit_index][item_index] += amount;
}

fn inventory_take(private: &mut PrivateState, unit_index: usize, item: Item, amount: u32) -> bool {
    ensure_inventory(private, unit_index);
    let item_index = item.index();
    if private.inventories[unit_index][item_index] < amount {
        return false;
    }
    private.inventories[unit_index][item_index] -= amount;
    if private.inventories[unit_index][item_index] == 0 {
        private.inventory_orders[unit_index].retain(|ordered| *ordered != item);
    }
    true
}

fn make_plant(crop: Crop, day: u32, turns_per_day: u32) -> PlantState {
    let data = CROPS[crop.index()];
    PlantState {
        crop,
        planted_day: day,
        watered_today: false,
        consecutive_unwatered: 1,
        yield_units: u32::from(!data.ongoing),
        max_lifespan_step: if data.ongoing {
            -1
        } else {
            ((day + data.max_yield_day + 1) * turns_per_day).min(i32::MAX as u32) as i32
        },
        fertilized_until_day: -1,
    }
}

fn make_animal(animal: Animal, day: u32) -> AnimalState {
    AnimalState {
        animal,
        placed_day: day,
        yield_units: 0,
        consecutive_unfed: 0,
        fed_today: false,
        cared_today: false,
        fertilizer_available: false,
        pending_care_bonus: 0,
    }
}

fn unit_position(farm: &Farm, index: usize) -> Option<crate::model::Position> {
    if index == 0 {
        Some(farm.farmer)
    } else {
        farm.hands.get(index - 1).copied()
    }
}

// Keeping the official interpreter arguments explicit makes rule-order audits
// substantially easier than hiding them behind mutable global context.
#[allow(clippy::too_many_arguments)]
fn apply_unit_action(
    farm: &mut Farm,
    private: &mut PrivateState,
    unit_index: usize,
    action: &UnitAction,
    board_size: u16,
    day: u32,
    turns_per_day: u32,
    shed_capacity: u32,
) {
    let Some(position) = unit_position(farm, unit_index) else {
        return;
    };
    ensure_inventory(private, unit_index);

    if let UnitAction::Move(direction) = action {
        let (dx, dy) = match direction {
            Direction::North => (0_i32, -1_i32),
            Direction::South => (0, 1),
            Direction::East => (1, 0),
            Direction::West => (-1, 0),
            Direction::Pass => return,
        };
        let nx = i32::from(position.x) + dx;
        let ny = i32::from(position.y) + dy;
        if nx < 0 || ny < 0 || nx >= i32::from(board_size) || ny >= i32::from(board_size) {
            return;
        }
        let next = crate::model::Position {
            x: nx as u16,
            y: ny as u16,
        };
        if unit_index == 0 {
            farm.farmer = next;
        } else {
            farm.hands[unit_index - 1] = next;
        }
        return;
    }
    if matches!(action, UnitAction::Pass) {
        return;
    }

    let Some(index) = tile_index(position.x, position.y, board_size) else {
        return;
    };
    if index >= farm.tiles.len() {
        return;
    }

    // Shed operations precede the LOCKED guard in the Python interpreter.
    if matches!(action, UnitAction::Drop) {
        if !is_shed_adjacent(position.x, position.y, board_size) {
            return;
        }
        let order = private.inventory_orders[unit_index].clone();
        for item in order {
            let item_index = item.index();
            let room = shed_capacity.saturating_sub(inventory_total(&private.shed));
            let amount = private.inventories[unit_index][item_index].min(room);
            private.shed[item_index] += amount;
            // Python DROP discards overflow instead of leaving it with the unit.
            private.inventories[unit_index][item_index] = 0;
        }
        private.inventory_orders[unit_index].clear();
        return;
    }

    if let UnitAction::Pickup(item, requested) = action {
        if !is_shed_adjacent(position.x, position.y, board_size) || *requested == 0 {
            return;
        }
        let item_index = item.index();
        let amount = (*requested).min(private.shed[item_index]);
        private.shed[item_index] -= amount;
        inventory_add(private, unit_index, *item, amount);
        return;
    }

    if let UnitAction::Place(item, requested) = action {
        if let Item::Animal(animal) = item {
            let expected = animal_structure(*animal);
            let can_place = matches!(
                &farm.tiles[index],
                Tile::Structure(StructureState { kind, animal: None }) if *kind == expected
            );
            if can_place {
                if inventory_take(private, unit_index, *item, 1) {
                    farm.tiles[index] = Tile::Structure(StructureState {
                        kind: expected,
                        animal: Some(make_animal(*animal, day)),
                    });
                }
                return;
            }
        }
        if is_shed_adjacent(position.x, position.y, board_size) && *requested > 0 {
            let item_index = item.index();
            let room = shed_capacity.saturating_sub(inventory_total(&private.shed));
            let amount = (*requested)
                .min(private.inventories[unit_index][item_index])
                .min(room);
            let _ = inventory_take(private, unit_index, *item, amount);
            private.shed[item_index] += amount;
        }
        return;
    }

    if matches!(&farm.tiles[index], Tile::Locked) {
        return;
    }
    match action {
        UnitAction::Plant(crop) => {
            if matches!(&farm.tiles[index], Tile::Empty) && private.seeds[crop.index()] > 0 {
                private.seeds[crop.index()] -= 1;
                farm.tiles[index] = Tile::Plant(make_plant(*crop, day, turns_per_day));
            }
        }
        UnitAction::Water => {
            let Tile::Plant(plant) = &mut farm.tiles[index] else {
                return;
            };
            if plant.watered_today {
                return;
            }
            plant.watered_today = true;
            let data = CROPS[plant.crop.index()];
            if !data.ongoing {
                let age = day.saturating_sub(plant.planted_day);
                let start = data.max_yield_day.div_ceil(2);
                if (start..=data.max_yield_day).contains(&age) {
                    let bonus = if plant.fertilized_until_day >= day as i32 {
                        2
                    } else {
                        1
                    };
                    plant.yield_units = data.max_yield.min(plant.yield_units + bonus);
                }
            }
        }
        UnitAction::Harvest => match &mut farm.tiles[index] {
            Tile::Plant(plant) if plant.yield_units > 0 => {
                let data = CROPS[plant.crop.index()];
                if day.saturating_sub(plant.planted_day) < data.first_yield_day {
                    return;
                }
                let product = Product::ALL[plant.crop.index()];
                let units = plant.yield_units;
                plant.yield_units = 0;
                inventory_add(private, unit_index, Item::Product(product), units);
                if !data.ongoing {
                    farm.tiles[index] = Tile::Empty;
                }
            }
            Tile::Structure(StructureState {
                animal: Some(animal),
                ..
            }) if animal.yield_units > 0 => {
                let product = ANIMALS[animal.animal.index()].product;
                let units = animal.yield_units;
                animal.yield_units = 0;
                inventory_add(private, unit_index, Item::Product(product), units);
            }
            _ => {}
        },
        UnitAction::Fertilize => {
            let Tile::Plant(plant) = &mut farm.tiles[index] else {
                return;
            };
            if !inventory_take(private, unit_index, Item::Product(Product::Fertilizer), 1) {
                return;
            }
            plant.fertilized_until_day =
                plant.fertilized_until_day.max(day.saturating_add(2) as i32);
        }
        UnitAction::Dig => {
            if !matches!(
                &farm.tiles[index],
                Tile::Empty
                    | Tile::Structure(StructureState {
                        animal: Some(_),
                        ..
                    })
            ) {
                farm.tiles[index] = Tile::Empty;
            }
        }
        UnitAction::BuildCoop if matches!(&farm.tiles[index], Tile::Empty) => {
            farm.tiles[index] = Tile::Structure(StructureState {
                kind: StructureKind::Coop,
                animal: None,
            });
        }
        UnitAction::BuildPasture if matches!(&farm.tiles[index], Tile::Empty) => {
            farm.tiles[index] = Tile::Structure(StructureState {
                kind: StructureKind::Pasture,
                animal: None,
            });
        }
        UnitAction::Feed => {
            let Tile::Structure(StructureState {
                animal: Some(animal),
                ..
            }) = &mut farm.tiles[index]
            else {
                return;
            };
            if animal.fed_today {
                return;
            }
            if !inventory_take(private, unit_index, Item::Product(Product::Wheat), 1) {
                return;
            }
            animal.fed_today = true;
        }
        UnitAction::CollectFertilizer => {
            let Tile::Structure(StructureState {
                animal: Some(animal),
                ..
            }) = &mut farm.tiles[index]
            else {
                return;
            };
            if animal.fertilizer_available {
                animal.fertilizer_available = false;
                inventory_add(private, unit_index, Item::Product(Product::Fertilizer), 1);
            }
        }
        UnitAction::Care => {
            let Tile::Structure(StructureState {
                animal: Some(animal),
                ..
            }) = &mut farm.tiles[index]
            else {
                return;
            };
            if !animal.cared_today {
                animal.cared_today = true;
            }
        }
        _ => {}
    }
}

/// What a unit action reports besides its state change.
enum UnitObserver<'a> {
    /// Plain `step`: nothing.
    Silent,
    /// A full `UnitActionTrace` row.
    Trace(&'a mut Vec<UnitActionTrace>),
    /// Add one when the action had no effect (PASS / MOVE PASS never count).
    Count(&'a mut u32),
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn apply_unit_action_observed(
    farm: &mut Farm,
    private: &mut PrivateState,
    unit_index: usize,
    action: &UnitAction,
    board_size: u16,
    day: u32,
    turns_per_day: u32,
    shed_capacity: u32,
    step: u32,
    player: usize,
    blocked: bool,
    observer: UnitObserver<'_>,
) {
    let counter = match observer {
        UnitObserver::Silent => {
            if !blocked {
                apply_unit_action(
                    farm,
                    private,
                    unit_index,
                    action,
                    board_size,
                    day,
                    turns_per_day,
                    shed_capacity,
                );
            }
            return;
        }
        UnitObserver::Trace(rows) => {
            apply_unit_action_with_trace(
                farm,
                private,
                unit_index,
                action,
                board_size,
                day,
                turns_per_day,
                shed_capacity,
                step,
                player,
                blocked,
                rows,
            );
            return;
        }
        UnitObserver::Count(counter) => counter,
    };
    // Counting: the same `success` predicate as the trace row, without the row.
    // PASS and MOVE PASS are never failures, so they need no before/after read.
    if matches!(action, UnitAction::Pass | UnitAction::Move(Direction::Pass)) {
        if !blocked {
            apply_unit_action(
                farm,
                private,
                unit_index,
                action,
                board_size,
                day,
                turns_per_day,
                shed_capacity,
            );
        }
        return;
    }
    let position_before = unit_position(farm, unit_index);
    let tile_before = position_before
        .and_then(|position| tile_index(position.x, position.y, board_size))
        .and_then(|index| farm.tiles.get(index).cloned());
    let inventory_before = private
        .inventories
        .get(unit_index)
        .copied()
        .unwrap_or([0; ITEM_COUNT]);
    let seeds_before = private.seeds;
    if !blocked {
        apply_unit_action(
            farm,
            private,
            unit_index,
            action,
            board_size,
            day,
            turns_per_day,
            shed_capacity,
        );
    }
    let position_after = unit_position(farm, unit_index);
    let tile_after = position_after
        .and_then(|position| tile_index(position.x, position.y, board_size))
        .and_then(|index| farm.tiles.get(index).cloned());
    let inventory_after = private
        .inventories
        .get(unit_index)
        .copied()
        .unwrap_or([0; ITEM_COUNT]);
    let success = position_before != position_after
        || tile_before != tile_after
        || inventory_after != inventory_before
        || private.seeds != seeds_before;
    if !success {
        *counter += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_unit_action_with_trace(
    farm: &mut Farm,
    private: &mut PrivateState,
    unit_index: usize,
    action: &UnitAction,
    board_size: u16,
    day: u32,
    turns_per_day: u32,
    shed_capacity: u32,
    step: u32,
    player: usize,
    blocked: bool,
    rows: &mut Vec<UnitActionTrace>,
) {
    let position_before = unit_position(farm, unit_index);
    let tile_before = position_before
        .and_then(|position| tile_index(position.x, position.y, board_size))
        .and_then(|index| farm.tiles.get(index).cloned());
    let inventory_before = private
        .inventories
        .get(unit_index)
        .copied()
        .unwrap_or([0; ITEM_COUNT]);
    let seeds_before = private.seeds;
    if !blocked {
        apply_unit_action(
            farm,
            private,
            unit_index,
            action,
            board_size,
            day,
            turns_per_day,
            shed_capacity,
        );
    }
    let position_after = unit_position(farm, unit_index);
    let tile_after = position_after
        .and_then(|position| tile_index(position.x, position.y, board_size))
        .and_then(|index| farm.tiles.get(index).cloned());
    let inventory_after = private
        .inventories
        .get(unit_index)
        .copied()
        .unwrap_or([0; ITEM_COUNT]);
    let inventory_delta = std::array::from_fn(|index| {
        i64::from(inventory_after[index]) - i64::from(inventory_before[index])
    });
    let seed_delta = std::array::from_fn(|index| {
        i64::from(private.seeds[index]) - i64::from(seeds_before[index])
    });
    let success = position_before != position_after
        || tile_before != tile_after
        || inventory_delta != [0; ITEM_COUNT]
        || seed_delta != [0; CROP_COUNT];
    rows.push(UnitActionTrace {
        step,
        player,
        unit_index,
        action: action.clone(),
        position_before,
        position_after,
        tile_before,
        tile_after,
        inventory_delta,
        seed_delta,
        success,
    });
}

fn spawn_hand(farm: &Farm, board_size: u16) -> crate::model::Position {
    let candidates = shed_tiles(board_size);
    let mut counts = [0_u32; 4];
    for position in std::iter::once(&farm.farmer).chain(farm.hands.iter()) {
        if let Some(index) = candidates
            .iter()
            .position(|p| *p == (position.x, position.y))
        {
            counts[index] += 1;
        }
    }
    let best = (0..4).min_by_key(|&i| (counts[i], i)).unwrap_or(0);
    crate::model::Position {
        x: candidates[best].0,
        y: candidates[best].1,
    }
}

fn fib(index: u32) -> u64 {
    let (mut a, mut b) = (1_u64, 1_u64);
    for _ in 0..index {
        (a, b) = (b, a.saturating_add(b));
    }
    a
}

fn hire(state: &mut GameState, player: usize) {
    let farm = &mut state.farms[player];
    let cost =
        u64::from(state.config.farm_hand_cost_multiplier).saturating_mul(fib(farm.hires_today));
    if farm.money < cost as f64 {
        return;
    }
    farm.money -= cost as f64;
    farm.hires_today += 1;
    let position = spawn_hand(farm, state.config.board_size);
    farm.hands.push(position);
    state.private[player].inventories.push([0; ITEM_COUNT]);
    state.private[player].inventory_orders.push(Vec::new());
}

fn buy_land(state: &mut GameState, player: usize) {
    let farm = &mut state.farms[player];
    // The compact model stores NW/NE/SW/SE as bits, not as a count.
    let extra = farm.unlocked_quadrants.count_ones().saturating_sub(1) as usize;
    let Some(&cost) = LAND_PRICES.get(extra) else {
        return;
    };
    if farm.money < f64::from(cost) {
        return;
    }
    farm.money -= f64::from(cost);
    farm.unlocked_quadrants |= 1 << (extra + 1);
    let board = state.config.board_size as usize;
    let half = board / 2;
    for y in 0..board {
        for x in 0..board {
            let in_quadrant = match extra {
                0 => x >= half && y < half,  // NE
                1 => x < half && y >= half,  // SW
                2 => x >= half && y >= half, // SE
                _ => false,
            };
            let index = y * board + x;
            if in_quadrant && index < farm.tiles.len() && matches!(&farm.tiles[index], Tile::Locked)
            {
                farm.tiles[index] = Tile::Empty;
            }
        }
    }
}

#[derive(Clone, Copy)]
enum PendingOrder {
    BuySeed(Crop, u32),
    BuyProduct(Product, u32),
    BuyAnimal(Animal, u32),
    Sell(Item, u32),
}

fn commit_unit(state: &mut GameState, player: usize, order: PendingOrder, price: f64) -> bool {
    let shed_capacity = state.config.shed_capacity;
    match order {
        PendingOrder::Sell(Item::Product(product), _) => {
            let index = product.index();
            if state.private[player].shed[index] == 0 {
                return false;
            }
            state.private[player].shed[index] -= 1;
            state.farms[player].money += price;
            if price > PRICE_FLOOR {
                state.market.inventory[index] += 1;
            }
            true
        }
        PendingOrder::Sell(Item::Animal(_), _) => false,
        PendingOrder::BuyProduct(product @ (Product::Wheat | Product::Fertilizer), _) => {
            if state.farms[player].money < price
                || inventory_total(&state.private[player].shed) >= shed_capacity
            {
                return false;
            }
            state.farms[player].money -= price;
            state.private[player].shed[product.index()] += 1;
            state.market.inventory[product.index()] -= 1;
            true
        }
        PendingOrder::BuyProduct(_, _) => false,
        PendingOrder::BuySeed(crop, _) => {
            if state.farms[player].money < price {
                return false;
            }
            state.farms[player].money -= price;
            state.private[player].seeds[crop.index()] += 1;
            true
        }
        PendingOrder::BuyAnimal(animal, _) => {
            if state.farms[player].money < price
                || inventory_total(&state.private[player].shed) >= shed_capacity
            {
                return false;
            }
            state.farms[player].money -= price;
            state.private[player].shed[Item::Animal(animal).index()] += 1;
            true
        }
    }
}

/// What a market slot reports besides its state change.
enum MarketObserver<'a> {
    Silent,
    /// One `MarketOrderTrace` row per player with an order in the slot.
    Trace(&'a mut Vec<MarketOrderTrace>),
    /// Add one when `seat`'s order in the slot changed nothing.
    Count { seat: usize, failed: &'a mut u32 },
}

fn process_market(state: &mut GameState, actions: &JointAction, mut observer: MarketObserver<'_>) {
    let limit = state.config.max_market_orders_per_turn.max(1);
    let max_len = actions
        .iter()
        .map(|a| a.market.len().min(limit))
        .max()
        .unwrap_or(0);
    for order_index in 0..max_len {
        // The pre-slot values are only needed by an observer (and, when
        // counting, only when the seat actually has an order in this slot).
        let observe_slot = match &observer {
            MarketObserver::Silent => false,
            MarketObserver::Trace(_) => true,
            MarketObserver::Count { seat, .. } => actions[*seat].market.get(order_index).is_some(),
        };
        let (money_before, shed_before, seeds_before, hands_before, quadrants_before) =
            if observe_slot {
                (
                    std::array::from_fn::<f64, PLAYER_COUNT, _>(|player| state.farms[player].money),
                    std::array::from_fn::<[u32; ITEM_COUNT], PLAYER_COUNT, _>(|player| {
                        state.private[player].shed
                    }),
                    std::array::from_fn::<[u32; CROP_COUNT], PLAYER_COUNT, _>(|player| {
                        state.private[player].seeds
                    }),
                    std::array::from_fn::<usize, PLAYER_COUNT, _>(|player| {
                        state.farms[player].hands.len()
                    }),
                    std::array::from_fn::<u32, PLAYER_COUNT, _>(|player| {
                        state.farms[player].unlocked_quadrants.count_ones()
                    }),
                )
            } else {
                (
                    [0.0; PLAYER_COUNT],
                    [[0; ITEM_COUNT]; PLAYER_COUNT],
                    [[0; CROP_COUNT]; PLAYER_COUNT],
                    [0; PLAYER_COUNT],
                    [0; PLAYER_COUNT],
                )
            };
        let mut pending: [Option<PendingOrder>; PLAYER_COUNT] = [None; PLAYER_COUNT];
        for player in 0..PLAYER_COUNT {
            let Some(order) = actions[player]
                .market
                .get(order_index)
                .filter(|_| order_index < limit)
            else {
                continue;
            };
            match *order {
                MarketOrder::Hire => hire(state, player),
                MarketOrder::BuyLand => buy_land(state, player),
                MarketOrder::BuySeed(crop, count) if count > 0 => {
                    pending[player] = Some(PendingOrder::BuySeed(crop, count))
                }
                MarketOrder::BuyProduct(product, count) if count > 0 => {
                    pending[player] = Some(PendingOrder::BuyProduct(product, count))
                }
                MarketOrder::BuyAnimal(animal, count) if count > 0 => {
                    pending[player] = Some(PendingOrder::BuyAnimal(animal, count))
                }
                MarketOrder::Sell(item, count) if count > 0 => {
                    pending[player] = Some(PendingOrder::Sell(item, count))
                }
                _ => {}
            }
        }

        // Only SELL <product> and BUY_PRODUCT WHEAT/FERTILIZER move market
        // inventory (see `commit_unit`), and the slot's pending products are
        // fixed before the unit loop (the loop only decrements counts or drops
        // an order), so this is a superset of what the slot can re-price.
        let mut touched = [false; PRODUCT_COUNT];
        for order in pending.iter().flatten() {
            match *order {
                PendingOrder::Sell(Item::Product(product), _) => touched[product.index()] = true,
                PendingOrder::BuyProduct(product @ (Product::Wheat | Product::Fertilizer), _) => {
                    touched[product.index()] = true;
                }
                _ => {}
            }
        }
        // Both quotes are taken from the same pre-commit inventory for each unit.
        let mut market_iterations = 0_u32;
        loop {
            market_iterations += 1;
            if market_iterations >= 100_000 {
                break;
            }
            let mut quotes: [Option<f64>; PLAYER_COUNT] = [None; PLAYER_COUNT];
            for player in 0..PLAYER_COUNT {
                quotes[player] = match pending[player] {
                    Some(PendingOrder::Sell(Item::Product(product), left)) if left > 0 => Some(
                        market_price(state, product, state.market.inventory[product.index()]),
                    ),
                    Some(PendingOrder::BuyProduct(
                        product @ (Product::Wheat | Product::Fertilizer),
                        left,
                    )) if left > 0 => Some(market_price(
                        state,
                        product,
                        state.market.inventory[product.index()] - 1,
                    )),
                    Some(PendingOrder::BuySeed(crop, left)) if left > 0 => {
                        Some(f64::from(CROPS[crop.index()].seed))
                    }
                    Some(PendingOrder::BuyAnimal(animal, left)) if left > 0 => {
                        Some(f64::from(ANIMALS[animal.index()].cost))
                    }
                    Some(_) => {
                        pending[player] = None;
                        None
                    }
                    None => None,
                };
            }
            if quotes.iter().all(Option::is_none) {
                break;
            }
            let mut committed = false;
            for player in 0..PLAYER_COUNT {
                let (Some(order), Some(price)) = (pending[player], quotes[player]) else {
                    continue;
                };
                if commit_unit(state, player, order, price) {
                    committed = true;
                    pending[player] = match order {
                        PendingOrder::BuySeed(item, left) if left > 1 => {
                            Some(PendingOrder::BuySeed(item, left - 1))
                        }
                        PendingOrder::BuyProduct(item, left) if left > 1 => {
                            Some(PendingOrder::BuyProduct(item, left - 1))
                        }
                        PendingOrder::BuyAnimal(item, left) if left > 1 => {
                            Some(PendingOrder::BuyAnimal(item, left - 1))
                        }
                        PendingOrder::Sell(item, left) if left > 1 => {
                            Some(PendingOrder::Sell(item, left - 1))
                        }
                        _ => None,
                    };
                } else {
                    pending[player] = None;
                }
            }
            if !committed {
                break;
            }
        }
        refresh_touched_prices(state, touched);
        match &mut observer {
            MarketObserver::Silent => {}
            MarketObserver::Trace(rows) => {
                for player in 0..PLAYER_COUNT {
                    let Some(order) = actions[player].market.get(order_index).copied() else {
                        continue;
                    };
                    rows.push(MarketOrderTrace {
                        step: state.step,
                        player,
                        order_index,
                        order,
                        money_delta: state.farms[player].money - money_before[player],
                        shed_delta: std::array::from_fn(|index| {
                            i64::from(state.private[player].shed[index])
                                - i64::from(shed_before[player][index])
                        }),
                        seed_delta: std::array::from_fn(|index| {
                            i64::from(state.private[player].seeds[index])
                                - i64::from(seeds_before[player][index])
                        }),
                        hands_delta: state.farms[player].hands.len() as i64
                            - hands_before[player] as i64,
                        quadrants_delta: i64::from(
                            state.farms[player].unlocked_quadrants.count_ones(),
                        ) - i64::from(quadrants_before[player]),
                    });
                }
            }
            MarketObserver::Count { seat, failed } => {
                // The evaluator's predicate on a trace row: every delta zero.
                let seat = *seat;
                if observe_slot
                    && state.farms[seat].money == money_before[seat]
                    && state.private[seat].shed == shed_before[seat]
                    && state.private[seat].seeds == seeds_before[seat]
                    && state.farms[seat].hands.len() == hands_before[seat]
                    && state.farms[seat].unlocked_quadrants.count_ones() == quadrants_before[seat]
                {
                    **failed += 1;
                }
            }
        }
    }
}

fn shop_products(shop: Shop) -> &'static [Product] {
    use Product::*;
    match shop {
        Shop::Bakery => &[Egg, Wheat],
        Shop::PizzaShop => &[Milk, Tomato, Wheat],
        Shop::BrunchSpot => &[Egg, Wheat, Strawberry],
        Shop::YarnStore => &[Wool],
        Shop::IceCreamShop => &[Strawberry, Milk, Wheat],
        Shop::PetCafe => &[Carrot],
        Shop::SmoothieShop => &[Strawberry, Milk],
        Shop::FarmersMarket => &[Wheat, Carrot, Tomato, Strawberry],
    }
}

fn town_consume(state: &mut GameState, step: u32) {
    if step.is_multiple_of(state.config.town_shop_sell_interval.max(1)) {
        for shop in &state.town.unlocked_shops {
            let products = shop_products(*shop);
            let amount = if products.len() == 1 { 2 } else { 1 };
            for product in products {
                state.market.inventory[product.index()] -= amount;
            }
        }
    }
    if step.is_multiple_of(state.config.town_center_sell_interval.max(1)) {
        for product in Product::ALL {
            if product != Product::Fertilizer {
                state.market.inventory[product.index()] -= 1;
            }
        }
    }
    // Deliberately a full refresh, not the incremental one: this runs once per
    // step, so it is what repairs `prices[i] == market_price(inventory[i])` for
    // a state that was not built by the engine.  `rollout.rs` reconstructs a
    // state from a live observation and writes both `market.inventory` and
    // `market.prices` from the outside; without this the injected prices would
    // survive to the end of the rollout.  It also covers step 0, where
    // `new_game` leaves the raw base in `prices` (a custom fractional base is
    // only visible before this point, as in Python).
    refresh_prices(state);
}

/// Age the plants that are due this step.
///
/// `farm.decay_from` is a cache: no plant on `farm` can decay below it.  The
/// scan is skipped while that holds and the watermark is recomputed from the
/// surviving plants on every scan, so it never outlives its evidence.  It
/// travels inside the `Farm`, so replacing or cloning a farm carries the right
/// mark with the board (see `Farm::decay_from`).  Callers reset it to 0
/// whenever a plant may have been created or had its lifespan set (a PLANT
/// action, the end of a day).
fn decay_plants(farm: &mut Farm, step: u32) {
    if step < farm.decay_from {
        return;
    }
    let mut next = u32::MAX;
    for tile in &mut farm.tiles {
        let Tile::Plant(plant) = tile else { continue };
        let lifespan = plant.max_lifespan_step;
        if lifespan < 0 {
            // Ongoing crop with no end yet; `daily_refresh_plants` sets one at
            // the last production, and the end of that day resets the mark.
            continue;
        }
        let lifespan = lifespan as u32;
        if step < lifespan {
            next = next.min(lifespan);
            continue;
        }
        if !(step - lifespan).is_multiple_of(2) {
            next = next.min(step + 1);
            continue;
        }
        plant.yield_units = plant.yield_units.saturating_sub(1);
        if plant.yield_units == 0 {
            *tile = Tile::Weed;
        } else {
            next = next.min(step + 2);
        }
    }
    farm.decay_from = next;
}

fn daily_refresh_plants(farm: &mut Farm, day: u32, turns_per_day: u32) {
    let next_day = day + 1;
    for tile in &mut farm.tiles {
        let Tile::Plant(plant) = tile else { continue };
        let was_watered = plant.watered_today;
        if was_watered {
            plant.consecutive_unwatered = 0;
        } else {
            plant.consecutive_unwatered += 1;
        }
        plant.watered_today = false;
        if plant.consecutive_unwatered >= 2 {
            *tile = Tile::Weed;
            continue;
        }
        let data = CROPS[plant.crop.index()];
        if !data.ongoing {
            continue;
        }
        let days_since_first =
            i64::from(next_day) - i64::from(plant.planted_day) - i64::from(data.first_yield_day);
        if days_since_first < 0 || days_since_first % i64::from(data.interval) != 0 {
            continue;
        }
        let production_count = days_since_first as u32 / data.interval + 1;
        if production_count > data.max_yield {
            continue;
        }
        let bonus = if was_watered && plant.fertilized_until_day >= day as i32 {
            2
        } else {
            1
        };
        plant.yield_units = data.max_yield.min(plant.yield_units + bonus);
        if production_count == data.max_yield {
            plant.max_lifespan_step = ((next_day + 1) * turns_per_day).min(i32::MAX as u32) as i32;
        }
    }
}

fn daily_refresh_animals(farm: &mut Farm, day: u32) {
    let next_day = day + 1;
    for tile in &mut farm.tiles {
        let Tile::Structure(structure) = tile else {
            continue;
        };
        let Some(animal) = &mut structure.animal else {
            continue;
        };
        if animal.fed_today {
            animal.consecutive_unfed = 0;
        } else {
            animal.consecutive_unfed += 1;
        }
        if animal.consecutive_unfed >= 2 {
            structure.animal = None;
            continue;
        }
        let data = ANIMALS[animal.animal.index()];
        let days_since_first =
            i64::from(next_day) - i64::from(animal.placed_day) - i64::from(data.first_yield_day);
        if days_since_first >= 0 && days_since_first % i64::from(data.interval) == 0 {
            let bonus = if animal.fed_today {
                animal.pending_care_bonus
            } else {
                0
            };
            animal.yield_units = data.max_held.min(animal.yield_units + 1 + bonus);
            animal.pending_care_bonus = 0;
        }
        if animal.cared_today && animal.fed_today {
            animal.pending_care_bonus += 1;
        }
        animal.fertilizer_available = true;
        animal.fed_today = false;
        animal.cared_today = false;
    }
}

/// Minimal CPython-compatible MT19937 for end-of-day weed/shop draws.
struct PythonRandom {
    mt: [u32; 624],
    index: usize,
}

thread_local! {
    /// Initialised MT19937 state per end-of-day seed.  The seed is a pure
    /// function of `(state.seed, day)`, so the 624-word `init_by_array` result
    /// can be reused by every later day boundary with the same seed.  Only the
    /// post-init words are stored; draws always start at `index = 624`.
    static MT_INIT_CACHE: std::cell::RefCell<std::collections::HashMap<i128, Box<[u32; 624]>>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Upper bound on cached seeds per thread (2.5 KB each); the cache is emptied
/// when it is reached so long runs over many seeds stay bounded.
const MT_INIT_CACHE_LIMIT: usize = 4096;

impl PythonRandom {
    /// `new`, memoised on `seed`: the same 624 words, so the same draws.
    fn new_cached(seed: i128) -> Self {
        MT_INIT_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if let Some(mt) = cache.get(&seed) {
                return Self {
                    mt: **mt,
                    index: 624,
                };
            }
            let rng = Self::new(seed);
            if cache.len() >= MT_INIT_CACHE_LIMIT {
                cache.clear();
            }
            cache.insert(seed, Box::new(rng.mt));
            rng
        })
    }

    fn new(seed: i128) -> Self {
        let mut value = seed.unsigned_abs();
        let mut key = Vec::new();
        while value != 0 {
            key.push(value as u32);
            value >>= 32;
        }
        if key.is_empty() {
            key.push(0);
        }
        let mut rng = Self {
            mt: [0; 624],
            index: 624,
        };
        rng.init_by_array(&key);
        rng
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.mt[0] = 19_650_218;
        for i in 1..624 {
            self.mt[i] = 1_812_433_253_u32
                .wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        let (mut i, mut j) = (1_usize, 0_usize);
        for _ in 0..624.max(key.len()) {
            self.mt[i] = (self.mt[i]
                ^ (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1_664_525))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                self.mt[0] = self.mt[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            self.mt[i] = (self.mt[i]
                ^ (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                self.mt[0] = self.mt[623];
                i = 1;
            }
        }
        self.mt[0] = 0x8000_0000;
        self.index = 624;
    }

    fn u32(&mut self) -> u32 {
        if self.index >= 624 {
            for i in 0..624 {
                let y = (self.mt[i] & 0x8000_0000) | (self.mt[(i + 1) % 624] & 0x7fff_ffff);
                self.mt[i] =
                    self.mt[(i + 397) % 624] ^ (y >> 1) ^ if y & 1 != 0 { 0x9908_b0df } else { 0 };
            }
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    fn random(&mut self) -> f64 {
        let a = (self.u32() >> 5) as u64;
        let b = (self.u32() >> 6) as u64;
        ((a << 26) + b) as f64 / 9_007_199_254_740_992.0
    }

    fn choice_eight(&mut self) -> usize {
        loop {
            let value = (self.u32() >> 28) as usize;
            if value < 8 {
                return value;
            }
        }
    }
}

fn drop_inventories(private: &mut PrivateState, capacity: u32) {
    for unit_index in 0..private.inventories.len() {
        let order = private
            .inventory_orders
            .get(unit_index)
            .cloned()
            .unwrap_or_default();
        for item in order {
            let item_index = item.index();
            let room = capacity.saturating_sub(inventory_total(&private.shed));
            let amount = private.inventories[unit_index][item_index].min(room);
            private.shed[item_index] += amount;
            private.inventories[unit_index][item_index] = 0;
        }
    }
    for order in &mut private.inventory_orders {
        order.clear();
    }
}

fn end_of_day(state: &mut GameState, day: u32) {
    let turns = state.config.turns_per_day.max(1);
    let board = state.config.board_size;
    let seed = (i128::from(state.seed) * 1_000_003_i128) ^ i128::from(day);
    let mut rng = PythonRandom::new_cached(seed);
    for player in 0..PLAYER_COUNT {
        daily_refresh_plants(&mut state.farms[player], day, turns);
        daily_refresh_animals(&mut state.farms[player], day);
        for tile in &mut state.farms[player].tiles {
            if state.config.decoupled_rng {
                // Audit mode: one draw per tile, occupancy-independent.
                let draw = rng.random();
                if matches!(tile, Tile::Empty) && draw < state.config.weed_spawn_chance {
                    *tile = Tile::Weed;
                }
            } else if matches!(tile, Tile::Empty) && rng.random() < state.config.weed_spawn_chance {
                *tile = Tile::Weed;
            }
        }
        drop_inventories(&mut state.private[player], state.config.shed_capacity);
        let spawn = shed_tiles(board)[0];
        state.farms[player].farmer = crate::model::Position {
            x: spawn.0,
            y: spawn.1,
        };
        state.farms[player].hands.clear();
        state.farms[player].hires_today = 0;
        state.private[player].inventories.clear();
        state.private[player].inventories.push([0; ITEM_COUNT]);
        state.private[player].inventory_orders.clear();
        state.private[player].inventory_orders.push(Vec::new());
    }
    let next_day = day + 1;
    if next_day.is_multiple_of(state.config.town_shop_unlock_interval.max(1))
        && state.town.unlocked_shops.len() < MAX_SHOP_INSTANCES
    {
        // Alphabetical Python order: BAKERY, BRUNCH_SPOT, FARMERS_MARKET,
        // ICE_CREAM_SHOP, PET_CAFE, PIZZA_SHOP, SMOOTHIE_SHOP, YARN_STORE.
        const SORTED: [Shop; 8] = [
            Shop::Bakery,
            Shop::BrunchSpot,
            Shop::FarmersMarket,
            Shop::IceCreamShop,
            Shop::PetCafe,
            Shop::PizzaShop,
            Shop::SmoothieShop,
            Shop::YarnStore,
        ];
        state.town.unlocked_shops.push(SORTED[rng.choice_eight()]);
    }
}

/// No-effect counts for one seat over one `step`, with the evaluator's
/// predicates: a unit action fails when it changed neither position, tile,
/// inventory nor seeds (PASS and MOVE PASS are never failures); a market order
/// fails when its slot changed neither money, shed, seeds, hands nor land.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FailureCounts {
    pub units: u32,
    pub orders: u32,
}

enum StepObserver<'a> {
    Silent,
    Trace {
        units: Option<&'a mut Vec<UnitActionTrace>>,
        orders: Option<&'a mut Vec<MarketOrderTrace>>,
    },
    Count {
        seat: usize,
        counts: &'a mut FailureCounts,
    },
}

/// The unit-level observer implied by a step observer for `player`.
#[inline]
fn unit_observer<'a>(observer: &'a mut StepObserver<'_>, player: usize) -> UnitObserver<'a> {
    match observer {
        StepObserver::Silent | StepObserver::Trace { units: None, .. } => UnitObserver::Silent,
        StepObserver::Trace {
            units: Some(rows), ..
        } => UnitObserver::Trace(rows),
        StepObserver::Count { seat, counts } if *seat == player => {
            UnitObserver::Count(&mut counts.units)
        }
        StepObserver::Count { .. } => UnitObserver::Silent,
    }
}

/// Advance an initialized game by one joint action.
fn step_inner(state: &mut GameState, actions: &JointAction, mut observer: StepObserver<'_>) {
    if state.statuses.iter().all(|status| *status == Status::Done) {
        return;
    }
    let turns = state.config.turns_per_day.max(1);
    let day = state.step / turns;
    for (player, action) in actions.iter().enumerate() {
        let mut plant_demand = [0_u32; CROP_COUNT];
        for unit_action in std::iter::once(&action.farmer).chain(action.hands.iter()) {
            if let UnitAction::Plant(crop) = unit_action {
                plant_demand[crop.index()] += 1;
            }
        }
        let mut blocked = [false; CROP_COUNT];
        for crop in Crop::ALL {
            blocked[crop.index()] =
                plant_demand[crop.index()] > state.private[player].seeds[crop.index()];
        }
        if plant_demand.iter().any(|demand| *demand > 0) {
            // A new plant may appear this step, with a lifespan that can fall
            // below the current watermark: rescan.
            state.farms[player].decay_from = 0;
        }
        let current_step = state.step;
        let (farm, private) = (&mut state.farms[player], &mut state.private[player]);
        let farmer_blocked =
            matches!(&action.farmer, UnitAction::Plant(crop) if blocked[crop.index()]);
        apply_unit_action_observed(
            farm,
            private,
            0,
            &action.farmer,
            state.config.board_size,
            day,
            turns,
            state.config.shed_capacity,
            current_step,
            player,
            farmer_blocked,
            unit_observer(&mut observer, player),
        );
        for (hand, unit_action) in action.hands.iter().enumerate() {
            let hand_blocked =
                matches!(unit_action, UnitAction::Plant(crop) if blocked[crop.index()]);
            apply_unit_action_observed(
                farm,
                private,
                hand + 1,
                unit_action,
                state.config.board_size,
                day,
                turns,
                state.config.shed_capacity,
                current_step,
                player,
                hand_blocked,
                unit_observer(&mut observer, player),
            );
        }
    }
    let market_observer = match &mut observer {
        StepObserver::Silent | StepObserver::Trace { orders: None, .. } => MarketObserver::Silent,
        StepObserver::Trace {
            orders: Some(rows), ..
        } => MarketObserver::Trace(rows),
        StepObserver::Count { seat, counts } => MarketObserver::Count {
            seat: *seat,
            failed: &mut counts.orders,
        },
    };
    process_market(state, actions, market_observer);
    let step = state.step;
    town_consume(state, step);
    for player in 0..PLAYER_COUNT {
        decay_plants(&mut state.farms[player], step);
    }
    if (step + 1).is_multiple_of(turns) {
        end_of_day(state, day);
        // `daily_refresh_plants` may have set a lifespan, and weeds replaced
        // plants: rescan.
        for farm in &mut state.farms {
            farm.decay_from = 0;
        }
    }
    state.step = step + 1;
    state.day = state.step / turns;
    state.hour = state.step % turns;
    if step >= state.config.episode_steps.saturating_sub(2) {
        for player in 0..PLAYER_COUNT {
            state.statuses[player] = Status::Done;
            state.rewards[player] = Some(state.farms[player].money);
        }
    }
}

/// Advance a game while recording the realized effect of each market slot.
pub fn step_with_market_trace(
    state: &mut GameState,
    actions: &JointAction,
    market_trace: &mut Vec<MarketOrderTrace>,
) {
    step_inner(
        state,
        actions,
        StepObserver::Trace {
            units: None,
            orders: Some(market_trace),
        },
    );
}

/// Advance a game while recording realized unit and market effects.
pub fn step_with_trace(
    state: &mut GameState,
    actions: &JointAction,
    unit_trace: &mut Vec<UnitActionTrace>,
    market_trace: &mut Vec<MarketOrderTrace>,
) {
    step_inner(
        state,
        actions,
        StepObserver::Trace {
            units: Some(unit_trace),
            orders: Some(market_trace),
        },
    );
}

/// Advance a game and return `seat`'s no-effect unit-action and order counts
/// for this step.  Same transition as [`step`]; the same counts as filtering
/// the [`step_with_trace`] rows of `seat` with the evaluator's predicates,
/// without building the rows or observing the other player.
pub fn step_counting_failures(
    state: &mut GameState,
    actions: &JointAction,
    seat: usize,
) -> FailureCounts {
    let mut counts = FailureCounts::default();
    step_inner(
        state,
        actions,
        StepObserver::Count {
            seat,
            counts: &mut counts,
        },
    );
    counts
}

/// Advance an initialized game by one joint action.
pub fn step(state: &mut GameState, actions: &JointAction) {
    step_inner(state, actions, StepObserver::Silent);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pass_action() -> JointAction {
        let action = crate::model::PlayerAction::default();
        [action.clone(), action]
    }

    #[test]
    fn python_rng_matches_known_seed_zero_prefix() {
        let mut rng = PythonRandom::new(0);
        let expected = [
            0.8444218515250481,
            0.7579544029403025,
            0.420571580830845,
            0.25891675029296335,
        ];
        for expected in expected {
            assert_eq!(rng.random(), expected);
        }
    }

    #[test]
    fn cached_python_rng_matches_fresh_init() {
        for seed in [
            0_i128,
            1,
            -7,
            6_300_000_i128 * 1_000_003 ^ 17,
            i128::from(i64::MAX) * 1_000_003,
        ] {
            let mut fresh = PythonRandom::new(seed);
            let mut cached = PythonRandom::new_cached(seed);
            let mut again = PythonRandom::new_cached(seed);
            assert_eq!(fresh.mt, cached.mt);
            assert_eq!(cached.index, 624);
            for _ in 0..1_000 {
                let expected = fresh.random();
                assert_eq!(cached.random(), expected);
                assert_eq!(again.random(), expected);
            }
        }
    }

    #[test]
    fn incremental_price_refresh_matches_full_refresh() {
        // After every step, a full recomputation must not change any price.
        let mut config = crate::model::Config::default();
        config.market_params[Product::Fertilizer.index()].base = 100.5;
        let mut state = crate::init::new_game(11, config);
        let mut lcg = 12_345_u64;
        let mut next = move || {
            lcg = lcg.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (lcg >> 33) as u32
        };
        for _ in 0..240 {
            let mut actions = pass_action();
            for player in 0..PLAYER_COUNT {
                state.private[player].shed[Product::Wheat.index()] += next() % 4;
                state.private[player].shed[Product::Milk.index()] += next() % 3;
                let orders = &mut actions[player].market;
                match next() % 6 {
                    0 => orders.push(MarketOrder::Sell(
                        Item::Product(Product::Wheat),
                        1 + next() % 3,
                    )),
                    1 => orders.push(MarketOrder::Sell(
                        Item::Product(Product::Milk),
                        1 + next() % 3,
                    )),
                    2 => orders.push(MarketOrder::BuyProduct(Product::Wheat, 1 + next() % 5)),
                    3 => orders.push(MarketOrder::BuyProduct(Product::Fertilizer, 1 + next() % 2)),
                    4 => orders.push(MarketOrder::BuySeed(Crop::Carrot, 1)),
                    _ => {}
                }
                if next() % 5 == 0 {
                    orders.push(MarketOrder::Sell(Item::Product(Product::Wheat), 2));
                }
            }
            if state.step % 48 == 5 {
                state.town.unlocked_shops.push(Shop::Bakery);
            }
            step(&mut state, &actions);
            let mut refreshed = state.clone();
            refresh_prices(&mut refreshed);
            assert_eq!(
                refreshed.market.prices, state.market.prices,
                "step {}",
                state.step
            );
        }
        // The script must actually move inventory, or the equality is vacuous.
        assert_ne!(
            state.market.inventory,
            [crate::constants::MARKET_INITIAL_INVENTORY; PRODUCT_COUNT]
        );
    }

    /// The evaluator's predicates on trace rows
    /// (`rust/kaggriculture-route/src/eval.rs`, `play_cell`).
    fn counts_from_trace(
        seat: usize,
        units: &[UnitActionTrace],
        orders: &[MarketOrderTrace],
    ) -> FailureCounts {
        FailureCounts {
            units: units
                .iter()
                .filter(|t| {
                    t.player == seat
                        && !t.success
                        && !matches!(t.action, UnitAction::Pass | UnitAction::Move(Direction::Pass))
                })
                .count() as u32,
            orders: orders
                .iter()
                .filter(|t| {
                    t.player == seat
                        && t.money_delta == 0.0
                        && t.hands_delta == 0
                        && t.quadrants_delta == 0
                        && t.shed_delta.iter().all(|d| *d == 0)
                        && t.seed_delta.iter().all(|d| *d == 0)
                })
                .count() as u32,
        }
    }

    fn scripted_action(next: &mut impl FnMut() -> u32, hands: usize) -> crate::model::PlayerAction {
        let unit = |r: u32| match r % 14 {
            0 => UnitAction::Move(Direction::North),
            1 => UnitAction::Move(Direction::South),
            2 => UnitAction::Move(Direction::East),
            3 => UnitAction::Move(Direction::West),
            4 => UnitAction::Plant(Crop::ALL[(r / 16) as usize % CROP_COUNT]),
            5 => UnitAction::Water,
            6 => UnitAction::Harvest,
            7 => UnitAction::Drop,
            8 => UnitAction::Pickup(Item::Product(Product::Wheat), 1 + r / 16 % 3),
            9 => UnitAction::Dig,
            10 => UnitAction::BuildCoop,
            11 => UnitAction::Place(Item::Animal(Animal::Goose), 1),
            12 => UnitAction::Feed,
            _ => UnitAction::Pass,
        };
        let mut action = crate::model::PlayerAction {
            farmer: unit(next()),
            hands: (0..hands).map(|_| unit(next())).collect(),
            market: Vec::new(),
        };
        for _ in 0..next() % 4 {
            let r = next();
            action.market.push(match r % 8 {
                0 => MarketOrder::Sell(Item::Product(Product::Wheat), 1 + r / 8 % 5),
                1 => MarketOrder::Sell(Item::Product(Product::Egg), 1 + r / 8 % 3),
                2 => MarketOrder::BuySeed(Crop::Wheat, 1 + r / 8 % 4),
                3 => MarketOrder::BuySeed(Crop::Carrot, 1),
                4 => MarketOrder::BuyProduct(Product::Wheat, 1 + r / 8 % 3),
                5 => MarketOrder::BuyAnimal(Animal::Goose, 1),
                6 => MarketOrder::Hire,
                _ => MarketOrder::BuyLand,
            });
        }
        action
    }

    #[test]
    fn counting_failures_matches_trace_rows_and_plain_step() {
        let mut lcg = 987_654_321_u64;
        let mut next = move || {
            lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (lcg >> 33) as u32
        };
        let mut plain = crate::init::new_game(5, crate::model::Config::default());
        let mut traced = plain.clone();
        let mut counted = [plain.clone(), plain.clone()];
        let mut totals = [FailureCounts::default(); PLAYER_COUNT];
        let mut unit_trace = Vec::new();
        let mut market_trace = Vec::new();
        while plain.statuses.iter().any(|s| *s == Status::Active) {
            let actions: JointAction = std::array::from_fn(|player| {
                scripted_action(&mut next, plain.farms[player].hands.len())
            });
            step(&mut plain, &actions);
            unit_trace.clear();
            market_trace.clear();
            step_with_trace(&mut traced, &actions, &mut unit_trace, &mut market_trace);
            assert_eq!(plain, traced);
            for seat in 0..PLAYER_COUNT {
                let counts = step_counting_failures(&mut counted[seat], &actions, seat);
                assert_eq!(counted[seat], plain, "seat {seat} step {}", plain.step);
                assert_eq!(
                    counts,
                    counts_from_trace(seat, &unit_trace, &market_trace),
                    "seat {seat} step {}",
                    plain.step
                );
                totals[seat].units += counts.units;
                totals[seat].orders += counts.orders;
            }
        }
        // The script must exercise both outcomes, or the equality is vacuous.
        for seat in 0..PLAYER_COUNT {
            assert!(totals[seat].units > 100, "{:?}", totals[seat]);
            assert!(totals[seat].orders > 20, "{:?}", totals[seat]);
        }
        assert!(plain.farms.iter().any(|farm| farm.money != 3_000.0));
    }

    #[test]
    fn decay_watermark_matches_scanning_every_step() {
        let mut lcg = 24_680_u64;
        let mut next = move || {
            lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (lcg >> 33) as u32
        };
        let mut config = crate::model::Config::default();
        config.starting_money = 20_000;
        let mut cached = crate::init::new_game(9, config);
        let mut scanned = cached.clone();
        let mut skipped = 0_u32;
        let mut decayed = 0_u32;
        let weeds = |state: &GameState| -> u32 {
            state
                .farms
                .iter()
                .map(|f| f.tiles.iter().filter(|t| matches!(t, Tile::Weed)).count() as u32)
                .sum()
        };
        while cached.statuses.iter().any(|s| *s == Status::Active) {
            let mut actions: JointAction = std::array::from_fn(|player| {
                scripted_action(&mut next, cached.farms[player].hands.len())
            });
            // Keep seeds coming and water the plants, so some reach a lifespan.
            for action in &mut actions {
                action.market.push(MarketOrder::BuySeed(Crop::Wheat, 2));
                action.market.push(MarketOrder::BuySeed(Crop::Tomato, 1));
                if next() % 3 == 0 {
                    action.farmer = UnitAction::Water;
                }
            }
            let before = weeds(&cached);
            let step_no = cached.step;
            if cached.farms.iter().all(|farm| step_no < farm.decay_from) {
                skipped += 1;
            }
            step(&mut cached, &actions);
            // The reference always scans.
            for farm in &mut scanned.farms {
                farm.decay_from = 0;
            }
            step(&mut scanned, &actions);
            assert_eq!(cached, scanned, "step {step_no}");
            // Away from a day boundary a new weed can only come from decay.
            if !(step_no + 1).is_multiple_of(24) {
                decayed += weeds(&cached).saturating_sub(before);
            }
        }
        assert!(skipped > 100, "skipped {skipped}");
        let _ = decayed;
    }

    #[test]
    fn decay_watermark_ages_a_plant_on_the_same_steps_as_a_full_scan() {
        let mut config = crate::model::Config::default();
        config.weed_spawn_chance = 0.0;
        let mut cached = crate::init::new_game(4, config);
        let actions = pass_action();
        // A plant already past its last harvest: it loses one unit every second
        // step and turns into a weed when the last one is gone.  Kept inside
        // one day, so the unwatered rule cannot claim it first.
        let mut plant = make_plant(Crop::Wheat, 0, cached.config.turns_per_day);
        plant.yield_units = 2;
        plant.consecutive_unwatered = 0;
        plant.max_lifespan_step = 4;
        cached.farms[0].tiles[0] = Tile::Plant(plant);
        let mut scanned = cached.clone();
        let mut yields = Vec::new();
        for _ in 0..12 {
            step(&mut cached, &actions);
            for farm in &mut scanned.farms {
                farm.decay_from = 0;
            }
            step(&mut scanned, &actions);
            assert_eq!(cached, scanned, "step {}", cached.step);
            yields.push(match &cached.farms[0].tiles[0] {
                Tile::Plant(plant) => plant.yield_units as i32,
                Tile::Weed => -1,
                _ => -2,
            });
        }
        // Steps 0-3: untouched.  Step 4: 2 -> 1.  Step 6: 1 -> 0, so a weed.
        assert_eq!(yields, vec![2, 2, 2, 2, 1, 1, -1, -1, -1, -1, -1, -1]);
    }

    #[test]
    fn decay_watermark_is_replaced_together_with_the_farm() {
        // `rollout.rs` rebuilds a state from a live observation by replacing a
        // whole `Farm`; the new tiles may decay earlier than the watermark the
        // replayed board left behind.  The mark lives inside `Farm`, so the
        // replacement carries its own (`decay_from: 0`, scan) with it.
        let mut config = crate::model::Config::default();
        config.weed_spawn_chance = 0.0;
        let mut state = crate::init::new_game(3, config);
        let actions = pass_action();
        for _ in 0..30 {
            step(&mut state, &actions);
        }
        // A stale mark that would skip every remaining scan if it survived.
        state.farms[0].decay_from = u32::MAX;
        let mut plant = make_plant(Crop::Wheat, 0, state.config.turns_per_day);
        plant.max_lifespan_step = state.step as i32;
        plant.yield_units = 1;
        let mut tiles = state.farms[0].tiles.clone();
        tiles[0] = Tile::Plant(plant);
        // Exactly the shape of the `rollout.rs` reconstruction.
        state.farms[0] = Farm {
            money: state.farms[0].money,
            tiles,
            farmer: state.farms[0].farmer,
            hands: state.farms[0].hands.clone(),
            unlocked_quadrants: state.farms[0].unlocked_quadrants,
            hires_today: state.farms[0].hires_today,
            decay_from: 0,
        };
        step(&mut state, &actions);
        assert_eq!(state.farms[0].tiles[0], Tile::Weed);
    }

    #[test]
    fn decay_watermark_travels_with_a_cloned_farm_and_is_not_serialized() {
        let mut config = crate::model::Config::default();
        config.weed_spawn_chance = 0.0;
        let mut state = crate::init::new_game(3, config);
        let actions = pass_action();
        let mut plant = make_plant(Crop::Wheat, 0, state.config.turns_per_day);
        plant.max_lifespan_step = 200;
        plant.yield_units = 3;
        state.farms[0].tiles[0] = Tile::Plant(plant);
        step(&mut state, &actions);
        // A live mark: the plant cannot decay before step 200.
        assert_eq!(state.farms[0].decay_from, 200);

        let copy = state.clone();
        assert_eq!(copy.farms[0].decay_from, state.farms[0].decay_from);
        assert_eq!(copy.farms[1].decay_from, state.farms[1].decay_from);

        // The mark is a cache, not game state: it is not compared and not saved.
        let mut other = state.clone();
        other.farms[0].decay_from = 7;
        assert_eq!(other, state);
        let restored: GameState =
            serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
        assert_eq!(restored.farms[0].decay_from, 0);
        assert_eq!(restored, state);
    }

    #[test]
    fn python_round_is_ties_to_even() {
        assert_eq!(py_round_nonnegative(2.5), 2.0);
        assert_eq!(py_round_nonnegative(3.5), 4.0);
    }

    #[test]
    fn shed_overflow_uses_python_inventory_insertion_order() {
        let mut state = crate::init::new_game(0, crate::model::Config::default());
        let private = &mut state.private[0];
        inventory_add(private, 0, Item::Product(Product::Milk), 1);
        inventory_add(private, 0, Item::Product(Product::Wheat), 1);

        drop_inventories(private, 1);

        assert_eq!(private.shed[Product::Milk.index()], 1);
        assert_eq!(private.shed[Product::Wheat.index()], 0);
        assert!(private.inventory_orders[0].is_empty());
    }

    #[test]
    fn removing_and_readding_an_item_moves_it_to_the_end() {
        let mut state = crate::init::new_game(0, crate::model::Config::default());
        let private = &mut state.private[0];
        inventory_add(private, 0, Item::Product(Product::Milk), 1);
        inventory_add(private, 0, Item::Product(Product::Wheat), 1);
        assert!(inventory_take(private, 0, Item::Product(Product::Milk), 1));
        inventory_add(private, 0, Item::Product(Product::Milk), 1);

        drop_inventories(private, 1);

        assert_eq!(private.shed[Product::Wheat.index()], 1);
        assert_eq!(private.shed[Product::Milk.index()], 0);
    }

    #[test]
    fn plant_requests_are_rejected_atomically() {
        let mut state = crate::init::new_game(0, crate::model::Config::default());
        state.private[0].seeds[Crop::Wheat.index()] = 1;
        state.farms[0]
            .hands
            .push(crate::model::Position { x: 3, y: 4 });
        state.private[0].inventories.push([0; ITEM_COUNT]);
        let mut actions = pass_action();
        actions[0].farmer = UnitAction::Plant(Crop::Wheat);
        actions[0].hands = vec![UnitAction::Plant(Crop::Wheat)];

        step(&mut state, &actions);

        assert_eq!(state.private[0].seeds[Crop::Wheat.index()], 1);
        assert!(
            state.farms[0]
                .tiles
                .iter()
                .all(|tile| !matches!(tile, Tile::Plant(_)))
        );
    }

    #[test]
    fn market_quotes_both_players_before_committing_each_unit() {
        let mut state = crate::init::new_game(0, crate::model::Config::default());
        state.private[0].shed[Product::Wheat.index()] = 1;
        state.private[1].shed[Product::Wheat.index()] = 1;
        let mut actions = pass_action();
        actions[0].market = vec![MarketOrder::Sell(Item::Product(Product::Wheat), 1)];
        actions[1].market = vec![MarketOrder::Sell(Item::Product(Product::Wheat), 1)];

        step(&mut state, &actions);

        assert_eq!(state.farms[0].money, 3_025.0);
        assert_eq!(state.farms[1].money, 3_025.0);
        // Includes the town-center consumption at step zero.
        assert_eq!(state.market.inventory[Product::Wheat.index()], 10_001);
    }

    #[test]
    fn market_trace_preserves_step_semantics_and_records_realized_sale() {
        let mut plain = crate::init::new_game(17, crate::model::Config::default());
        plain.private[0].shed[Product::Wheat.index()] = 3;
        let mut traced = plain.clone();
        let mut actions = pass_action();
        actions[0].market = vec![
            MarketOrder::Sell(Item::Product(Product::Wheat), 2),
            MarketOrder::Hire,
        ];

        step(&mut plain, &actions);
        let mut trace = Vec::new();
        step_with_market_trace(&mut traced, &actions, &mut trace);

        assert_eq!(plain, traced);
        assert_eq!(trace.len(), 2);
        assert_eq!(trace[0].player, 0);
        assert_eq!(trace[0].order_index, 0);
        assert_eq!(trace[0].shed_delta[Product::Wheat.index()], -2);
        assert!(trace[0].money_delta > 0.0);
        assert_eq!(trace[1].hands_delta, 1);
        assert_eq!(trace[1].money_delta, -1.0);
    }

    #[test]
    fn full_trace_preserves_step_semantics_and_records_unit_success() {
        let mut plain = crate::init::new_game(23, crate::model::Config::default());
        plain.farms[0].farmer = crate::model::Position { x: 0, y: 0 };
        plain.private[0].seeds[Crop::Strawberry.index()] = 1;
        let mut traced = plain.clone();
        let mut actions = pass_action();
        actions[0].farmer = UnitAction::Plant(Crop::Strawberry);

        step(&mut plain, &actions);
        let mut unit_trace = Vec::new();
        let mut market_trace = Vec::new();
        step_with_trace(&mut traced, &actions, &mut unit_trace, &mut market_trace);

        assert_eq!(plain, traced);
        assert!(market_trace.is_empty());
        assert_eq!(unit_trace.len(), 2);
        assert_eq!(unit_trace[0].action, UnitAction::Plant(Crop::Strawberry));
        assert!(unit_trace[0].success);
        assert!(matches!(
            unit_trace[0].tile_after,
            Some(Tile::Plant(PlantState {
                crop: Crop::Strawberry,
                ..
            }))
        ));
    }

    #[test]
    fn land_unlocks_follow_bitmask_order() {
        let mut state = crate::init::new_game(0, crate::model::Config::default());
        let mut actions = pass_action();
        actions[0].market = vec![MarketOrder::BuyLand];
        step(&mut state, &actions);
        assert_eq!(state.farms[0].unlocked_quadrants, 0b0011);
        actions[0].market = vec![MarketOrder::BuyLand];
        step(&mut state, &actions);
        assert_eq!(state.farms[0].unlocked_quadrants, 0b0111);
    }
}
