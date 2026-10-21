// Derived from kaggle-environments 1.32.7, kaggle_environments/envs/kaggriculture/kaggriculture.py
// (published by Kaggle under the Apache License 2.0; see LICENSES/Apache-2.0.txt and NOTICE at the repository root):
// game data constants copied from the Python interpreter and modified (translated to Rust, restructured for batched rollouts).
//! Static game data copied from the official `kaggriculture.py` interpreter.

use crate::model::{
    ANIMAL_COUNT, Animal, CROP_COUNT, Config, Crop, MarketParam, PRODUCT_COUNT, PriceShape,
    Product, Shop, StructureKind,
};

pub const MARKET_INITIAL_INVENTORY: i32 = 10_000;
pub const PRICE_FLOOR: u32 = 1;
pub const HINGE_GAIN: f64 = 8.0;
pub const LAND_PRICES: [u32; 3] = [1_000, 2_000, 4_000];
pub const MAX_SHOP_INSTANCES: usize = 8;

pub const QUADRANT_NW: u8 = 1 << 0;
pub const QUADRANT_NE: u8 = 1 << 1;
pub const QUADRANT_SW: u8 = 1 << 2;
pub const QUADRANT_SE: u8 = 1 << 3;
pub const LAND_ORDER: [u8; 3] = [QUADRANT_NE, QUADRANT_SW, QUADRANT_SE];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CropData {
    pub seed_cost: u32,
    pub first_yield_day: u32,
    pub max_yield_day: u32,
    pub interval: u32,
    pub max_yield: u32,
    pub ongoing: bool,
}

pub const CROP_DATA: [CropData; CROP_COUNT] = [
    CropData {
        seed_cost: 10,
        first_yield_day: 2,
        max_yield_day: 4,
        interval: 0,
        max_yield: 6,
        ongoing: false,
    },
    CropData {
        seed_cost: 20,
        first_yield_day: 2,
        max_yield_day: 3,
        interval: 0,
        max_yield: 4,
        ongoing: false,
    },
    CropData {
        seed_cost: 50,
        first_yield_day: 8,
        max_yield_day: 8,
        interval: 1,
        max_yield: 4,
        ongoing: true,
    },
    CropData {
        seed_cost: 100,
        first_yield_day: 10,
        max_yield_day: 10,
        interval: 2,
        max_yield: 4,
        ongoing: true,
    },
    CropData {
        seed_cost: 80,
        first_yield_day: 10,
        max_yield_day: 12,
        interval: 0,
        max_yield: 6,
        ongoing: false,
    },
];

pub const fn crop_data(crop: Crop) -> &'static CropData {
    &CROP_DATA[crop.index()]
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnimalData {
    pub cost: u32,
    pub structure: StructureKind,
    pub first_yield_day: u32,
    pub interval: u32,
    pub max_held: u32,
    pub product: Product,
}

pub const ANIMAL_DATA: [AnimalData; ANIMAL_COUNT] = [
    AnimalData {
        cost: 300,
        structure: StructureKind::Coop,
        first_yield_day: 4,
        interval: 1,
        max_held: 4,
        product: Product::Egg,
    },
    AnimalData {
        cost: 400,
        structure: StructureKind::Pasture,
        first_yield_day: 8,
        interval: 2,
        max_held: 6,
        product: Product::Milk,
    },
    AnimalData {
        cost: 500,
        structure: StructureKind::Pasture,
        first_yield_day: 6,
        interval: 3,
        max_held: 6,
        product: Product::Wool,
    },
];

pub const fn animal_data(animal: Animal) -> &'static AnimalData {
    &ANIMAL_DATA[animal.index()]
}

pub const MARKET_PARAMS: [MarketParam; PRODUCT_COUNT] = [
    market_param(
        25.0,
        400.0,
        PriceShape::SquareRoot,
        0.80,
        PriceShape::Log,
        0.20,
    ),
    market_param(
        35.0,
        450.0,
        PriceShape::Hinge,
        1.00,
        PriceShape::SquareRoot,
        0.70,
    ),
    market_param(
        60.0,
        200.0,
        PriceShape::Hinge,
        0.40,
        PriceShape::SquareRoot,
        0.60,
    ),
    market_param(
        120.0,
        100.0,
        PriceShape::SquareRoot,
        0.70,
        PriceShape::Linear,
        1.60,
    ),
    market_param(
        250.0,
        300.0,
        PriceShape::Log,
        0.20,
        PriceShape::Square,
        3.60,
    ),
    market_param(50.0, 332.0, PriceShape::Hinge, 0.40, PriceShape::Log, 0.20),
    market_param(
        160.0,
        122.0,
        PriceShape::SquareRoot,
        0.60,
        PriceShape::Linear,
        1.60,
    ),
    market_param(
        200.0,
        105.0,
        PriceShape::Log,
        0.20,
        PriceShape::Square,
        3.20,
    ),
    market_param(
        100.0,
        200.0,
        PriceShape::Linear,
        0.40,
        PriceShape::Linear,
        0.40,
    ),
];

const fn market_param(
    base: f64,
    scale: f64,
    below_shape: PriceShape,
    below_target: f64,
    above_shape: PriceShape,
    above_target: f64,
) -> MarketParam {
    MarketParam {
        base,
        initial_inventory: MARKET_INITIAL_INVENTORY,
        scale,
        below_shape,
        below_target,
        above_shape,
        above_target,
    }
}

pub const BAKERY_PRODUCTS: &[Product] = &[Product::Egg, Product::Wheat];
pub const PIZZA_SHOP_PRODUCTS: &[Product] = &[Product::Milk, Product::Tomato, Product::Wheat];
pub const BRUNCH_SPOT_PRODUCTS: &[Product] = &[Product::Egg, Product::Wheat, Product::Strawberry];
pub const YARN_STORE_PRODUCTS: &[Product] = &[Product::Wool];
pub const ICE_CREAM_SHOP_PRODUCTS: &[Product] =
    &[Product::Strawberry, Product::Milk, Product::Wheat];
pub const PET_CAFE_PRODUCTS: &[Product] = &[Product::Carrot];
pub const SMOOTHIE_SHOP_PRODUCTS: &[Product] = &[Product::Strawberry, Product::Milk];
pub const FARMERS_MARKET_PRODUCTS: &[Product] = &[
    Product::Wheat,
    Product::Carrot,
    Product::Tomato,
    Product::Strawberry,
];
pub const TOWN_CENTER_PRODUCTS: &[Product] = &[
    Product::Wheat,
    Product::Carrot,
    Product::Tomato,
    Product::Strawberry,
    Product::Melon,
    Product::Egg,
    Product::Milk,
    Product::Wool,
];

/// Alphabetical order used by Python's `rng.choice(sorted(SHOPS))`.
pub const SHOP_DRAW_ORDER: [Shop; 8] = [
    Shop::Bakery,
    Shop::BrunchSpot,
    Shop::FarmersMarket,
    Shop::IceCreamShop,
    Shop::PetCafe,
    Shop::PizzaShop,
    Shop::SmoothieShop,
    Shop::YarnStore,
];

pub const fn shop_products(shop: Shop) -> &'static [Product] {
    match shop {
        Shop::Bakery => BAKERY_PRODUCTS,
        Shop::PizzaShop => PIZZA_SHOP_PRODUCTS,
        Shop::BrunchSpot => BRUNCH_SPOT_PRODUCTS,
        Shop::YarnStore => YARN_STORE_PRODUCTS,
        Shop::IceCreamShop => ICE_CREAM_SHOP_PRODUCTS,
        Shop::PetCafe => PET_CAFE_PRODUCTS,
        Shop::SmoothieShop => SMOOTHIE_SHOP_PRODUCTS,
        Shop::FarmersMarket => FARMERS_MARKET_PRODUCTS,
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            episode_steps: 720,
            board_size: 10,
            starting_money: 3_000,
            max_market_orders_per_turn: 10,
            turns_per_day: 24,
            shed_capacity: 100,
            weed_spawn_chance: 0.005,
            town_shop_unlock_interval: 3,
            town_shop_sell_interval: 4,
            town_center_sell_interval: 24,
            farm_hand_cost_multiplier: 1,
            market_params: MARKET_PARAMS,
            decoupled_rng: false,
        }
    }
}

fn shape(kind: PriceShape, x: f64, scale: f64) -> f64 {
    let x = x.max(0.0);
    match kind {
        PriceShape::Linear => x,
        PriceShape::Square => x * x,
        PriceShape::SquareRoot => x.sqrt(),
        // Match Python's math.log(1.0 + x), rather than the subtly different log1p.
        PriceShape::Log => (1.0 + x).ln(),
        PriceShape::Log10 => (1.0 + x).log10(),
        PriceShape::Hinge if scale > 0.0 => {
            let u = x / scale;
            u + HINGE_GAIN * (u - 1.0).max(0.0).powi(2)
        }
        PriceShape::Hinge => x,
    }
}

/// Equivalent to Python 3's `round()` for the finite, non-negative prices used here.
fn round_ties_even(value: f64) -> f64 {
    let floor = value.floor();
    let fraction = value - floor;
    if fraction < 0.5 {
        floor
    } else if fraction > 0.5 {
        floor + 1.0
    } else if floor % 2.0 == 0.0 {
        floor
    } else {
        floor + 1.0
    }
}

pub fn market_price(
    product: Product,
    inventory: i32,
    params: &[MarketParam; PRODUCT_COUNT],
) -> f64 {
    let p = params[product.index()];
    let distance = (inventory - p.initial_inventory).unsigned_abs() as f64;
    let (kind, target, sign) = if inventory < p.initial_inventory {
        (p.below_shape, p.below_target, 1.0)
    } else {
        (p.above_shape, p.above_target, -1.0)
    };
    let amplitude = target * p.base / shape(kind, p.scale, p.scale);
    let price = p.base + sign * amplitude * shape(kind, distance, p.scale);
    round_ties_even(price).max(PRICE_FLOOR as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_specification() {
        let config = Config::default();
        assert_eq!(config.episode_steps, 720);
        assert_eq!(config.board_size, 10);
        assert_eq!(config.starting_money, 3_000);
        assert_eq!(config.turns_per_day, 24);
        assert_eq!(config.market_params, MARKET_PARAMS);
    }

    #[test]
    fn prices_match_official_python_reference_points() {
        let points = [0, 9_900, 9_999, 10_000, 10_001, 10_100, 20_000];
        let wheat = [125, 35, 26, 25, 24, 21, 17];
        let carrot = [126_920, 43, 35, 35, 34, 23, 1];
        let melon = [331, 290, 256, 250, 250, 150, 1];
        for (inventory, expected) in points.into_iter().zip(wheat) {
            assert_eq!(
                market_price(Product::Wheat, inventory, &MARKET_PARAMS),
                f64::from(expected)
            );
        }
        for (inventory, expected) in points.into_iter().zip(carrot) {
            assert_eq!(
                market_price(Product::Carrot, inventory, &MARKET_PARAMS),
                f64::from(expected)
            );
        }
        for (inventory, expected) in points.into_iter().zip(melon) {
            assert_eq!(
                market_price(Product::Melon, inventory, &MARKET_PARAMS),
                f64::from(expected)
            );
        }
    }

    #[test]
    fn ties_round_to_even_like_python() {
        assert_eq!(round_ties_even(2.5), 2.0);
        assert_eq!(round_ties_even(3.5), 4.0);
    }
}
