// Derived from kaggle-environments 1.32.7, kaggle_environments/envs/kaggriculture/kaggriculture.py
// (published by Kaggle under the Apache License 2.0; see LICENSES/Apache-2.0.txt and NOTICE at the repository root):
// game initialisation ported from the Python interpreter and modified (translated to Rust, restructured for batched rollouts).
//! Construction of the compact Rust state corresponding to Python `_initialize`.

use crate::constants::{QUADRANT_NE, QUADRANT_NW, QUADRANT_SE, QUADRANT_SW};
use crate::model::{
    CROP_COUNT, Config, Farm, GameState, ITEM_COUNT, Market, PLAYER_COUNT, Position, PrivateState,
    Status, Tile, Town,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quadrant {
    NorthWest,
    NorthEast,
    SouthWest,
    SouthEast,
}

impl Quadrant {
    pub const fn mask(self) -> u8 {
        match self {
            Self::NorthWest => QUADRANT_NW,
            Self::NorthEast => QUADRANT_NE,
            Self::SouthWest => QUADRANT_SW,
            Self::SouthEast => QUADRANT_SE,
        }
    }
}

pub const fn quadrant_of(x: u16, y: u16, board_size: u16) -> Quadrant {
    let half = board_size / 2;
    match (y < half, x < half) {
        (true, true) => Quadrant::NorthWest,
        (true, false) => Quadrant::NorthEast,
        (false, true) => Quadrant::SouthWest,
        (false, false) => Quadrant::SouthEast,
    }
}

pub const fn tile_index(x: u16, y: u16, board_size: u16) -> usize {
    y as usize * board_size as usize + x as usize
}

/// Four inner-corner positions in the same NW, NE, SW, SE order as Python.
pub const fn shed_access_tiles(board_size: u16) -> [Position; 4] {
    let half = board_size / 2;
    [
        Position {
            x: half - 1,
            y: half - 1,
        },
        Position {
            x: half,
            y: half - 1,
        },
        Position {
            x: half - 1,
            y: half,
        },
        Position { x: half, y: half },
    ]
}

pub const fn is_shed_adjacent(position: Position, board_size: u16) -> bool {
    let tiles = shed_access_tiles(board_size);
    let mut index = 0;
    while index < tiles.len() {
        if tiles[index].x == position.x && tiles[index].y == position.y {
            return true;
        }
        index += 1;
    }
    false
}

pub const fn default_spawn(board_size: u16) -> Position {
    // For every supported board (size >= 4), the first inner-corner is in NW.
    shed_access_tiles(board_size)[0]
}

fn new_farm(config: &Config) -> Farm {
    let board_size = config.board_size;
    let mut tiles = Vec::with_capacity(board_size as usize * board_size as usize);
    for y in 0..board_size {
        for x in 0..board_size {
            tiles.push(if quadrant_of(x, y, board_size) == Quadrant::NorthWest {
                Tile::Empty
            } else {
                Tile::Locked
            });
        }
    }
    Farm {
        money: f64::from(config.starting_money),
        tiles,
        farmer: default_spawn(board_size),
        hands: Vec::new(),
        unlocked_quadrants: QUADRANT_NW,
        hires_today: 0,
        decay_from: 0,
    }
}

fn new_private() -> PrivateState {
    PrivateState {
        shed: [0; ITEM_COUNT],
        seeds: [0; CROP_COUNT],
        inventories: vec![[0; ITEM_COUNT]],
        inventory_orders: vec![Vec::new()],
    }
}

fn new_market(config: &Config) -> Market {
    let inventory = std::array::from_fn(|index| config.market_params[index].initial_inventory);
    // Python exposes the raw base before the first refresh, including a
    // fractional custom base. At the default I0 this equals market_price.
    let prices = std::array::from_fn(|index| config.market_params[index].base);
    Market {
        inventory,
        prices,
        params: config.market_params,
    }
}

pub fn new_game(seed: i64, config: Config) -> GameState {
    let farms = std::array::from_fn(|_| new_farm(&config));
    let private = std::array::from_fn(|_| new_private());
    let market = new_market(&config);
    GameState {
        config,
        seed,
        step: 0,
        day: 0,
        hour: 0,
        farms,
        private,
        market,
        town: Town::default(),
        statuses: [Status::Active; PLAYER_COUNT],
        rewards: [None; PLAYER_COUNT],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{MARKET_INITIAL_INVENTORY, MARKET_PARAMS};
    use crate::model::{PRODUCT_COUNT, Product};

    #[test]
    fn quadrants_and_shed_positions_match_python() {
        assert_eq!(quadrant_of(0, 0, 10), Quadrant::NorthWest);
        assert_eq!(quadrant_of(5, 0, 10), Quadrant::NorthEast);
        assert_eq!(quadrant_of(0, 5, 10), Quadrant::SouthWest);
        assert_eq!(quadrant_of(5, 5, 10), Quadrant::SouthEast);
        assert_eq!(
            shed_access_tiles(10),
            [
                Position { x: 4, y: 4 },
                Position { x: 5, y: 4 },
                Position { x: 4, y: 5 },
                Position { x: 5, y: 5 },
            ]
        );
        assert_eq!(default_spawn(10), Position { x: 4, y: 4 });
    }

    #[test]
    fn default_game_matches_python_initialize() {
        let game = new_game(42, Config::default());
        assert_eq!(game.seed, 42);
        assert_eq!((game.step, game.day, game.hour), (0, 0, 0));
        assert_eq!(game.statuses, [Status::Active, Status::Active]);
        assert_eq!(game.rewards, [None, None]);
        assert!(game.town.unlocked_shops.is_empty());
        for farm in &game.farms {
            assert_eq!(farm.money, 3_000.0);
            assert_eq!(farm.tiles.len(), 100);
            assert_eq!(
                farm.tiles
                    .iter()
                    .filter(|tile| **tile == Tile::Empty)
                    .count(),
                25
            );
            assert_eq!(
                farm.tiles
                    .iter()
                    .filter(|tile| **tile == Tile::Locked)
                    .count(),
                75
            );
            assert_eq!(farm.farmer, Position { x: 4, y: 4 });
            assert_eq!(farm.unlocked_quadrants, QUADRANT_NW);
        }
        for private in &game.private {
            assert_eq!(private.shed, [0; ITEM_COUNT]);
            assert_eq!(private.seeds, [0; CROP_COUNT]);
            assert_eq!(private.inventories, vec![[0; ITEM_COUNT]]);
            assert_eq!(private.inventory_orders, vec![Vec::new()]);
        }
        assert_eq!(
            game.market.inventory,
            [MARKET_INITIAL_INVENTORY; PRODUCT_COUNT]
        );
        assert_eq!(
            game.market.prices,
            [25.0, 35.0, 60.0, 120.0, 250.0, 50.0, 160.0, 200.0, 100.0]
        );
        assert_eq!(game.market.params, MARKET_PARAMS);
    }

    #[test]
    fn custom_board_and_market_params_are_used() {
        let mut config = Config {
            board_size: 12,
            starting_money: 77,
            ..Config::default()
        };
        config.market_params[Product::Wheat.index()].initial_inventory = 123;
        config.market_params[Product::Wheat.index()].base = 25.5;
        let game = new_game(-9, config);
        assert_eq!(game.farms[0].tiles.len(), 144);
        assert_eq!(game.farms[0].farmer, Position { x: 5, y: 5 });
        assert_eq!(game.farms[0].money, 77.0);
        assert_eq!(game.market.inventory[Product::Wheat.index()], 123);
        assert_eq!(game.market.prices[Product::Wheat.index()], 25.5);
    }
}
