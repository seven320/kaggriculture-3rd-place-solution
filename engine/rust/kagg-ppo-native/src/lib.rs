//! Isolated batch FFI: identical Rust game rules and the visible features defined in masspeaks' BC kit.
use std::{ffi::{CString, c_char}, slice};
use kaggriculture_engine::{new_game, step};
use kaggriculture_engine::model::*;
use kaggriculture_python_bridge::observation_value;

const NU: usize = 20;
const NC: usize = 44;
const ACTION_STRIDE: usize = 92;
const META: usize = 18;

pub struct Pool { games: Vec<GameState>, debug: CString }

fn tile_features(tile: &Tile, day: u32, clock: u32) -> [f32; 13] {
    let mut v = [0f64; 13]; v[10]=-1.; v[11]=-1.; v[12]=-1.;
    match tile {
        Tile::Empty => {},
        Tile::Locked => v[0]=1.,
        Tile::Weed => v[0]=2.,
        Tile::Plant(p) => {
            v[0]=5.; v[1]=p.crop.index() as f64; v[2]=p.watered_today as u8 as f64;
            v[4]=p.yield_units as f64; v[8]=p.consecutive_unwatered as f64;
            v[10]=(day as f64-p.planted_day as f64)/30.;
            if p.max_lifespan_step>=0 { v[11]=(p.max_lifespan_step as f64-clock as f64)/720.; }
            if p.fertilized_until_day>=0 { v[12]=(p.fertilized_until_day as f64-day as f64)/30.; }
        },
        Tile::Structure(s) => {
            v[0]=if s.kind==StructureKind::Coop {3.} else {4.};
            if let Some(a)=&s.animal {
                v[1]=(9+a.animal.index()) as f64; v[3]=a.fed_today as u8 as f64;
                v[4]=a.yield_units as f64; v[5]=a.fertilizer_available as u8 as f64;
                v[6]=1.; v[7]=a.cared_today as u8 as f64;
                v[8]=a.consecutive_unfed as f64; v[9]=a.pending_care_bonus as f64;
                v[10]=(day as f64-a.placed_day as f64)/30.;
            }
        }
    }
    v[0]/=5.; v[1]/=12.; v[4]/=6.; v[8]/=4.; v[9]/=4.;
    v.map(|x| x as f32)
}

fn positions(farm: &Farm) -> Vec<Position> {
    std::iter::once(farm.farmer).chain(farm.hands.iter().copied()).collect()
}

fn legal_mask(state: &GameState, me: usize, out: &mut [u8]) {
    out.fill(0);
    let farm=&state.farms[me]; let private=&state.private[me];
    let shed_total: u64=private.shed.iter().map(|x| *x as u64).sum();
    for (u,p) in positions(farm).iter().take(NU).enumerate() {
        let row=&mut out[u*NC..(u+1)*NC];
        let iv=private.inventories.get(u).copied().unwrap_or([0;ITEM_COUNT]);
        let tile=&farm.tiles[p.y as usize*10+p.x as usize];
        let adj=(p.x==4 || p.x==5) && (p.y==4 || p.y==5);
        row[0]=1; row[1]=(p.y>0) as u8; row[2]=(p.y<9) as u8;
        row[3]=(p.x<9) as u8; row[4]=(p.x>0) as u8;
        if adj && iv.iter().any(|x| *x>0) { row[14]=1; }
        for i in 0..ITEM_COUNT {
            if adj && private.shed[i]>0 { row[20+i]=1; }
            if i>=9 {
                let want=if i==9 {StructureKind::Coop} else {StructureKind::Pasture};
                if let Tile::Structure(s)=tile {
                    if s.kind==want && s.animal.is_none() {
                        row[32+i]=(iv[i]>=1) as u8; continue;
                    }
                }
            }
            if adj && iv[i]>0 && shed_total<100 {row[32+i]=1;}
        }
        match tile {
            Tile::Locked => continue,
            Tile::Empty => {
                for c in 0..5 {row[15+c]=(private.seeds[c]>0) as u8;}
                row[9]=1; row[10]=1;
            },
            Tile::Weed => row[8]=1,
            Tile::Plant(p) => {
                row[5]=(!p.watered_today) as u8;
                row[6]=(p.yield_units>0 && state.day as i64-p.planted_day as i64 >= [2,2,8,10,10][p.crop.index()]) as u8;
                row[7]=(iv[8]>=1) as u8; row[8]=1;
            },
            Tile::Structure(s) => {
                if let Some(a)=&s.animal {
                    row[6]=(a.yield_units>0) as u8;
                    row[11]=(!a.fed_today && iv[0]>=1) as u8;
                    row[12]=a.fertilizer_available as u8; row[13]=(!a.cared_today) as u8;
                } else {row[8]=1;}
            }
        }
    }
}

fn features(state: &GameState, me: usize, tiles: &mut [f32], ou: &mut [f32], pu: &mut [f32], g: &mut [f32], legal: &mut [u8], meta: &mut [f64]) {
    let op=1-me; let clock=state.day*24+state.hour;
    for (side,p) in [me,op].into_iter().enumerate() {
        for (t,tile) in state.farms[p].tiles.iter().take(100).enumerate() {
            tiles[(side*100+t)*13..(side*100+t+1)*13].copy_from_slice(&tile_features(tile,state.day,clock));
        }
    }
    ou.fill(0.); pu.fill(0.);
    for u in 0..NU {ou[u*15+1]=-1.;ou[u*15+2]=-1.;pu[u*3+1]=-1.;pu[u*3+2]=-1.;}
    for (u,p) in positions(&state.farms[me]).into_iter().take(NU).enumerate() {
        ou[u*15]=1.; ou[u*15+1]=p.x as f32; ou[u*15+2]=p.y as f32;
    }
    for (u,inv) in state.private[me].inventories.iter().take(NU).enumerate() {
        for i in 0..12 {ou[u*15+3+i]=inv[i] as f32;}
    }
    for (u,p) in positions(&state.farms[op]).into_iter().take(NU).enumerate() {
        pu[u*3]=1.; pu[u*3+1]=p.x as f32; pu[u*3+2]=p.y as f32;
    }
    let private=&state.private[me];
    for i in 0..9 {
        g[i]=(state.market.inventory[i] as f64/10000.) as f32;
        g[9+i]=(state.market.prices[i]/250.) as f32;
        g[18+i]=(private.shed[i] as f64/100.) as f32;
    }
    for i in 0..5 {g[27+i]=(private.seeds[i] as f64/20.) as f32;}
    g[32]=(state.farms[me].money/50000.) as f32;
    g[33]=(state.farms[op].money/50000.) as f32;
    g[34]=(clock as f64/720.) as f32; g[35]=(state.hour as f64/24.) as f32; g[36]=(state.day as f64/30.) as f32;
    let shed_total: u64=private.shed.iter().map(|x| *x as u64).sum();
    for (side,p) in [me,op].into_iter().enumerate() {
        let f=&state.farms[p]; let offset=37+side*5;
        g[offset]=(f.money/50000.) as f32; g[offset+1]=((1+f.hands.len()) as f64/13.) as f32;
        g[offset+2]=((f.unlocked_quadrants & 15).count_ones() as f64/4.) as f32;
        g[offset+3]=(f.hires_today as f64/13.) as f32;
        g[offset+4]=if side==0 {(shed_total as f64/100.) as f32} else {-1.};
    }
    g[47..63].fill(-1.);
    for (i,s) in state.town.unlocked_shops.iter().take(16).enumerate() {
        g[47+i]=match s {Shop::Bakery=>0.,Shop::BrunchSpot=>1.,Shop::FarmersMarket=>2.,Shop::IceCreamShop=>3.,Shop::PetCafe=>4.,Shop::PizzaShop=>5.,Shop::SmoothieShop=>6.,Shop::YarnStore=>7.};
    }
    for i in 0..12 {g[63+i]=private.shed[i] as f32;}
    legal_mask(state,me,legal);
    meta[0]=state.step as f64; meta[1]=state.day as f64;meta[2]=state.hour as f64;meta[3]=me as f64;
    meta[4]=state.farms[0].money;meta[5]=state.farms[1].money;
    for i in 0..12 {meta[6+i]=private.shed[i] as f64;}
}

fn item(i: usize) -> Option<Item> {
    if i<9 {Some(Item::Product(Product::ALL[i]))} else {Animal::ALL.get(i.checked_sub(9)?).copied().map(Item::Animal)}
}

fn unit(c: i64, q: u32) -> Option<UnitAction> {
    Some(match c {
        0=>UnitAction::Pass,1=>UnitAction::Move(Direction::North),2=>UnitAction::Move(Direction::South),
        3=>UnitAction::Move(Direction::East),4=>UnitAction::Move(Direction::West),5=>UnitAction::Water,
        6=>UnitAction::Harvest,7=>UnitAction::Fertilize,8=>UnitAction::Dig,9=>UnitAction::BuildCoop,
        10=>UnitAction::BuildPasture,11=>UnitAction::Feed,12=>UnitAction::CollectFertilizer,13=>UnitAction::Care,14=>UnitAction::Drop,
        15..=19=>UnitAction::Plant(Crop::ALL[(c-15) as usize]),
        20..=31=>UnitAction::Pickup(item((c-20) as usize)?,q),
        32..=43=>UnitAction::Place(item((c-32) as usize)?,q),_=>return None,
    })
}

fn action(raw: &[i64]) -> Option<PlayerAction> {
    let hands=usize::try_from(raw[0]).ok()?; let count=usize::try_from(raw[1]).ok()?;
    if hands>=NU || count>10 {return None;}
    let mut units=Vec::with_capacity(hands+1);
    for u in 0..=hands {units.push(unit(raw[2+3*u],u32::try_from(raw[4+3*u]).ok()?)?);}
    let mut market=Vec::with_capacity(count);
    for i in 0..count {
        let c=raw[62+3*i];let q=u32::try_from(raw[64+3*i]).ok()?;
        market.push(match c {
            0..=8=>MarketOrder::Sell(item(c as usize)?,q),
            9..=13=>MarketOrder::BuySeed(Crop::ALL[(c-9) as usize],q),
            14=>MarketOrder::BuyProduct(Product::Wheat,q),15=>MarketOrder::BuyProduct(Product::Fertilizer,q),
            16..=18=>MarketOrder::BuyAnimal(Animal::ALL[(c-16) as usize],q),
            19=>MarketOrder::Hire,20=>MarketOrder::BuyLand,
            21=>MarketOrder::Sell(Item::Product(Product::Wheat),0),_=>return None,
        });
    }
    Some(PlayerAction{farmer:units[0],hands:units[1..].to_vec(),market})
}

#[unsafe(no_mangle)]
pub extern "C" fn mp_new(seed: i64, count: usize, episode_steps: u32) -> *mut Pool {
    if count==0 || count>4096 || !(2..=720).contains(&episode_steps) || seed.checked_add(count as i64).is_none() {return std::ptr::null_mut();}
    let games=(0..count).map(|i| {let mut config=Config::default();config.episode_steps=episode_steps;new_game(seed+i as i64,config)}).collect();
    Box::into_raw(Box::new(Pool{games,debug:CString::default()}))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mp_free(pool: *mut Pool) {if !pool.is_null() {drop(unsafe{Box::from_raw(pool)});}}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mp_step(pool: *mut Pool, actions: *const i64, length: usize) -> i32 {
    if pool.is_null() || actions.is_null() {return -1;}
    let pool=unsafe{&mut *pool};
    if length!=pool.games.len()*2*ACTION_STRIDE || pool.games.iter().any(|g|g.statuses[0]==Status::Done) {return -1;}
    let raw=unsafe{slice::from_raw_parts(actions,length)};
    let Some(actions)=raw.chunks_exact(ACTION_STRIDE).map(action).collect::<Option<Vec<_>>>() else {return -2;};
    for (g,pair) in pool.games.iter_mut().zip(actions.chunks_exact(2)) {step(g,&[pair[0].clone(),pair[1].clone()]);}
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mp_features(pool: *const Pool, batch: usize, tiles: *mut f32, ou: *mut f32, pu: *mut f32, g: *mut f32, legal: *mut u8, meta: *mut f64, done: *mut u8) -> i32 {
    if pool.is_null() || tiles.is_null() || ou.is_null() || pu.is_null() || g.is_null() || legal.is_null() || meta.is_null() || done.is_null() {return -1;}
    let pool=unsafe{&*pool};if batch!=pool.games.len()*2 {return -1;}
    let tiles=unsafe{slice::from_raw_parts_mut(tiles,batch*2600)};
    let ou=unsafe{slice::from_raw_parts_mut(ou,batch*300)};
    let pu=unsafe{slice::from_raw_parts_mut(pu,batch*60)};
    let g=unsafe{slice::from_raw_parts_mut(g,batch*75)};
    let legal=unsafe{slice::from_raw_parts_mut(legal,batch*880)};
    let meta=unsafe{slice::from_raw_parts_mut(meta,batch*META)};
    let done=unsafe{slice::from_raw_parts_mut(done,batch)};
    for (i,state) in pool.games.iter().enumerate() {for player in 0..2 {
        let b=i*2+player;
        features(state,player,&mut tiles[b*2600..(b+1)*2600],&mut ou[b*300..(b+1)*300],&mut pu[b*60..(b+1)*60],&mut g[b*75..(b+1)*75],&mut legal[b*880..(b+1)*880],&mut meta[b*META..(b+1)*META]);
        done[b]=(state.statuses[player]==Status::Done) as u8;
    }}
    0
}

/// Validation only; actor never receives hidden state or game seed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mp_debug(pool: *mut Pool) -> *const c_char {
    if pool.is_null() {return std::ptr::null();}
    let pool=unsafe{&mut *pool};
    let data: Vec<_>=pool.games.iter().map(|s|serde_json::json!({"observations":[observation_value(s,0),observation_value(s,1)],"rewards":s.rewards,"statuses":s.statuses})).collect();
    pool.debug=CString::new(serde_json::to_vec(&data).unwrap()).unwrap();pool.debug.as_ptr()
}

const LAND_SCHEMA: i64 = 1;
const LAND_UNITS: usize = 241;

// Stable land-only tile codes; these are not the normalized feature codes.
const LAND_EMPTY: i64 = 0;
const LAND_LOCKED: i64 = 1;
const LAND_WEED: i64 = 2;
const LAND_PLANT: i64 = 3;
const LAND_COOP: i64 = 4;
const LAND_PASTURE: i64 = 5;

fn land_tile(tile: &Tile) -> [i64; 8] {
    let mut v = [LAND_EMPTY, -1, 0, 0, 0, -1, 0, 0];
    match tile {
        Tile::Empty => {},
        Tile::Locked => v[0] = LAND_LOCKED,
        Tile::Weed => v[0] = LAND_WEED,
        Tile::Plant(p) => {
            v[0] = LAND_PLANT; v[1] = p.crop.index() as i64;
            v[2] = i64::from(p.planted_day); v[3] = i64::from(p.yield_units);
            v[4] = i64::from(p.watered_today); v[5] = i64::from(p.fertilized_until_day);
        },
        Tile::Structure(s) => {
            v[0] = if s.kind == StructureKind::Coop { LAND_COOP } else { LAND_PASTURE };
            if let Some(a) = &s.animal {
                v[1] = (9 + a.animal.index()) as i64; v[3] = i64::from(a.yield_units);
                v[6] = i64::from(a.fed_today); v[7] = i64::from(a.fertilizer_available);
            }
        },
    }
    v
}

fn land_range<T>(ptr: *const T, len: usize) -> Result<(usize, usize), i32> {
    if ptr.is_null() || (ptr as usize) % std::mem::align_of::<T>() != 0 { return Err(-1); }
    let bytes = len.checked_mul(std::mem::size_of::<T>()).ok_or(-4)?;
    if bytes > isize::MAX as usize { return Err(-4); }
    Ok((ptr as usize, (ptr as usize).checked_add(bytes).ok_or(-4)?))
}

/// Export requested games in request order, player 0 then player 1 (schema 1).
/// The second axis of farm_meta/tile/position/inventory_* is the owner:
/// player 0's farm 0 and player 1's farm 1, each exactly once, NOT two farms
/// per observation. Money contains both farms. Python adds only the opponent's
/// money to each observation. Item IDs follow native.py ITEMS; absent = -1.
/// Shapes (M = count, U = 241): clock M*3; farm_meta M*2*4; money M*2;
/// tile M*2*100*8; position M*2*U*2; inventory_qty/order M*2*U*12;
/// inventory_order_len M*2*U; shed M*2*12; seeds M*2*5; market_* M*9.
/// farm_meta columns: quadrant mask, hires_today, position count, inventory count.
/// Only actual unit slots are written; count invalidates padding. Order padding
/// within each actual unit is -1. No missing order is synthesized.
/// Returns 0, or -1 pointer/alignment, -2 schema, -3 length, -4 overflow,
/// -5 game ID, -6 capacity/config, -7 inventory order, -8 overlapping buffers,
/// -9 malformed state, -10 caught panic. All validation precedes output writes.
///
/// # Safety
/// pool must be a live mp_new handle; every pointer must cover its declared
/// allocation, and outputs must not alias pool or any of its backing storage.
/// No concurrent step/free/read of this pool or access to output buffers is
/// allowed. Null/alignment/range checks cannot prove allocation liveness.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mp_land_observations(
    pool: *const Pool, schema: i64, ids: *const i64, count: usize,
    clock: *mut i64, clock_len: usize,
    farm_meta: *mut i64, farm_meta_len: usize,
    money: *mut f64, money_len: usize,
    tile: *mut i64, tile_len: usize,
    position: *mut i64, position_len: usize,
    inventory_qty: *mut i64, inventory_qty_len: usize,
    inventory_order: *mut i8, inventory_order_len: usize,
    order_len: *mut u8, order_len_len: usize,
    shed: *mut i64, shed_len: usize,
    seeds: *mut i64, seeds_len: usize,
    market_inventory: *mut i64, market_inventory_len: usize,
    market_prices: *mut f64, market_prices_len: usize,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), i32> {
        if schema != LAND_SCHEMA { return Err(-2); }
        let pool_range = land_range(pool, 1)?;
        let ids_range = land_range(ids, count)?;
        let mut ranges = vec![pool_range, ids_range];
        macro_rules! validate {
            ($ptr:ident, $len:ident, $stride:expr) => {{
                let expected = count.checked_mul($stride).ok_or(-4)?;
                let range = land_range($ptr, $len)?;
                if $len != expected { return Err(-3); }
                ranges.push(range);
            }};
        }
        validate!(clock, clock_len, 3);
        validate!(farm_meta, farm_meta_len, 2*4);
        validate!(money, money_len, 2);
        validate!(tile, tile_len, 2*100*8);
        validate!(position, position_len, 2*LAND_UNITS*2);
        validate!(inventory_qty, inventory_qty_len, 2*LAND_UNITS*12);
        validate!(inventory_order, inventory_order_len, 2*LAND_UNITS*12);
        validate!(order_len, order_len_len, 2*LAND_UNITS);
        validate!(shed, shed_len, 2*12);
        validate!(seeds, seeds_len, 2*5);
        validate!(market_inventory, market_inventory_len, 9);
        validate!(market_prices, market_prices_len, 9);
        for (i, &(start, end)) in ranges.iter().enumerate() {
            if start == end { continue; }
            if ranges[..i].iter().any(|&(a, b)| a != b && start < b && a < end) { return Err(-8); }
        }
        let pool = unsafe { &*pool };
        let ids = unsafe { slice::from_raw_parts(ids, count) };
        for &id in ids {
            let id = usize::try_from(id).map_err(|_| -5)?;
            let state = pool.games.get(id).ok_or(-5)?;
            if state.config.board_size != 10 || state.config.turns_per_day != 24
                || state.config.max_market_orders_per_turn != 10 { return Err(-6); }
            for p in 0..2 {
                let f = &state.farms[p]; let pr = &state.private[p];
                if f.hands.len() >= LAND_UNITS || pr.inventories.len() > LAND_UNITS { return Err(-6); }
                if f.tiles.len() != 100 || f.unlocked_quadrants & !15 != 0 { return Err(-9); }
                if pr.inventory_orders.len() != pr.inventories.len() { return Err(-7); }
                for (inv, order) in pr.inventories.iter().zip(&pr.inventory_orders) {
                    if order.len() > ITEM_COUNT { return Err(-7); }
                    let mut seen = [false; ITEM_COUNT];
                    for item in order {
                        let i = item.index();
                        if i >= ITEM_COUNT || seen[i] || inv[i] == 0 { return Err(-7); }
                        seen[i] = true;
                    }
                    if inv.iter().enumerate().any(|(i, &q)| (q > 0) != seen[i]) { return Err(-7); }
                }
            }
        }
        macro_rules! output {
            ($ptr:ident, $len:ident) => { let $ptr = unsafe { slice::from_raw_parts_mut($ptr, $len) }; };
        }
        output!(clock, clock_len); output!(farm_meta, farm_meta_len); output!(money, money_len);
        output!(tile, tile_len); output!(position, position_len); output!(inventory_qty, inventory_qty_len);
        output!(inventory_order, inventory_order_len); output!(order_len, order_len_len);
        output!(shed, shed_len); output!(seeds, seeds_len);
        output!(market_inventory, market_inventory_len); output!(market_prices, market_prices_len);
        for (g, &id) in ids.iter().enumerate() {
            let s = &pool.games[id as usize];
            clock[g*3..g*3+3].copy_from_slice(&[i64::from(s.step), i64::from(s.day), i64::from(s.hour)]);
            for i in 0..9 { market_inventory[g*9+i] = i64::from(s.market.inventory[i]); }
            market_prices[g*9..g*9+9].copy_from_slice(&s.market.prices);
            for p in 0..2 {
                let b = g*2+p; let f = &s.farms[p]; let pr = &s.private[p];
                farm_meta[b*4..b*4+4].copy_from_slice(&[i64::from(f.unlocked_quadrants),
                    i64::from(f.hires_today), (1+f.hands.len()) as i64, pr.inventories.len() as i64]);
                money[b] = f.money;
                for (t, value) in f.tiles.iter().enumerate() {
                    tile[(b*100+t)*8..(b*100+t+1)*8].copy_from_slice(&land_tile(value));
                }
                for (u, pos) in std::iter::once(&f.farmer).chain(&f.hands).enumerate() {
                    let start = (b*LAND_UNITS+u)*2;
                    position[start..start+2].copy_from_slice(&[i64::from(pos.x), i64::from(pos.y)]);
                }
                for (u, (inv, order)) in pr.inventories.iter().zip(&pr.inventory_orders).enumerate() {
                    let unit = b*LAND_UNITS+u; let start = unit*12;
                    for i in 0..12 { inventory_qty[start+i] = i64::from(inv[i]); }
                    inventory_order[start..start+12].fill(-1);
                    for (i, item) in order.iter().enumerate() { inventory_order[start+i] = item.index() as i8; }
                    order_len[unit] = order.len() as u8;
                }
                for i in 0..12 { shed[b*12+i] = i64::from(pr.shed[i]); }
                for i in 0..5 { seeds[b*5+i] = i64::from(pr.seeds[i]); }
            }
        }
        Ok(())
    })).map_or(-10, |result| result.map_or_else(|code| code, |()| 0))
}
