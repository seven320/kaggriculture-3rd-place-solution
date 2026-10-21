// Python bindings for the bit-exact C++ Kaggriculture engine (1.32.7).
//  - run_tapes:  massively parallel tape-vs-tape evaluation (win/loss gating)
//  - VecMarket:  N parallel episodes whose FIELD actions come from a fixed tape
//                and whose MARKET orders are supplied by the caller each step.
//                This is the RL surface: the field plan is the shared meta line,
//                the market layer is what we learn.
#include "sim.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <thread>
#include <vector>
#include <atomic>
#include <cstring>
#include <cstdlib>

namespace py = pybind11;
using namespace kag;

// A tape is a flat int32 buffer produced by pack_tape() in Python:
//   per step: n_units, (op,arg,n)*n_units, n_orders, (op,item,n)*n_orders
struct Tape {
    std::vector<Action> steps;
};

static Tape unpack(const py::array_t<int32_t>& flat, int n_steps) {
    auto b = flat.unchecked<1>();
    Tape t; t.steps.resize(n_steps);
    size_t i = 0;
    for (int s = 0; s < n_steps; ++s) {
        Action& a = t.steps[s]; a.clear();
        int nu = b(i++); if (nu > MAX_UNITS) nu = MAX_UNITS;
        a.n_units = nu;
        for (int u = 0; u < nu; ++u) {
            a.units[u].op  = (uint8_t)b(i++);
            a.units[u].arg = (uint8_t)b(i++);
            a.units[u].n   = (int16_t)b(i++);
        }
        int no = b(i++); if (no > 16) no = 16;
        a.n_orders = no;
        for (int o = 0; o < no; ++o) {
            a.orders[o].op   = (uint8_t)b(i++);
            a.orders[o].item = (uint8_t)b(i++);
            a.orders[o].n    = b(i++);
        }
    }
    return t;
}

static void run_one(const Tape& A, const Tape& B, uint64_t seed, int steps, double out[2], double weed = -1.0) {
    Config c; c.seed = seed; c.episode_steps = steps;
    if (weed >= 0.0) c.weed_chance = weed;
    Sim sim(c);
    int n = (int)std::min(A.steps.size(), B.steps.size());
    for (int s = 0; s < steps - 1 && !sim.st.done; ++s) {
        const Action& a = A.steps[std::min(s, n - 1)];
        const Action& b = B.steps[std::min(s, n - 1)];
        sim.step(a, b);
    }
    out[0] = sim.st.farms[0].money;
    out[1] = sim.st.farms[1].money;
}

// seats: for each seed we play both seats, so the caller gets a paired result.
static py::array_t<double> run_tapes(py::array_t<int32_t> ta, int na,
                                     py::array_t<int32_t> tb, int nb,
                                     std::vector<uint64_t> seeds,
                                     int steps, int nthreads, double weed) {
    Tape A = unpack(ta, na), B = unpack(tb, nb);
    size_t N = seeds.size() * 2;                 // seed x seat
    py::array_t<double> out({(py::ssize_t)N, (py::ssize_t)2});
    auto o = out.mutable_unchecked<2>();
    std::vector<std::array<double,2>> res(N);
    {
        py::gil_scoped_release rel;
        std::atomic<size_t> next{0};
        int T = nthreads > 0 ? nthreads : (int)std::thread::hardware_concurrency();
        if (T < 1) T = 1;
        std::vector<std::thread> th;
        for (int t = 0; t < T; ++t) th.emplace_back([&]{
            for (;;) {
                size_t i = next.fetch_add(1);
                if (i >= N) break;
                size_t si = i / 2; int seat = (int)(i % 2);
                double r[2];
                if (seat == 0) { run_one(A, B, seeds[si], steps, r, weed); res[i] = {r[0], r[1]}; }
                else           { run_one(B, A, seeds[si], steps, r, weed); res[i] = {r[1], r[0]}; }
            }
        });
        for (auto& x : th) x.join();
    }
    for (size_t i = 0; i < N; ++i) { o(i,0)=res[i][0]; o(i,1)=res[i][1]; }
    return out;
}

// ---------------------------------------------------------------- RL surface
// Field actions from a fixed tape for BOTH seats; market orders supplied per step.
struct VecMarket {
    int N, steps;
    Tape A, B;
    std::vector<Sim> sims;
    std::vector<uint64_t> seeds;
    int cur = 0;
    // which seat the LEARNER controls (0 or 1); the other seat uses tape orders
    int learner_seat = 0;
    bool keep_nonsell = true;

    VecMarket(py::array_t<int32_t> ta, int na, py::array_t<int32_t> tb, int nb,
              std::vector<uint64_t> sds, int st, int seat)
        : N((int)sds.size()), steps(st), A(unpack(ta,na)), B(unpack(tb,nb)),
          seeds(std::move(sds)), learner_seat(seat) {
        sims.reserve(N);
        for (int i = 0; i < N; ++i) { Config c; c.seed = seeds[i]; c.episode_steps = steps; sims.emplace_back(c); }
    }
    void reset() {
        cur = 0;
        for (int i = 0; i < N; ++i) { Config c = sims[i].cfg; sims[i] = Sim(c); }
    }
    int step_index() const { return cur; }
    bool done() const { return cur >= steps - 1; }
    // 自己対戦用(exp176): 両方の座席のユニットの手と市場の注文を渡して1手進める。座席は絶対番号(u0/o0 が座席0)。
    // 両座席とも幅の広い空テープを置き、ユニットの手は op 255 以外を上書きする(step_units と同じ規則)
    template <class U> static void fill_units(Action& a, const U& u, int i) {
        int nu = (int)u.shape(1); if (nu > a.n_units) nu = a.n_units;
        for (int k = 0; k < nu; ++k) {
            int op = u(i,k,0); if (op == 255) continue;
            a.units[k].op = (uint8_t)op; a.units[k].arg = (uint8_t)u(i,k,1); a.units[k].n = (int16_t)u(i,k,2);
        }
    }
    void step_both(py::array_t<int32_t> u0, py::array_t<int32_t> o0, py::array_t<int32_t> u1, py::array_t<int32_t> o1) {
        auto U0 = u0.unchecked<3>(); auto U1 = u1.unchecked<3>();
        auto B0 = o0.unchecked<3>(); auto B1 = o1.unchecked<3>();
        int nA = (int)A.steps.size(), nB = (int)B.steps.size();
        py::gil_scoped_release rel;
        for (int i = 0; i < N; ++i) {
            if (sims[i].st.done) continue;
            Action a0 = A.steps[std::min(cur, nA-1)];
            Action a1 = B.steps[std::min(cur, nB-1)];
            fill_units(a0, U0, i); fill_units(a1, U1, i);
            apply_orders(a0, B0, i); apply_orders(a1, B1, i);
            sims[i].step(a0, a1);
        }
        cur++;
    }
    // 特徴・合法手・所持金を「どちらの座席から見るか」を切り替える(自己対戦で相手側の入力を作る)
    void set_view(int s) { learner_seat = s; }
    int view() const { return learner_seat; }
    // 探索用: sims[i] = sims[src[i]]。状態(乱数の状態を含む)を丸ごと複製する。全 sim は同じ手番 cur で揃って進む
    void copy_sims(std::vector<int> src) {
        std::vector<Sim> nw; nw.reserve(N);
        for (int i = 0; i < N; ++i) nw.push_back(sims[src[i]]);
        sims.swap(nw);
    }

    // observation: [N, F]
    py::array_t<float> obs() {
        const int F = 9 /*market inv*/ + 9 /*prices*/ + 9 /*own shed*/ + 9 /*opp shed*/
                    + 6 /*money own/opp, step frac, day frac, lead, n_units*/;
        py::array_t<float> a({(py::ssize_t)N, (py::ssize_t)F});
        auto o = a.mutable_unchecked<2>();
        for (int i = 0; i < N; ++i) {
            const State& s = sims[i].st;
            int me = learner_seat, op = 1 - learner_seat;
            int k = 0;
            for (int j = 0; j < 9; ++j) o(i,k++) = (float)s.market.inventory[j] / 10000.0f;
            for (int j = 0; j < 9; ++j) o(i,k++) = (float)s.market.prices[j] / 250.0f;
            for (int j = 0; j < 9; ++j) o(i,k++) = (float)s.farms[me].shed[j] / 100.0f;
            for (int j = 0; j < 9; ++j) o(i,k++) = (float)s.farms[op].shed[j] / 100.0f;
            o(i,k++) = (float)(s.farms[me].money / 50000.0);
            o(i,k++) = (float)(s.farms[op].money / 50000.0);
            o(i,k++) = (float)((s.farms[me].money - s.farms[op].money) / 20000.0);
            o(i,k++) = (float)s.step / (float)steps;
            o(i,k++) = (float)s.hour / 24.0f;
            o(i,k++) = (float)s.farms[me].n_units / 13.0f;
        }
        return a;
    }
    // orders: int32 [N, 10, 3] (op,item,n); op==0 = empty slot.
    // opp_orders may be an empty array, in which case the opponent plays its tape.
    void step2(py::array_t<int32_t> orders, py::array_t<int32_t> opp_orders) {
        auto b = orders.unchecked<3>();
        bool has_opp = opp_orders.ndim() == 3 && opp_orders.shape(0) == N;
        int nA = (int)A.steps.size(), nB = (int)B.steps.size();
        if (has_opp) {
            auto c = opp_orders.unchecked<3>();
            py::gil_scoped_release rel;
            for (int i = 0; i < N; ++i) {
                if (sims[i].st.done) continue;
                Action a0 = A.steps[std::min(cur, nA-1)];
                Action a1 = B.steps[std::min(cur, nB-1)];
                apply_orders(learner_seat == 0 ? a0 : a1, b, i);
                apply_orders(learner_seat == 0 ? a1 : a0, c, i);
                sims[i].step(a0, a1);
            }
        } else {
            py::gil_scoped_release rel;
            for (int i = 0; i < N; ++i) {
                if (sims[i].st.done) continue;
                Action a0 = A.steps[std::min(cur, nA-1)];
                Action a1 = B.steps[std::min(cur, nB-1)];
                apply_orders(learner_seat == 0 ? a0 : a1, b, i);
                sims[i].step(a0, a1);
            }
        }
        cur++;
    }
    void step(py::array_t<int32_t> orders) {
        py::array_t<int32_t> empty(std::vector<py::ssize_t>{0});
        step2(orders, empty);
    }
    // units: int32 [N, MAX_U, 3] (op,arg,n); op 255 = "keep the tape's action".
    // Overriding unit actions is what lets a policy be adaptive at all: the shop
    // draw explains a third of the bank variance and an open-loop tape cannot
    // respond to it.
    // opp_orders must be supplied too, or the opponent silently plays the raw
    // tape. Leaving it out made a carrot overlay look like a 0.990 win rate that
    // was really the terminal liquidation we had already banked in v17.
    void step_units(py::array_t<int32_t> units, py::array_t<int32_t> orders,
                    py::array_t<int32_t> opp_orders) {
        auto u = units.unchecked<3>();
        auto b = orders.unchecked<3>();
        bool has_opp = opp_orders.ndim() == 3 && opp_orders.shape(0) == N;
        int nA = (int)A.steps.size(), nB = (int)B.steps.size();
        py::gil_scoped_release rel;
        for (int i = 0; i < N; ++i) {
            if (sims[i].st.done) continue;
            Action a0 = A.steps[std::min(cur, nA-1)];
            Action a1 = B.steps[std::min(cur, nB-1)];
            Action& mine = (learner_seat == 0) ? a0 : a1;
            int nu = (int)u.shape(1);
            if (nu > mine.n_units) nu = mine.n_units;
            for (int k = 0; k < nu; ++k) {
                int op = u(i,k,0);
                if (op == 255) continue;                 // keep tape action
                mine.units[k].op  = (uint8_t)op;
                mine.units[k].arg = (uint8_t)u(i,k,1);
                mine.units[k].n   = (int16_t)u(i,k,2);
            }
            apply_orders(mine, b, i);
            if (has_opp) {
                auto c = opp_orders.unchecked<3>();
                apply_orders((learner_seat == 0) ? a1 : a0, c, i);
            }
            sims[i].step(a0, a1);
        }
        cur++;
    }
    // The OPPONENT's tape orders this step. Needed to build enhanced opponents:
    // the other teams' routes lose to us largely because they lack the generic
    // improvements we already ship (terminal liquidation is worth ~0.98 on its
    // own), so a pool of raw tapes collapses to the one family that still
    // contests us -- 14 of 14 "hard" opponents turned out to be one team.
    py::array_t<int32_t> opp_tape_orders() {
        const Tape& T = (learner_seat == 0) ? B : A;
        const Action& a = T.steps[std::min(cur, (int)T.steps.size()-1)];
        py::array_t<int32_t> out({(py::ssize_t)10, (py::ssize_t)3});
        auto o = out.mutable_unchecked<2>();
        for (int j = 0; j < 10; ++j) { o(j,0)=0; o(j,1)=0; o(j,2)=0; }
        for (int j = 0; j < a.n_orders && j < 10; ++j) {
            o(j,0)=a.orders[j].op; o(j,1)=a.orders[j].item; o(j,2)=a.orders[j].n;
        }
        return out;
    }
    // how many units the learner's tape drives this step
    int n_units() {
        const Tape& T = (learner_seat == 0) ? A : B;
        return T.steps[std::min(cur, (int)T.steps.size()-1)].n_units;
    }
    template <class B> void apply_orders(Action& act, const B& b, int i) {
        Order kept[16]; int nk = 0;
        if (keep_nonsell)
            for (int j = 0; j < act.n_orders; ++j)
                if (act.orders[j].op != M_SELL) kept[nk++] = act.orders[j];
        act.n_orders = 0;
        for (int j = 0; j < (int)b.shape(1) && act.n_orders < 10; ++j) {
            int op = b(i,j,0);
            if (op == 255) break;          // これ以降は注文なし(終端)
            if (op == 0) continue;         // 従来通り: 枠を消費しない
            Order& o = act.orders[act.n_orders++];
            if (op == 254) { o.op = M_NONE; o.item = 0; o.n = 0; continue; }
            // 254 = 「空の注文だが枠は占める」。テープの market にある [] がこれ。
            // 市場は両者から1単位ずつ交互に処理されるので、枠の位置がずれると
            // 相手との対応が変わり価格が動く(実測で 5〜268 コインの差になった)。
            o.op = (uint8_t)op; o.item = (uint8_t)b(i,j,1); o.n = b(i,j,2);
        }
        for (int j = 0; j < nk && act.n_orders < 10; ++j) act.orders[act.n_orders++] = kept[j];
    }
    // Compact board view for adaptive policies: tile kind, what occupies it, and
    // each unit's position. kind: 0 EMPTY 1 LOCKED 2 WEED 3 COOP 4 PASTURE 5 PLANT
    // Tile は13フィールドあるが unit_obs はうち6つしか出していない。
    // 欠けていたのは全部**時間発展の状態**(いつ植えたか/何日水を切らしたか/
    // 施肥はいつまで効くか/家畜はいつ去るか)で、収穫・灌水・世話・施肥の
    // タイミングを決める情報がそっくり入力から抜けていた(2026-09-10)。
    py::array_t<float> tiles_dyn(int seat) {
        const int C = 7;
        py::array_t<float> a({(py::ssize_t)N, (py::ssize_t)10, (py::ssize_t)10, (py::ssize_t)C});
        auto o = a.mutable_unchecked<4>();
        for (int i = 0; i < N; ++i) {
            const State& s = sims[i].st;
            const Farm& f = s.farms[seat ? (1-learner_seat) : learner_seat];
            int day = s.step / 24;
            for (int y = 0; y < 10; ++y)
                for (int x = 0; x < 10; ++x) {
                    const Tile& t = f.tiles[y][x];
                    o(i,y,x,0) = t.has_animal ? 1.0f : 0.0f;
                    o(i,y,x,1) = t.cared_today ? 1.0f : 0.0f;
                    o(i,y,x,2) = (float)t.consecutive_dry / 4.0f;
                    o(i,y,x,3) = (float)t.pending_care_bonus / 4.0f;
                    // 経過日数。未作付は -1。
                    o(i,y,x,4) = (t.kind == T_PLANT || t.has_animal)
                                 ? (float)(day - t.planted_day) / 30.0f : -1.0f;
                    o(i,y,x,5) = (t.max_lifespan_step >= 0)
                                 ? (float)(t.max_lifespan_step - s.step) / 720.0f : -1.0f;
                    o(i,y,x,6) = (t.fertilized_until_day >= 0)
                                 ? (float)(t.fertilized_until_day - day) / 30.0f : -1.0f;
                }
        }
        return a;
    }
    // 公式 obs に合わせた観測(2026-09-10、ルール文書を読んで整合させた)。
    //   ・private.inventories = Farm.inv。**エンジンにあったが観測に出ていなかった**。
    //     小麦を携行しているかで FEED の可否が決まるので、無いと軌道外で必ず壊れる。
    //   ・相手の納屋は**公式では見えない**(ルール「Players are unable to see the
    //     state of the other's shed」)。obs() の 9次元はリーク。obs_legal() では外す。
    //   ・相手の farm(タイル/位置/hires_today/quadrants)は**公式では見える**(public)。
    // 相手の farm(タイル/位置)は**ルール上 public**。自分の盤面しか見ていなかったので足す。
    py::array_t<float> tiles_base(int seat) {
        const int C = 6;
        py::array_t<float> a({(py::ssize_t)N, (py::ssize_t)10, (py::ssize_t)10, (py::ssize_t)C});
        auto o = a.mutable_unchecked<4>();
        for (int i = 0; i < N; ++i) {
            const Farm& f = sims[i].st.farms[seat ? (1-learner_seat) : learner_seat];
            for (int y = 0; y < 10; ++y)
                for (int x = 0; x < 10; ++x) {
                    const Tile& t = f.tiles[y][x];
                    o(i,y,x,0) = (float)t.kind / 5.0f;
                    o(i,y,x,1) = (float)t.what / 12.0f;
                    o(i,y,x,2) = t.watered_today ? 1.0f : 0.0f;
                    o(i,y,x,3) = t.fed_today ? 1.0f : 0.0f;
                    o(i,y,x,4) = (float)t.yield_units / 6.0f;
                    o(i,y,x,5) = t.fertilizer_available ? 1.0f : 0.0f;
                }
        }
        return a;
    }
    py::array_t<int32_t> unit_pos2(int seat) {
        py::array_t<int32_t> a({(py::ssize_t)N, (py::ssize_t)MAX_UNITS, (py::ssize_t)2});
        auto o = a.mutable_unchecked<3>();
        for (int i = 0; i < N; ++i) {
            const Farm& f = sims[i].st.farms[seat ? (1-learner_seat) : learner_seat];
            for (int u = 0; u < MAX_UNITS; ++u) {
                o(i,u,0) = u < f.n_units ? f.pos_x[u] : -1;
                o(i,u,1) = u < f.n_units ? f.pos_y[u] : -1;
            }
        }
        return a;
    }
    // 納屋の全12品目(家畜3種を含む)。obs_legal は9品目しか出しておらず、
    // 公式の private.shed にある家畜が抜けていた(PICKUP COW の可否が判断できない。2026-09-10)。
    // 合法手の判定を**エンジン自身の apply_unit で**行う(2026-09-10)。
    // 公式エンジンに合法手一覧の API は無く、不正な手は黙って無視される。条件を書き写すと
    // 取りこぼすので、農場を複製して候補を実際に適用し、状態が変われば合法とする。
    // 候補 44: 0待機 1北 2南 3東 4西 5水やり 6収穫 7施肥 8掘る 9小屋 10牧草地 11給餌 12肥料採取
    //          13世話 14全部置く 15-19植える(作物5) 20-31取り出す(品目12) 32-43置く(品目12)
    static void cand_of(int c, UnitAction& u) {
        static const uint8_t base[15] = {OP_PASS, OP_NORTH, OP_SOUTH, OP_EAST, OP_WEST, OP_WATER, OP_HARVEST,
            OP_FERTILIZE, OP_DIG, OP_BUILD_COOP, OP_BUILD_PASTURE, OP_FEED, OP_COLLECT_FERTILIZER, OP_CARE, OP_DROP};
        u.n = 1; u.arg = 0;
        if (c < 15) { u.op = base[c]; return; }
        if (c < 20) { u.op = OP_PLANT; u.arg = (uint8_t)(c - 15); return; }
        if (c < 32) { u.op = OP_PICKUP; u.arg = (uint8_t)(c - 20); return; }
        u.op = OP_PLACE; u.arg = (uint8_t)(c - 32);
    }
    static bool farm_changed(const Farm& a, const Farm& b, int u) {
        if (std::memcmp(a.tiles, b.tiles, sizeof(a.tiles)) != 0) return true;
        if (a.pos_x[u] != b.pos_x[u] || a.pos_y[u] != b.pos_y[u]) return true;
        if (std::memcmp(a.inv[u], b.inv[u], sizeof(a.inv[u])) != 0) return true;
        if (std::memcmp(a.shed, b.shed, sizeof(a.shed)) != 0) return true;
        if (std::memcmp(a.seeds, b.seeds, sizeof(a.seeds)) != 0) return true;
        return false;
    }
    // 09-21: 手番開始のスナップショット1つで全ユニットを判定すると、同じ手番で前のユニットが
    //   作った状況に依存する手(受け渡し)を「非合法」と誤判定する。公式エンジンはユニットを
    //   順に適用し、しかも条件を満たさない手は no-op になるだけで非合法という概念が無い。
    //   そこで「他のユニットが先に動いたらこうなりうる」楽観的な盤面でも試し、どちらかで
    //   状態が変われば合法とする。公式は余計な手を no-op にするだけなので、緩めても壊れない。
    //   LEGAL_OPT=0 で従来どおり。
    Farm optimistic(const Farm& f) const {
        Farm o = f;
        // 他のユニットが DROP/PLACE すれば納屋に入りうる → PICKUP が可能になる
        for (int u = 0; u < f.n_units && u < MAX_UNITS; ++u)
            for (int j = 0; j < N_ITEMS; ++j) {
                int add = (int)f.inv[u][j];
                o.shed[j] = (int16_t)std::min(30000, (int)o.shed[j] + add); o.shed_total += add;   // 09-21: 容量判定に使うので合計も更新
            }
        // 他のユニットが WATER すれば実りが増えうる → HARVEST が可能になる
        for (int y = 0; y < BOARD; ++y)
            for (int x = 0; x < BOARD; ++x) {
                Tile& t = o.tiles[y][x];
                if (t.kind == T_EMPTY) continue;
                if (t.yield_units < 1) t.yield_units = 1;
            }
        return o;
    }
    // 09-22: opts(局ごとの版、負なら環境変数 LEGAL_OPT)で局ごとに版を変えられる。旧マスクで学習した相手を
    //   新マスクの学習の中で本来の旧マスクのまま打たせるため(新マスクでは旧方策は貪欲でも所持金ほぼ 0)。opts を渡さなければ従来と同一。
    py::array_t<uint8_t> legal_mask(int max_units, py::object opts_obj) {
        const int NC = 44;
        static const int LEGAL_ENV = (getenv("LEGAL_OPT") == nullptr) ? 0 : atoi(getenv("LEGAL_OPT"));   // 0=従来(既定) 1=楽観盤面 2=手番内に変わりうる種を常に許す 3=全部許す
        std::vector<int> opts((size_t)N, LEGAL_ENV);
        if (!opts_obj.is_none()) {
            auto oa = opts_obj.cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
            if (oa.ndim() != 1 || oa.shape(0) != N) throw std::runtime_error("legal_mask: opts は長さ N の1次元配列");
            auto r = oa.unchecked<1>();
            for (int i = 0; i < N; ++i) if (r(i) >= 0) opts[i] = r(i);
        }
        py::array_t<uint8_t> a({(py::ssize_t)N, (py::ssize_t)max_units, (py::ssize_t)NC});
        auto o = a.mutable_unchecked<3>();
        for (int i = 0; i < N; ++i) {
            const int LEGAL_OPT = opts[i];
            const Farm& f = sims[i].st.farms[learner_seat];
            int day = sims[i].st.day;
            Farm fo = LEGAL_OPT ? optimistic(f) : f;
            for (int u = 0; u < max_units; ++u)
                for (int c = 0; c < NC; ++c) {
                    o(i,u,c) = 0;
                    if (u >= f.n_units) continue;
                    if (c == 0) { o(i,u,c) = 1; continue; }          // 待機は常に可能
                    Farm g = f; UnitAction ua; cand_of(c, ua);
                    sims[i].apply_unit(g, u, ua, day);
                    if (farm_changed(f, g, u)) { o(i,u,c) = 1; continue; }
                    if (!LEGAL_OPT) continue;
                    // LEGAL_OPT=2: 手番内に前のユニットが変えうる状態に依存する種類は常に許す。
                    //   公式は条件を満たさない手を no-op にするだけなので、緩めても壊れない。
                    //   (LEGAL_OPT=1 は「楽観的な盤面でも試す」だけで、CARE/WATER/COLLECT を取りこぼした)
                    if (LEGAL_OPT >= 3) { o(i,u,c) = 1; continue; }   // 全部許す(公式に非合法は存在しない)。切り分け用
                    if (LEGAL_OPT >= 2) {
                        uint8_t op = ua.op;
                        // 手番内に前のユニットがマスの種別や在庫を変えうる種は全部入れる。
                        // PLANT/DIG/BUILD 系は tile.kind に依存し、前のユニットの HARVEST/DIG/PLANT/BUILD で変わる。
                        // さらに PLANT は「待機に置換」が no-op 等価ですらない(提出された PLANT の数で種の超過を判定するので、
                        // 1つ消すと他のユニットの PLANT が通ってしまう)。
                        if (op == OP_WATER || op == OP_HARVEST || op == OP_FERTILIZE || op == OP_FEED
                            || op == OP_COLLECT_FERTILIZER || op == OP_CARE || op == OP_PICKUP || op == OP_PLACE
                            || op == OP_DIG || op == OP_BUILD_COOP || op == OP_BUILD_PASTURE || op == OP_DROP
                            || (op == OP_PLANT && f.seeds[ua.arg] > 0)) {
                            o(i,u,c) = 1; continue;
                        }
                    }
                    Farm go = fo; sims[i].apply_unit(go, u, ua, day);
                    o(i,u,c) = farm_changed(fo, go, u) ? 1 : 0;
                }
        }
        return a;
    }
    py::array_t<float> shed_all() {
        py::array_t<float> a({(py::ssize_t)N, (py::ssize_t)N_ITEMS});
        auto o = a.mutable_unchecked<2>();
        for (int i = 0; i < N; ++i) {
            const Farm& f = sims[i].st.farms[learner_seat];
            for (int j = 0; j < N_ITEMS; ++j) o(i,j) = (float)f.shed[j];
        }
        return a;
    }
    py::array_t<float> unit_inv(int max_units) {
        py::array_t<float> a({(py::ssize_t)N, (py::ssize_t)max_units, (py::ssize_t)N_ITEMS});
        auto o = a.mutable_unchecked<3>();
        for (int i = 0; i < N; ++i) {
            const Farm& f = sims[i].st.farms[learner_seat];
            for (int u = 0; u < max_units; ++u)
                for (int j = 0; j < N_ITEMS; ++j)
                    o(i,u,j) = (u < f.n_units) ? (float)f.inv[u][j] : 0.0f;
        }
        return a;
    }
    // exp274: 市場16特徴用。seat は learner から見た 0=自分 1=相手。(N, 9, 4):
    //   [0] 市場在庫に与えた正味の効果の累計(inv_eff)、[1] その作物の植え付け区画の数、[2] 収穫できる区画の数(yield_units>0)、
    //   [3] 生産設備の数(鶏小屋 → 卵、牧草地 → 牛乳・羊毛)。lk2_feat.board と同じ数え方。どれも公開情報か、自分の約定から分かる値。
    py::array_t<float> mkt_aux(int seat) {
        py::array_t<float> a({(py::ssize_t)N, (py::ssize_t)N_PRODUCTS, (py::ssize_t)4});
        auto o = a.mutable_unchecked<3>();
        for (int i = 0; i < N; ++i) {
            const Farm& f = sims[i].st.farms[seat ? (1-learner_seat) : learner_seat];
            float crop[N_PRODUCTS] = {0}, ready[N_PRODUCTS] = {0}, anim[N_PRODUCTS] = {0};
            for (int y = 0; y < 10; ++y)
                for (int x = 0; x < 10; ++x) {
                    const Tile& t = f.tiles[y][x];
                    if (t.kind == T_PLANT && t.what < N_CROPS) {
                        crop[t.what] += 1.0f;
                        if (t.yield_units > 0) ready[t.what] += 1.0f;
                    } else if (t.kind == T_COOP) {
                        anim[EGG] += 1.0f;
                    } else if (t.kind == T_PASTURE) {
                        anim[MILK] += 1.0f; anim[WOOL] += 1.0f;
                    }
                }
            for (int j = 0; j < N_PRODUCTS; ++j) {
                o(i,j,0) = (float)f.inv_eff[j]; o(i,j,1) = crop[j]; o(i,j,2) = ready[j]; o(i,j,3) = anim[j];
            }
        }
        return a;
    }
    // seat: 0=自分 1=相手。どちらも public。
    py::array_t<float> farm_pub(int seat) {
        const int F = 5;
        py::array_t<float> a({(py::ssize_t)N, (py::ssize_t)F});
        auto o = a.mutable_unchecked<2>();
        for (int i = 0; i < N; ++i) {
            const Farm& f = sims[i].st.farms[seat ? (1-learner_seat) : learner_seat];
            int k = 0;
            o(i,k++) = (float)(f.money / 50000.0);
            o(i,k++) = (float)f.n_units / 13.0f;
            o(i,k++) = (float)f.n_quadrants / 4.0f;
            o(i,k++) = (float)f.hires_today / 13.0f;
            // 納屋は private。相手側では出さない(自分で作りかけたリーク、2026-09-10)
            o(i,k++) = seat ? -1.0f : (float)f.shed_total / 100.0f;
        }
        return a;
    }
    // 自分の納屋 + 種。相手の納屋は入れない(リーク)。
    py::array_t<float> obs_legal() {
        const int F = 9 + 9 + 9 + 5 + 5;   // 市場在庫/価格/自分の納屋/自分の種/時刻など
        py::array_t<float> a({(py::ssize_t)N, (py::ssize_t)F});
        auto o = a.mutable_unchecked<2>();
        for (int i = 0; i < N; ++i) {
            const State& s = sims[i].st;
            const Farm& f = s.farms[learner_seat];
            const Farm& g = s.farms[1-learner_seat];
            int k = 0;
            for (int j = 0; j < 9; ++j) o(i,k++) = (float)s.market.inventory[j] / 10000.0f;
            for (int j = 0; j < 9; ++j) o(i,k++) = (float)s.market.prices[j] / 250.0f;
            for (int j = 0; j < 9; ++j) o(i,k++) = (float)f.shed[j] / 100.0f;
            for (int j = 0; j < 5; ++j) o(i,k++) = (float)f.seeds[j] / 20.0f;
            o(i,k++) = (float)(f.money / 50000.0);
            o(i,k++) = (float)(g.money / 50000.0);     // 相手の所持金は public
            o(i,k++) = (float)s.step / (float)steps;
            o(i,k++) = (float)s.hour / 24.0f;
            o(i,k++) = (float)s.day / 30.0f;
        }
        return a;
    }
    py::array_t<int32_t> tiles() {
        py::array_t<int32_t> a({(py::ssize_t)N, (py::ssize_t)10, (py::ssize_t)10});
        auto o = a.mutable_unchecked<3>();
        for (int i = 0; i < N; ++i) {
            const Farm& f = sims[i].st.farms[learner_seat];
            for (int y = 0; y < 10; ++y)
                for (int x = 0; x < 10; ++x) o(i,y,x) = (int)f.tiles[y][x].kind;
        }
        return a;
    }
    // Which animal occupies each tile (0 = none, else GOOSE/COW/SHEEP id).
    // tiles() returns only the kind, so an empty pasture and a stocked one look
    // identical -- not enough to decide whether another animal can be placed.
    py::array_t<int32_t> animals() {
        py::array_t<int32_t> a({(py::ssize_t)N, (py::ssize_t)10, (py::ssize_t)10});
        auto o = a.mutable_unchecked<3>();
        for (int i = 0; i < N; ++i) {
            const Farm& f = sims[i].st.farms[learner_seat];
            for (int y = 0; y < 10; ++y)
                for (int x = 0; x < 10; ++x)
                    o(i,y,x) = f.tiles[y][x].has_animal ? (int)f.tiles[y][x].what : 0;
        }
        return a;
    }
    py::array_t<int32_t> unit_pos() {
        py::array_t<int32_t> a({(py::ssize_t)N, (py::ssize_t)MAX_UNITS, (py::ssize_t)2});
        auto o = a.mutable_unchecked<3>();
        for (int i = 0; i < N; ++i) {
            const Farm& f = sims[i].st.farms[learner_seat];
            for (int u = 0; u < MAX_UNITS; ++u) {
                o(i,u,0) = u < f.n_units ? f.pos_x[u] : -1;
                o(i,u,1) = u < f.n_units ? f.pos_y[u] : -1;
            }
        }
        return a;
    }
    py::array_t<int32_t> seeds_held() {
        py::array_t<int32_t> a({(py::ssize_t)N, (py::ssize_t)N_CROPS});
        auto o = a.mutable_unchecked<2>();
        for (int i = 0; i < N; ++i)
            for (int c = 0; c < N_CROPS; ++c) o(i,c) = sims[i].st.farms[learner_seat].seeds[c];
        return a;
    }
    // which unit actions the tape drives this step, as [MAX_UNITS,3]
    py::array_t<int32_t> tape_units() {
        const Tape& T = (learner_seat == 0) ? A : B;
        const Action& a = T.steps[std::min(cur, (int)T.steps.size()-1)];
        py::array_t<int32_t> out({(py::ssize_t)MAX_UNITS, (py::ssize_t)3});
        auto o = out.mutable_unchecked<2>();
        for (int k = 0; k < MAX_UNITS; ++k) {
            bool live = k < a.n_units;
            o(k,0) = live ? a.units[k].op : 255;
            o(k,1) = live ? a.units[k].arg : 0;
            o(k,2) = live ? a.units[k].n : 1;
        }
        return out;
    }
    py::array_t<int32_t> shops() {
        py::array_t<int32_t> a({(py::ssize_t)N, (py::ssize_t)16});
        auto o = a.mutable_unchecked<2>();
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < 16; ++j) o(i,j) = -1;
            for (int j = 0; j < sims[i].st.n_shops && j < 16; ++j) o(i,j) = sims[i].st.shops[j];
        }
        return a;
    }
    // Per-unit observation for a policy that controls FIELD actions.
    // Layout per unit: 8 scalars + a 5x5 tile patch centred on the unit, each
    // tile as 6 numbers (kind, crop/animal id, watered, fed, yield, fertilizer).
    // The market layer is a 9-product decision; the route is where the games are
    // actually decided -- the two best routes differ by 4 sheep and 269 idle
    // unit-actions -- and that needs the board.
    static const int UOBS = 8 + 100*6;
    py::array_t<float> unit_obs(int max_units) {
        py::array_t<float> a({(py::ssize_t)N, (py::ssize_t)max_units, (py::ssize_t)UOBS});
        auto o = a.mutable_unchecked<3>();
        for (int i = 0; i < N; ++i) {
            const State& s = sims[i].st;
            const Farm& f = s.farms[learner_seat];
            for (int u = 0; u < max_units; ++u) {
                for (int k = 0; k < UOBS; ++k) o(i,u,k) = 0.0f;
                if (u >= f.n_units) continue;
                int ux = f.pos_x[u], uy = f.pos_y[u];
                int k = 0;
                o(i,u,k++) = (float)s.step / 720.0f;
                o(i,u,k++) = (float)s.hour / 24.0f;
                o(i,u,k++) = (float)u / 13.0f;
                o(i,u,k++) = (float)ux / 9.0f;
                o(i,u,k++) = (float)uy / 9.0f;
                o(i,u,k++) = (float)(f.money / 50000.0);
                o(i,u,k++) = (float)f.n_units / 13.0f;
                o(i,u,k++) = (float)f.shed_total / 100.0f;
                // The whole board, not a 5x5 patch. A clone with only local view
                // hit 0.9991 on the expert's own states and then diverged to an
                // 80% mismatch rate within three days of deployment: the route
                // walks units across the full 10x10 and a local patch cannot see
                // where a unit is supposed to be going.
                for (int y = 0; y < 10; ++y)
                    for (int x = 0; x < 10; ++x) {
                        if (x >= sims[i].cfg.board_size || y >= sims[i].cfg.board_size) {
                            o(i,u,k++) = -1.0f; k += 5; continue;
                        }
                        const Tile& t = f.tiles[y][x];
                        o(i,u,k++) = (float)t.kind / 5.0f;
                        o(i,u,k++) = (float)t.what / 12.0f;
                        o(i,u,k++) = t.watered_today ? 1.0f : 0.0f;
                        o(i,u,k++) = t.fed_today ? 1.0f : 0.0f;
                        o(i,u,k++) = (float)t.yield_units / 6.0f;
                        o(i,u,k++) = (x == ux && y == uy) ? 1.0f
                                     : (t.fertilizer_available ? 0.5f : 0.0f);
                    }
            }
        }
        return a;
    }
    py::array_t<double> money() {
        py::array_t<double> a({(py::ssize_t)N, (py::ssize_t)2});
        auto o = a.mutable_unchecked<2>();
        for (int i = 0; i < N; ++i) {
            o(i,0) = sims[i].st.farms[learner_seat].money;
            o(i,1) = sims[i].st.farms[1-learner_seat].money;
        }
        return a;
    }
    // the market orders the tape would have played this step, for the learner seat
    py::array_t<int32_t> tape_orders() {
        const Tape& T = (learner_seat == 0) ? A : B;
        int n = (int)T.steps.size();
        const Action& a = T.steps[std::min(cur, n-1)];
        py::array_t<int32_t> out({(py::ssize_t)10, (py::ssize_t)3});
        auto o = out.mutable_unchecked<2>();
        for (int j = 0; j < 10; ++j) { o(j,0)=0; o(j,1)=0; o(j,2)=0; }
        for (int j = 0; j < a.n_orders && j < 10; ++j) {
            o(j,0)=a.orders[j].op; o(j,1)=a.orders[j].item; o(j,2)=a.orders[j].n;
        }
        return out;
    }
};

PYBIND11_MODULE(kagsim, m) {
    m.doc() = "bit-exact Kaggriculture engine (1.32.7)";
    m.def("run_tapes", &run_tapes,
          py::arg("tape_a"), py::arg("na"), py::arg("tape_b"), py::arg("nb"),
          py::arg("seeds"), py::arg("steps") = 720, py::arg("nthreads") = 0, py::arg("weed") = -1.0);
    py::class_<VecMarket>(m, "VecMarket")
        .def(py::init<py::array_t<int32_t>,int,py::array_t<int32_t>,int,std::vector<uint64_t>,int,int>(),
             py::arg("tape_a"), py::arg("na"), py::arg("tape_b"), py::arg("nb"),
             py::arg("seeds"), py::arg("steps")=720, py::arg("learner_seat")=0)
        .def_readwrite("keep_nonsell", &VecMarket::keep_nonsell)
        .def("reset", &VecMarket::reset)
        .def("obs", &VecMarket::obs)
        .def("step", &VecMarket::step)
        .def("step2", &VecMarket::step2)
        .def("money", &VecMarket::money)
        .def("shops", &VecMarket::shops)
        .def("tape_units", &VecMarket::tape_units)
        .def("seeds_held", &VecMarket::seeds_held)
        .def("unit_pos", &VecMarket::unit_pos)
        .def("tiles", &VecMarket::tiles)
        .def("tiles_dyn", &VecMarket::tiles_dyn, py::arg("seat") = 0)
        .def("tiles_base", &VecMarket::tiles_base, py::arg("seat") = 0)
        .def("unit_pos2", &VecMarket::unit_pos2, py::arg("seat") = 0)
        .def("unit_inv", &VecMarket::unit_inv, py::arg("max_units") = 16)
        .def("shed_all", &VecMarket::shed_all)
        .def("legal_mask", &VecMarket::legal_mask, py::arg("max_units") = 20, py::arg("opts") = py::none())
        .def("farm_pub", &VecMarket::farm_pub, py::arg("seat") = 0)
        .def("mkt_aux", &VecMarket::mkt_aux)
        .def("obs_legal", &VecMarket::obs_legal)
        .def("animals", &VecMarket::animals)
        .def("unit_obs", &VecMarket::unit_obs, py::arg("max_units") = 14)
        .def("step_units", &VecMarket::step_units, py::arg("units"), py::arg("orders"), py::arg("opp_orders") = py::array_t<int32_t>(std::vector<py::ssize_t>{0}))
        .def("n_units", &VecMarket::n_units)
        .def("tape_orders", &VecMarket::tape_orders)
        .def("opp_tape_orders", &VecMarket::opp_tape_orders)
        .def("done", &VecMarket::done)
        .def("step_index", &VecMarket::step_index)
        .def("copy_sims", &VecMarket::copy_sims)
        .def("step_both", &VecMarket::step_both)
        .def("set_view", &VecMarket::set_view)
        .def("view", &VecMarket::view);
}
