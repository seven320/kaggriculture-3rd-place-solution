"""Build a runnable tree of the kit under ROOT from training/kit/ (code) and DATA (weights, teacher data).

The kit code hard-codes Colab paths (/content/...) and cluster paths (written as /workspace/... in this repository).
This script writes a copy of kit/ with only those path prefixes rewritten:

    /content                                   -> ROOT/content
    /workspace/kaggriculture    -> ROOT/cluster
    /workspace/kagg_async      -> ROOT/content/async

and lays out the files as the kit expects (code under drive/MyDrive/kaggriculture,
weights and splits under local_logs, agents under lab/agents, teacher caches under cluster/ext4p).
Weights and data are symlinked from DATA, not copied. local_logs is a real directory: logs and new
weights written by BC / PPO land there.

The shimishige policies are added as local_logs/orbit_mmpqbcw_gen<N>.pt (+ .pt.cfg.json) and
lab/agents/mmpqbcw_gen<N>_paceland_liq, the market-search package as lab/agents/gen35_market_search.

Rerunning is safe: code is rewritten again, existing files in local_logs are kept.

Usage: materialize.py [--root ROOT] [--data DATA] [--no-build]
"""
import argparse, json, os, re, shutil, subprocess, sys, sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
KIT = HERE / "kit"
AGENT_KEYS = ("ptr", "nmb_pen", "nmb_mode")   # written by the kit's tools/mkagent_nmb.py into an agent's cfg.json
TEXT = {".py", ".sh", ".env", ".md", ".txt", ".sbatch", ".conf", ".json", ".cpp", ".hpp", ".tsv"}


def rewriter(root):
    table = [
        (re.compile(r"(?<![A-Za-z0-9_.-])/workspace/kaggriculture(?![A-Za-z0-9_])"), str(root / "cluster")),
        (re.compile(r"(?<![A-Za-z0-9_.-])/workspace/kagg_async(?![A-Za-z0-9_])"), str(root / "content/async")),
        (re.compile(r"(?<![A-Za-z0-9_.-])/content(?![A-Za-z0-9_.-])"), str(root / "content")),
    ]

    def rewrite(text):
        for pat, new in table:
            text = pat.sub(new, text)
        return text
    return rewrite


def copy_code(src, dst, rewrite):
    """Copy a code directory, rewriting path prefixes in text files. Files only in dst (built .so) are kept."""
    n = 0
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix in TEXT:
            old = p.read_text()
            new = rewrite(old)
            n += new != old
            out.write_text(new)
            shutil.copymode(p, out)
        else:
            shutil.copy2(p, out)
    return n


def link(target, name):
    """Symlink name -> target, replacing an old symlink but never a real file."""
    name.parent.mkdir(parents=True, exist_ok=True)
    if name.is_symlink():
        name.unlink()
    if not name.exists():
        name.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def link_children(src, dst):
    if not src.is_dir():
        return 0
    for p in sorted(src.iterdir()):
        link(p, dst / p.name)
    return len(list(src.iterdir()))


def build_native(fast):
    """Same compile line as code/fast/build.sh and build_var.sh, for the running interpreter."""
    import pybind11
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    for src, mod in (("kagsim.cpp", "kagsim"), ("kagsim_var.cpp", "kagsim_var")):
        if not (fast / src).exists():
            continue
        tmp = fast / (mod + "_build.so")
        subprocess.run(["g++", "-O3", "-shared", "-std=c++17", "-fPIC", "-march=native", "-I" + pybind11.get_include(),
                        "-I" + sysconfig.get_paths()["include"], src, "-o", tmp.name], cwd=fast, check=True)
        os.replace(tmp, fast / (mod + suffix))
        print("built", fast / (mod + suffix))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("TEAM_BC_PPO_ROOT", HERE / "work")))
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("TEAM_BC_PPO_DATA", HERE / "data")))
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()
    root = a.root.resolve()
    content, cluster = root / "content", root / "cluster"
    kagg = content / "drive/MyDrive/kaggriculture"
    rewrite = rewriter(root)
    code = KIT

    # code
    for sub in ("bc", "fast"):
        print("code/%s: %d files rewritten" % (sub, copy_code(code / sub, kagg / sub, rewrite)))
    print("code/fasteval: %d files rewritten" % copy_code(code / "fasteval", content / "lab/fasteval", rewrite))
    for p in sorted(code.iterdir()):
        if p.is_dir() and p.name not in ("bc", "fast"):
            copy_code(p, content / "repo/cluster" / p.name, rewrite)
    # league.py imports rust_eval2, which reads lab/rust_arena/ids.txt at import time.
    # The Rust arena (opponent agents ported to Rust) is not part of this repository; league.py needs it.
    if (KIT / "rust_arena").is_dir():
        copy_code(KIT / "rust_arena", content / "lab/rust_arena", rewrite)

    # weights, splits, agents, teacher data of the kit
    snap = a.data / "kit"
    logs = content / "local_logs"
    logs.mkdir(parents=True, exist_ok=True)
    (content / "async").mkdir(exist_ok=True)
    (content / "lab/agents").mkdir(parents=True, exist_ok=True)
    print("weights:", link_children(snap / "weights", logs), "splits:", link_children(snap / "splits", logs),
          "agents:", link_children(snap / "agents", content / "lab/agents"))
    for p in sorted((snap / "data").iterdir()) if (snap / "data").is_dir() else []:
        if p.name == "tape_ug":
            link(p, content / "repo/experiments/exp182_mix_tape_plus_repairs/code/tape_ug")
        elif (p / "stock_target.json").exists():
            link(p, content / "lab/agents" / p.name)
        else:
            link(p, cluster / "ext4p" / p.name)
    link(logs, cluster / "local_logs")
    link(content / "lab", cluster / "lab")

    # shimishige policies: raw weights for INIT / league, packages as agents
    pol = a.data / "shimishige_policies"
    for pkg in sorted(pol.glob("mmpqbcw_gen*_paceland_liq")):
        gen = pkg.name.split("_")[1]
        link(pkg, content / "lab/agents" / pkg.name)
        link(pkg / "policy.pt", logs / ("orbit_mmpqbcw_%s.pt" % gen))
        # a weight's .pt.cfg.json holds the architecture only; the package cfg.json adds the inference keys
        cfg = json.loads((pkg / "cfg.json").read_text())
        arch = {k: v for k, v in cfg.items() if k not in AGENT_KEYS}
        (logs / ("orbit_mmpqbcw_%s.pt.cfg.json" % gen)).write_text(json.dumps(arch))
    ms = a.data / "shimishige_market_search/gen35-market-search/package"
    if ms.is_dir():
        link(ms, content / "lab/agents/gen35_market_search")

    if not a.no_build:
        build_native(kagg / "fast")
    missing = [str(p) for p in (snap / "weights", snap / "data", pol) if not p.is_dir()]
    if missing:
        print("not found (download the weights and data first, see training/README.md):", *missing, sep="\n  ", file=sys.stderr)
    print("ROOT =", root)
