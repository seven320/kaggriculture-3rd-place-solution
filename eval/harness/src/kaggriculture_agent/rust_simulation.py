"""Python policy runner whose game transitions are performed by Rust."""

from __future__ import annotations

import inspect
import json
import math
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGURATION = {
    "episodeSteps": 720, "boardSize": 10, "startingMoney": 3000,
    "maxMarketOrdersPerTurn": 10, "turnsPerDay": 24, "shedCapacity": 100,
    "weedSpawnChance": 0.005, "townShopUnlockInterval": 3,
    "townShopSellInterval": 4, "townCenterSellInterval": 24,
    "farmHandCostMult": 1, "decoupledRng": False,
}
SUPPORTED_CONFIGURATION = frozenset(DEFAULT_CONFIGURATION) | {"seed"}


def _validate_configuration(config: Mapping[str, Any]) -> None:
    unsupported = sorted(set(config) - SUPPORTED_CONFIGURATION)
    if unsupported:
        raise ValueError(f"Rust runner does not support configuration keys: {unsupported}")
    for key, value in config.items():
        if key == "seed":
            valid = value is None or (type(value) is int and -(2**63) <= value < 2**63)
        elif key == "decoupledRng":
            valid = type(value) is bool
        elif key == "weedSpawnChance":
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        else:
            minimum = 0 if key in ("startingMoney", "farmHandCostMult") else 1
            if key == "boardSize":
                minimum = 4
            maximum = 2**16 - 1 if key == "boardSize" else 2**32 - 1
            valid = type(value) is int and minimum <= value <= maximum
        if not valid:
            raise ValueError(f"Invalid Rust runner configuration {key}={value!r}")


class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _attrs(value: Any) -> Any:
    if isinstance(value, dict):
        return AttrDict({key: _attrs(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_attrs(item) for item in value]
    return value


def _binary_path(binary: str | os.PathLike[str] | None) -> Path:
    candidates = ([Path(binary)] if binary else [
        ROOT.parents[1] / "engine/rust/target/release/simulation-server",
        ROOT / "target/release/simulation-server",
    ])
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError(
        "Rust simulation server not found; build it with "
        "`cargo build --release -p kaggriculture-python-bridge --bin simulation-server` "
        "or pass binary=. Python-engine fallback is intentionally disabled."
    )


class RustEnvironment:
    """Small env.run/state/steps-compatible facade over the Rust engine.

    This is a deterministic research runner. It does not reproduce judge action
    validation, timeout accounting, log capture, or replay metadata.
    An omitted or None seed uses 0; the resolved seed is recorded in env.info
    and is hidden from policies, as in the official runner.
    """

    def __init__(self, configuration: Mapping[str, Any] | None = None, *, binary=None):
        config = dict(configuration or {})
        _validate_configuration(config)
        # The official resolve_episode_seed stores the seed in env.info and
        # clears configuration.seed before calling policies. Exposing it here
        # changes seed-sensitive policies (including in-match search).
        self.configuration = _attrs({**DEFAULT_CONFIGURATION, **config, "seed": None})
        seed = config.get("seed")
        self.info = {"seed": 0 if seed is None else int(seed), "engine": "rust"}
        self._process = subprocess.Popen(
            [str(_binary_path(binary))], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self.steps: list[list[AttrDict]] = []
        try:
            self._exchange({"command": "start", "seed": self.info["seed"], "configuration": config})
        except BaseException:
            self.close()
            raise

    def _exchange(self, request: Mapping[str, Any]) -> None:
        assert self._process.stdin is not None and self._process.stdout is not None
        self._process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError(f"Rust simulation server exited with {self._process.poll()}")
        payload = json.loads(line)
        actions = request.get("actions", [None, None])
        self.state = [AttrDict(observation=_attrs(obs), reward=reward, status=status, action=action)
                      for obs, reward, status, action in zip(payload["observations"], payload["rewards"], payload["statuses"], actions, strict=True)]
        self.done = bool(payload["done"])
        self.steps.append(self.state)
        if self.done and self._process.poll() is None:
            self._process.terminate()
            self._process.wait()

    def step(self, actions: Sequence[Mapping[str, Any]]) -> None:
        self._exchange({"command": "step", "actions": list(actions)})

    def run(self, policies: Sequence[Any]) -> None:
        try:
            resolved = [_resolve_policy(policy) for policy in policies]
            if len(resolved) != 2:
                raise ValueError("Kaggriculture requires two policies")
            while not self.done:
                actions = [_call_policy(policy, player.observation, self.configuration)
                           for policy, player in zip(resolved, self.state, strict=True)]
                self.step(actions)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
        self._process.wait()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def _resolve_policy(policy: Any):
    if not isinstance(policy, str):
        if not callable(policy):
            raise TypeError(f"policy must be callable or built-in name, got {type(policy)!r}")
        return policy
    from kaggle_environments.envs.kaggriculture.kaggriculture import agents
    try:
        return agents[policy]
    except KeyError as exc:
        raise ValueError(f"unknown built-in policy: {policy}") from exc


def _call_policy(policy, observation, configuration):
    try:
        count = len(inspect.signature(policy).parameters)
    except (TypeError, ValueError):
        count = 1
    action = policy(observation) if count < 2 else policy(observation, configuration)
    if not isinstance(action, Mapping):
        raise TypeError(f"policies must return an action mapping, got {type(action)!r}")
    return action


def run_rust_episode(policies: Sequence[Any], *, configuration=None, binary=None) -> RustEnvironment:
    env = RustEnvironment(configuration, binary=binary)
    try:
        env.run(policies)
        return env
    except BaseException:
        env.close()
        raise


def make(
    name: str = "kaggriculture",
    configuration: Mapping[str, Any] | None = None,
    debug: bool = False,
    *,
    binary=None,
) -> RustEnvironment:
    """Drop-in factory for scripts that only need ``env.run/state/steps``."""
    if name != "kaggriculture":
        raise ValueError(f"Rust runner only supports kaggriculture, got {name!r}")
    if debug not in (False, True):
        raise TypeError("debug must be bool")
    return RustEnvironment(configuration, binary=binary)

