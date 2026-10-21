"""Kaggriculture agent package with lazy simulation imports."""

__all__ = [
    "CompactState",
    "encode_compact_state",
    "run_differential_episode",
    "run_fast_episode",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name in {"CompactState", "encode_compact_state"}:
        from kaggriculture_agent import compact
        return getattr(compact, name)
    if name in {"run_differential_episode", "run_fast_episode"}:
        from kaggriculture_agent import simulation
        return getattr(simulation, name)
    raise AttributeError(name)
