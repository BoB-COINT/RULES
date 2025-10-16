"""STE rule evaluation package."""

from .engine import evaluate, evaluate_slow
from .io import load_events, load_features, load_rule, load_slow_features
from .utils import WINDOW_SECONDS_DEFAULT

__all__ = [
    "evaluate",
    "load_events",
    "load_features",
    "load_rule",
    "load_slow_features",
    "WINDOW_SECONDS_DEFAULT",
    "evaluate_slow",
]
