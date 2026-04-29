"""
Plot registry: decorator-based registration of all plot functions.

Usage
-----
  # Register a new plot type:
  @register_plot("hovmoller_u")
  def plot_hovmoller_u(ds, ax, **kwargs):
      ...

  # Invoke by name:
  from mitgcm_diag.registry import get_plotter
  fn = get_plotter("hovmoller_u")
  fn(ds, ax, lon=140, depth_max=300)
"""

from __future__ import annotations
from typing import Callable, Iterator
import warnings

# ── internal store ─────────────────────────────────────────────────────────────
_REGISTRY: dict[str, Callable] = {}


# ── registration helpers ───────────────────────────────────────────────────────

def register_plot(name: str) -> Callable:
    """Decorator that registers a plot function under *name*.

    Parameters
    ----------
    name : str
        Unique key used to look up this plotter (e.g. ``"hovmoller_u"``).

    Raises
    ------
    ValueError
        If *name* is already registered (prevents silent overwrites).

    Examples
    --------
    >>> @register_plot("my_plot")
    ... def plot_something(ds, ax, **kwargs):
    ...     ax.plot(ds["U"].values)
    """
    def decorator(fn: Callable) -> Callable:
        if name in _REGISTRY:
            warnings.warn(
                f"Plotter name '{name}' is already registered, overwriting.",
                stacklevel=2,
            )
        _REGISTRY[name] = fn
        fn._plot_name = name          # attach metadata back to function
        return fn
    return decorator

# ── look-up helpers ────────────────────────────────────────────────────────────

def get_plotter(name: str) -> Callable:
    """Return the plotter registered under *name*.

    Raises
    ------
    KeyError
        With a helpful message listing available plotters.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(
            f"Unknown plot type '{name}'. "
            f"Available plotters: [{available}]"
        )
    return _REGISTRY[name]


def list_plotters() -> list[str]:
    """Return a sorted list of all registered plot names."""
    return sorted(_REGISTRY.keys())


def iter_plotters() -> Iterator[tuple[str, Callable]]:
    """Yield (name, fn) pairs for all registered plotters."""
    yield from sorted(_REGISTRY.items())


def is_registered(name: str) -> bool:
    return name in _REGISTRY
