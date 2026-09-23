"""
boundary.py -- boundary condition types for the spatial transport solver.

Per docs/spatial_pde_spec.md: "no-flux (zero-gradient) is the standard
default for a state with no known input/output at a boundary" -- that is
Neumann(0.0) here, and is what every boundary tag gets unless you say
otherwise. Dirichlet is for a fixed concentration (e.g. continuous
metabolite intake at a gut's proximal end); Outflow is for an
advection-dominated exit (e.g. excretion at the distal end) where
diffusive flux is assumed zero (zero-gradient) and material simply
leaves at the local advective velocity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class Dirichlet:
    """Fixed concentration/population at the boundary."""
    value: float


@dataclass(frozen=True)
class Neumann:
    """Fixed outward DIFFUSIVE flux at the boundary. flux=0.0 (the
    default) is the standard no-flux / zero-gradient condition. Advective
    transport at a Neumann boundary still proceeds via upwinding on the
    velocity field (set velocity=0 there too if you want a truly closed
    boundary for advection as well)."""
    flux: float = 0.0


@dataclass(frozen=True)
class Outflow:
    """Zero-gradient (no diffusive flux) boundary that lets material leave
    at the local advective velocity -- the standard "excretion" condition
    at a distal/downstream end. If the velocity field points INTO the
    domain here (unexpected inflow), no concentration was specified for
    what's flowing in, so that inflow is treated as carrying zero
    concentration; use Dirichlet instead if you need a real inflow value."""


BoundaryCondition = "Dirichlet | Neumann | Outflow"


@dataclass
class BoundaryConditions:
    """Per-boundary-tag conditions for ONE field (species). Any tag not
    listed falls back to `default` (no-flux)."""

    by_tag: Dict[str, object] = field(default_factory=dict)
    default: object = field(default_factory=lambda: Neumann(0.0))

    def get(self, tag: str):
        return self.by_tag.get(tag, self.default)
