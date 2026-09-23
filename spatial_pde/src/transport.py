"""
transport.py -- diffusion (TPFA) + advection (first-order upwind) operator
assembly, built purely from a Mesh's cell/face graph.

Because everything here is expressed in terms of mesh.faces (owner,
neighbor, area, distance, direction) rather than array indices along a
1D axis, the exact same assembly code works for a 1D line, a 2D
triangulated chamber, or a 3D tetrahedral mesh imported from real
geometry -- there is no dimension-specific branch anywhere in this file.

build_transport_operator() returns a sparse matrix M and vector b such
that, for state u (one field, one value per cell):

    du/dt|_transport = (M @ u + b) / mesh.cell_volumes

M encodes cell-to-cell diffusive+advective exchange (and is exactly
conservative for the internal-face terms: each internal face contributes
equal and opposite entries to its two cells); b carries boundary
Dirichlet/Neumann/inflow contributions.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import scipy.sparse as sparse

from .boundary import BoundaryConditions, Dirichlet, Neumann, Outflow
from .mesh import Mesh

VelocityField = Union[None, np.ndarray, Callable[[np.ndarray], np.ndarray]]


def build_transport_operator(mesh: Mesh, D, velocity: VelocityField = None,
                              bcs: Optional[BoundaryConditions] = None):
    n = mesh.n_cells
    bcs = bcs or BoundaryConditions()
    D_cell = np.full(n, float(D)) if np.isscalar(D) else np.asarray(D, dtype=float)

    def d_face(i, j=None):
        return D_cell[i] if j is None else 0.5 * (D_cell[i] + D_cell[j])

    def vel_at(centroid):
        if velocity is None:
            return np.zeros(mesh.dim)
        if callable(velocity):
            return np.asarray(velocity(centroid), dtype=float)
        return np.asarray(velocity, dtype=float)

    rows, cols, vals = [], [], []
    b = np.zeros(n)

    def add(i, j, v):
        rows.append(i)
        cols.append(j)
        vals.append(v)

    for f in mesh.faces:
        i, j = f.owner, f.neighbor
        v = vel_at(f.centroid)
        s = float(np.dot(v, f.direction))  # velocity component along owner->neighbor / outward normal
        s_plus, s_minus = max(s, 0.0), min(s, 0.0)

        if j != -1:
            k = d_face(i, j) * f.area / f.distance if f.distance > 0 else 0.0
            if k:
                add(i, i, -k); add(i, j, k)
                add(j, j, -k); add(j, i, k)
            if s != 0.0:
                add(i, i, -f.area * s_plus)
                add(i, j, -f.area * s_minus)
                add(j, i, f.area * s_plus)
                add(j, j, f.area * s_minus)
            continue

        bc = bcs.get(f.tag)
        if isinstance(bc, Dirichlet):
            if f.distance > 0:
                k = d_face(i) * f.area / f.distance
                add(i, i, -k)
                b[i] += k * bc.value
            if s < 0:  # inflow: material enters carrying the specified concentration
                b[i] -= f.area * s * bc.value
            else:
                add(i, i, -f.area * s_plus)
        elif isinstance(bc, Neumann):
            b[i] -= bc.flux * f.area
            add(i, i, -f.area * s_plus)
        elif isinstance(bc, Outflow):
            add(i, i, -f.area * s_plus)
        else:
            raise TypeError(f"unknown boundary condition type: {type(bc)!r}")

    M = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return M, b
