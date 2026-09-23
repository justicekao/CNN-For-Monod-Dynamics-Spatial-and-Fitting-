"""
solve_spatial() supports first-order (Lie) and second-order (Strang)
operator splitting between the reaction and transport sub-steps (see
solve.py's docstring). This test proves Strang is actually more accurate
at a given timestep, not just a documented-but-unverified option: run
both at a coarse dt, compare each to a Lie run at a much finer dt (the
reference), and check Strang's error is smaller.
"""

import numpy as np

from shared.monod_core import MonodConfig
from spatial_pde.src.boundary import BoundaryConditions
from spatial_pde.src.mesh import generate_line
from spatial_pde.src.reaction import MeshState
from spatial_pde.src.solve import FieldSpec, SpatialConfig, solve_spatial


def _config():
    # a single logistic-like population with metabolite feedback, so both
    # reaction and transport are simultaneously active and nonlinear.
    return MonodConfig.build(
        S=1, M=1, T=0,
        r=2.0, k=1.0, c=0.3, delta=0.3,
        m_supply=0.4, D_dilution=0.2,
        share_metabolite_uptake=True,
    )


def _run(dt, splitting):
    config = _config()
    n_cells = 10
    mesh = generate_line(length=5.0, n_cells=n_cells)
    rng = np.random.default_rng(2)
    N0 = rng.uniform(0.5, 2.0, size=(n_cells, 1))
    x0 = rng.uniform(0.5, 2.0, size=(n_cells, 1))

    field_N = FieldSpec(diffusion=0.05, bc=BoundaryConditions())
    field_x = FieldSpec(diffusion=0.2, bc=BoundaryConditions())
    spatial = SpatialConfig(mesh=mesh, reaction=config, strain_fields=[field_N],
                             metabolite_fields=[field_x], toxin_fields=[])
    initial = MeshState(N=N0, x=x0, tox=np.zeros((n_cells, 0)), q=None)
    return solve_spatial(spatial, initial, t_span=(0.0, 4.0), dt=dt, splitting=splitting)


def test_strang_splitting_is_more_accurate_than_lie_at_matched_dt():
    reference = _run(dt=0.002, splitting="lie")  # fine-dt ground truth

    coarse_dt = 0.2
    lie_result = _run(dt=coarse_dt, splitting="lie")
    strang_result = _run(dt=coarse_dt, splitting="strang")

    lie_error = np.abs(lie_result.N[-1] - reference.N[-1]).max()
    strang_error = np.abs(strang_result.N[-1] - reference.N[-1]).max()

    assert strang_error < lie_error, (
        f"expected Strang splitting error ({strang_error}) < Lie splitting "
        f"error ({lie_error}) at the same coarse dt={coarse_dt}"
    )
