"""
Validation plan item 1 (docs/spatial_pde_spec.md): "with D=0 and v=0 at
every grid point, the spatial model at any single grid point should
reduce exactly to the well-mixed ODE result from shared/monod_core."

This directly verifies the reaction term was ported into the spatial
solver correctly: every mesh cell, with transport fully switched off,
must reproduce shared/monod_core.simulate_well_mixed() -- both paths
ultimately call the exact same kinetics.reaction_rhs().
"""

import numpy as np

from shared.monod_core import MonodConfig, simulate_well_mixed
from spatial_pde.src.boundary import BoundaryConditions
from spatial_pde.src.mesh import generate_line
from spatial_pde.src.reaction import MeshState
from spatial_pde.src.solve import FieldSpec, SpatialConfig, solve_spatial


def _config():
    return MonodConfig.build(
        S=2, M=3, T=0,
        r=[[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
        k=0.5, c=[[0.2, 0.2, 0.0], [0.0, 0.2, 0.2]], delta=[0.1, 0.1],
        m_supply=0.5, D_dilution=0.2,
        share_metabolite_uptake=True,
    )


def test_zero_diffusion_zero_advection_matches_well_mixed_ode():
    config = _config()
    n_cells = 4
    mesh = generate_line(length=10.0, n_cells=n_cells)  # D=0, velocity=None (defaults) below -> no transport at all

    N0 = np.tile([1.0, 0.2], (n_cells, 1))
    x0 = np.tile([2.0, 2.0, 2.0], (n_cells, 1))

    zero_field = FieldSpec(diffusion=0.0, velocity=None, bc=BoundaryConditions())
    spatial = SpatialConfig(
        mesh=mesh, reaction=config,
        strain_fields=[zero_field, zero_field],
        metabolite_fields=[zero_field, zero_field, zero_field],
        toxin_fields=[],
    )
    initial = MeshState(N=N0, x=x0, tox=np.zeros((n_cells, 0)), q=None)

    t_span = (0.0, 50.0)
    result = solve_spatial(spatial, initial, t_span, dt=1.0)

    reference = simulate_well_mixed(config, N0=N0[0], x0=x0[0], t_span=t_span, n_points=2)

    # every cell started identical and there's no transport, so every cell
    # must still be identical to each other AND match the well-mixed result
    for cell in range(n_cells):
        np.testing.assert_allclose(result.N[-1, cell], reference.N[-1], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(result.x[-1, cell], reference.x[-1], rtol=1e-4, atol=1e-6)


def test_zero_diffusion_cells_evolve_independently_with_different_initial_conditions():
    """A stronger version of the same check: give each cell a DIFFERENT
    initial condition. With no transport, each cell's trajectory must
    match its OWN independent well-mixed simulation -- proving cells
    aren't accidentally coupled when diffusion/advection are zero."""
    config = _config()
    n_cells = 3
    mesh = generate_line(length=6.0, n_cells=n_cells)

    N0 = np.array([[1.0, 0.2], [0.5, 0.5], [0.1, 1.0]])
    x0 = np.array([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [3.0, 0.5, 1.5]])

    zero_field = FieldSpec()
    spatial = SpatialConfig(
        mesh=mesh, reaction=config,
        strain_fields=[zero_field, zero_field],
        metabolite_fields=[zero_field, zero_field, zero_field],
        toxin_fields=[],
    )
    initial = MeshState(N=N0, x=x0, tox=np.zeros((n_cells, 0)), q=None)

    t_span = (0.0, 30.0)
    result = solve_spatial(spatial, initial, t_span, dt=1.0)

    for cell in range(n_cells):
        reference = simulate_well_mixed(config, N0=N0[cell], x0=x0[cell], t_span=t_span, n_points=2)
        np.testing.assert_allclose(result.N[-1, cell], reference.N[-1], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(result.x[-1, cell], reference.x[-1], rtol=1e-4, atol=1e-6)
