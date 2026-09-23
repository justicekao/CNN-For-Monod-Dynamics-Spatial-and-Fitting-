"""
Validation plan item 2 (docs/spatial_pde_spec.md): "with no supply/
dilution/consumption (i.e. only diffusion/advection), total mass
integrated across the domain should be constant (up to boundary flux)."

Parametrized over a 1D line, a 2D triangulated rectangle, and a 3D
tetrahedral box, so this also doubles as the "the mesh framework
actually runs in more than one dimension" smoke test -- the same
solve_spatial()/transport code path is exercised on all three, which is
the point of building the mesh/transport layer generically rather than
as a 1D-only special case.
"""

import numpy as np
import pytest

from shared.monod_core import MonodConfig
from spatial_pde.src.boundary import BoundaryConditions
from spatial_pde.src.mesh import generate_box, generate_line, generate_rectangle
from spatial_pde.src.reaction import MeshState
from spatial_pde.src.solve import FieldSpec, SpatialConfig, solve_spatial


def _no_reaction_config():
    """S=0 strains, M=1 metabolite, no supply/dilution/consumption -- pure
    diffusion, nothing to create or destroy mass."""
    return MonodConfig.build(S=0, M=1, T=0, m_supply=0.0, D_dilution=0.0)


@pytest.mark.parametrize("mesh_factory", [
    lambda: generate_line(10.0, 20),
    lambda: generate_rectangle(4.0, 3.0, 6, 5),
    lambda: generate_box(2.0, 2.0, 2.0, 3, 3, 3),
])
def test_pure_diffusion_conserves_mass_with_no_flux_boundaries(mesh_factory):
    mesh = mesh_factory()
    config = _no_reaction_config()

    rng = np.random.default_rng(1)
    x0 = rng.uniform(0.5, 5.0, size=(mesh.n_cells, 1))

    diffusing_field = FieldSpec(diffusion=0.3, velocity=None, bc=BoundaryConditions())  # default BC = no-flux
    spatial = SpatialConfig(
        mesh=mesh, reaction=config,
        strain_fields=[], metabolite_fields=[diffusing_field], toxin_fields=[],
    )
    initial = MeshState(N=np.zeros((mesh.n_cells, 0)), x=x0, tox=np.zeros((mesh.n_cells, 0)), q=None)

    result = solve_spatial(spatial, initial, t_span=(0.0, 20.0), dt=0.5)

    mass = result.x[:, :, 0] @ mesh.cell_volumes  # (n_steps+1,) total mass at each saved time
    np.testing.assert_allclose(mass, mass[0], rtol=1e-6)

    # and it should actually have spread out, not just sat still
    assert not np.allclose(result.x[-1], result.x[0], rtol=1e-3)


def test_pure_diffusion_relaxes_toward_uniform_concentration():
    mesh = generate_rectangle(4.0, 3.0, 6, 5)
    config = _no_reaction_config()

    x0 = np.where(mesh.cell_centroids[:, 0:1] < 2.0, 4.0, 0.0)
    diffusing_field = FieldSpec(diffusion=1.0)
    spatial = SpatialConfig(mesh=mesh, reaction=config, strain_fields=[],
                             metabolite_fields=[diffusing_field], toxin_fields=[])
    initial = MeshState(N=np.zeros((mesh.n_cells, 0)), x=x0, tox=np.zeros((mesh.n_cells, 0)), q=None)

    result = solve_spatial(spatial, initial, t_span=(0.0, 200.0), dt=1.0)
    uniform_value = (x0[:, 0] @ mesh.cell_volumes) / mesh.total_volume()
    np.testing.assert_allclose(result.x[-1, :, 0], uniform_value, rtol=0.05)
