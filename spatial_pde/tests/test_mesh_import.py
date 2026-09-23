"""
Mesh.from_meshio() is the mechanism for using a real/arbitrary geometry
(a segmented gut scan, a bespoke experimental vessel -- anything built or
scanned in Gmsh, Blender, or other CAD/meshing tools and exported to a
format meshio reads) instead of one of the built-in structured
generators. This test stands in for that with a small hand-built
triangular mesh (a unit square split into 4 triangles around a center
point) so it runs with no external mesh file or dependency beyond
meshio itself.
"""

import meshio
import numpy as np

from spatial_pde.src.mesh import Mesh


def test_from_meshio_computes_correct_geometry():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    cells = [("triangle", np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]))]
    imported = meshio.Mesh(points=points, cells=cells)

    mesh = Mesh.from_meshio(imported)

    assert mesh.dim == 2
    assert mesh.n_cells == 4
    np.testing.assert_allclose(mesh.total_volume(), 1.0)
    # every boundary edge should be tagged (default: one catch-all tag,
    # since there's no universal way to infer "inlet" vs "wall" without a
    # classifier -- see Mesh.tag_boundary's docstring) and every internal
    # edge (the 4 spokes from the center point) should NOT be a boundary face
    n_boundary = sum(1 for f in mesh.faces if f.neighbor == -1)
    n_internal = sum(1 for f in mesh.faces if f.neighbor != -1)
    assert n_boundary == 4  # the square's 4 outer edges
    assert n_internal == 4  # the 4 spokes from the center point


def test_from_meshio_custom_boundary_tag_fn():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    cells = [("triangle", np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]))]
    imported = meshio.Mesh(points=points, cells=cells)

    def tag_fn(centroid):
        return "left" if centroid[0] < 0.5 else "right"

    mesh = Mesh.from_meshio(imported, boundary_tag_fn=tag_fn)
    assert set(mesh.boundary_faces.keys()) <= {"left", "right"}
    assert sum(len(v) for v in mesh.boundary_faces.values()) == 4
