"""
mesh.py -- unstructured finite-volume mesh geometry, for 1D/2D/3D alike.

This is what makes the spatial solver geometry-general rather than
locked to a straight 1D gut-axis line: a Mesh is built from a simplicial
complex (1D line segments, 2D triangles, or 3D tetrahedra) and everything
downstream (transport.py, reaction.py, solve.py) is written purely in
terms of cells and faces, so the SAME code runs on:

  - a 1D line with a variable cross-sectional area profile (a cheap
    approximation of a real gut's changing diameter along its length),
  - a structured 2D/3D grid (e.g. an experimental chamber/well), or
  - an arbitrary unstructured mesh imported from Gmsh/CAD/etc. via
    Mesh.from_meshio(), which is how a real segmented-gut or bespoke
    experimental-vessel geometry gets in: build/scan the shape in
    whatever meshing tool your group already uses, export a .msh/.vtk/
    .xdmf file, and load it here.

Numerics note: face fluxes use a two-point flux approximation (TPFA),
which assumes the line between two cell centroids is (approximately)
perpendicular to the shared face. This holds for the generated
structured meshes and for Delaunay-quality unstructured meshes; a very
skewed/low-quality mesh will need a non-orthogonal correction this v1
does not implement -- flag that if it comes up.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

BoundaryTagFn = Callable[[np.ndarray], str]


def _default_boundary_tag_fn(centroid: np.ndarray) -> str:
    return "boundary"


def _simplex_measure(points: np.ndarray) -> float:
    """Length (1D), area (2D), or volume (3D) of a simplex given its
    (dim+1) vertex coordinates, shape (dim+1, dim)."""
    n = points.shape[0]
    if n == 2:
        return float(np.linalg.norm(points[1] - points[0]))
    if n == 3:
        e1, e2 = points[1] - points[0], points[2] - points[0]
        return 0.5 * abs(e1[0] * e2[1] - e1[1] * e2[0])
    if n == 4:
        m = points[1:] - points[0]
        return abs(np.linalg.det(m)) / 6.0
    raise ValueError(f"unsupported simplex with {n} vertices")


def _facet_measure(points: np.ndarray) -> float:
    """Measure of a facet (one dimension lower than its owning simplex):
    a point "length" of 1 (1D), an edge length (2D), or a triangle area (3D)."""
    n = points.shape[0]
    if n == 1:
        return 1.0
    if n == 2:
        return float(np.linalg.norm(points[1] - points[0]))
    if n == 3:
        e1, e2 = points[1] - points[0], points[2] - points[0]
        return 0.5 * float(np.linalg.norm(np.cross(e1, e2)))
    raise ValueError(f"unsupported facet with {n} vertices")


def _outward_normal(dim: int, facet_points: np.ndarray, cell_centroid: np.ndarray,
                     facet_centroid: np.ndarray) -> np.ndarray:
    """Unit normal to a facet, oriented away from the owning cell's centroid."""
    if dim == 1:
        d = facet_centroid - cell_centroid
        n = np.array([1.0]) if d[0] >= 0 else np.array([-1.0])
        return n
    if dim == 2:
        e = facet_points[1] - facet_points[0]
        n = np.array([-e[1], e[0]])
    elif dim == 3:
        e1, e2 = facet_points[1] - facet_points[0], facet_points[2] - facet_points[0]
        n = np.cross(e1, e2)
    else:
        raise ValueError(f"unsupported dim={dim}")
    norm = np.linalg.norm(n)
    if norm < 1e-300:
        raise ValueError("degenerate facet (zero-length/area)")
    n = n / norm
    if np.dot(facet_centroid - cell_centroid, n) < 0:
        n = -n
    return n


@dataclass
class Face:
    owner: int
    neighbor: int          # -1 for a boundary face
    area: float             # facet measure, already scaled by cross-section for 1D
    distance: float          # owner<->neighbor centroid distance (internal) or owner<->facet distance (boundary)
    direction: np.ndarray     # unit vector from owner toward neighbor / outward, shape (dim,)
    centroid: np.ndarray
    tag: str = ""            # boundary tag; "" for internal faces


@dataclass
class Mesh:
    dim: int
    points: np.ndarray                 # (n_points, dim)
    cells: List[Tuple[int, ...]]        # each a simplex of dim+1 vertex indices
    cross_section_area: Optional[np.ndarray] = None  # (n_cells,), 1D meshes only

    cell_volumes: np.ndarray = field(init=False)
    cell_centroids: np.ndarray = field(init=False)
    faces: List[Face] = field(init=False)
    boundary_faces: Dict[str, List[int]] = field(init=False)   # tag -> face indices into `faces`
    cell_faces: List[List[int]] = field(init=False)             # cell index -> face indices touching it

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != self.dim:
            raise ValueError(f"points must have shape (n_points, {self.dim})")
        n_cells = len(self.cells)
        expected_verts = self.dim + 1
        for c in self.cells:
            if len(c) != expected_verts:
                raise ValueError(f"dim={self.dim} needs simplices with {expected_verts} vertices, got {c}")
        if self.cross_section_area is None:
            self.cross_section_area = np.ones(n_cells)
        else:
            self.cross_section_area = np.asarray(self.cross_section_area, dtype=float)
            if self.cross_section_area.shape != (n_cells,):
                raise ValueError("cross_section_area must have shape (n_cells,)")
            if self.dim != 1 and not np.allclose(self.cross_section_area, 1.0):
                raise ValueError("cross_section_area is only meaningful for dim=1 meshes")

        self.cell_centroids = np.array([self.points[list(c)].mean(axis=0) for c in self.cells])
        raw_measure = np.array([_simplex_measure(self.points[list(c)]) for c in self.cells])
        # a 1D "cell volume" is length * local cross-sectional area (a thin
        # slab of gut/tube of that length and area); 2D/3D cells are
        # unaffected (cross_section_area is forced to 1.0 there).
        self.cell_volumes = raw_measure * self.cross_section_area

        self._build_faces()

    def _build_faces(self):
        facet_to_cells: Dict[frozenset, List[Tuple[int, Tuple[int, ...]]]] = {}
        for ci, cell in enumerate(self.cells):
            for facet in itertools.combinations(cell, len(cell) - 1):
                key = frozenset(facet)
                facet_to_cells.setdefault(key, []).append((ci, facet))

        faces: List[Face] = []
        cell_faces: List[List[int]] = [[] for _ in self.cells]
        boundary_faces: Dict[str, List[int]] = {}

        for key, entries in facet_to_cells.items():
            facet_points = self.points[list(key)]
            facet_centroid = facet_points.mean(axis=0)

            if len(entries) == 2:
                (ci, _), (cj, _) = entries
                area = _facet_measure(facet_points)
                if self.dim == 1:
                    area = 0.5 * (self.cross_section_area[ci] + self.cross_section_area[cj])
                d_vec = self.cell_centroids[cj] - self.cell_centroids[ci]
                distance = float(np.linalg.norm(d_vec))
                direction = d_vec / distance if distance > 0 else d_vec
                fidx = len(faces)
                faces.append(Face(owner=ci, neighbor=cj, area=area, distance=distance,
                                   direction=direction, centroid=facet_centroid, tag=""))
                cell_faces[ci].append(fidx)
                cell_faces[cj].append(fidx)
            elif len(entries) == 1:
                ci, _ = entries[0]
                area = _facet_measure(facet_points)
                if self.dim == 1:
                    area = self.cross_section_area[ci]
                normal = _outward_normal(self.dim, facet_points, self.cell_centroids[ci], facet_centroid)
                distance = float(np.linalg.norm(facet_centroid - self.cell_centroids[ci]))
                fidx = len(faces)
                faces.append(Face(owner=ci, neighbor=-1, area=area, distance=distance,
                                   direction=normal, centroid=facet_centroid, tag=""))
                cell_faces[ci].append(fidx)
                boundary_faces.setdefault("__untagged__", []).append(fidx)
            else:
                raise ValueError(f"facet shared by {len(entries)} cells (expected 1 or 2) -- non-manifold mesh?")

        self.faces = faces
        self.cell_faces = cell_faces
        self.boundary_faces = boundary_faces

    def tag_boundary(self, tag_fn: BoundaryTagFn) -> None:
        """(Re)classify every boundary face's tag using tag_fn(face_centroid).

        Built-in generators already call this with a sensible default
        (see generate_line/generate_rectangle/generate_box); call it again
        yourself to rename/regroup tags, or to tag a mesh loaded via
        from_meshio() (which otherwise leaves every boundary face tagged
        "__untagged__" -- there's no universal way to know which physical
        group is "inlet" vs "wall" without a classifier)."""
        new_tags: Dict[str, List[int]] = {}
        for fidx, f in enumerate(self.faces):
            if f.neighbor != -1:
                continue
            tag = tag_fn(f.centroid)
            f.tag = tag
            new_tags.setdefault(tag, []).append(fidx)
        self.boundary_faces = new_tags

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def total_volume(self) -> float:
        return float(self.cell_volumes.sum())

    @classmethod
    def from_meshio(cls, meshio_mesh, boundary_tag_fn: Optional[BoundaryTagFn] = None) -> "Mesh":
        """Build a Mesh from any meshio.Mesh (loaded via meshio.read(path)),
        auto-detecting dimensionality from the highest-dimensional simplex
        cell block present ("line" -> 1D, "triangle" -> 2D, "tetra" -> 3D).
        This is the entry point for real/arbitrary geometry: build the
        shape in Gmsh, Blender, or any CAD/meshing tool that can export a
        format meshio reads, then load it here.
        """
        cell_type_by_dim = {1: "line", 2: "triangle", 3: "tetra"}
        points = meshio_mesh.points
        dim = None
        cells = None
        for d, cell_type in sorted(cell_type_by_dim.items(), reverse=True):
            for block in meshio_mesh.cells:
                if block.type == cell_type and len(block.data) > 0:
                    dim, cells = d, block.data
                    break
            if cells is not None:
                break
        if cells is None:
            raise ValueError("no line/triangle/tetra cell block found in this meshio mesh")
        points = points[:, :dim]
        mesh = cls(dim=dim, points=points, cells=[tuple(c) for c in cells])
        mesh.tag_boundary(boundary_tag_fn or _default_boundary_tag_fn)
        return mesh


def generate_line(length: float, n_cells: int, area_profile: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                   left_tag: str = "left", right_tag: str = "right") -> Mesh:
    """A 1D mesh of n_cells equal segments from x=0 to x=length.

    area_profile(x) -> cross-sectional area at position x (array in,
    array out) lets this cheaply approximate a real gut/tube's varying
    diameter along its length; omit it for a constant unit cross-section.
    """
    x = np.linspace(0.0, length, n_cells + 1)
    points = x.reshape(-1, 1)
    cells = [(i, i + 1) for i in range(n_cells)]
    cell_centers = 0.5 * (x[:-1] + x[1:])
    area = np.ones(n_cells) if area_profile is None else np.asarray(area_profile(cell_centers), dtype=float)
    mesh = Mesh(dim=1, points=points, cells=cells, cross_section_area=area)

    def tag_fn(centroid, length=length, left_tag=left_tag, right_tag=right_tag):
        return left_tag if centroid[0] < length / 2 else right_tag

    mesh.tag_boundary(tag_fn)
    return mesh


def generate_rectangle(lx: float, ly: float, nx: int, ny: int) -> Mesh:
    """A structured 2D mesh (each grid cell split into 2 triangles) over
    [0, lx] x [0, ly] -- a simple stand-in for a flat experimental
    chamber/well. Boundary faces are tagged left/right/bottom/top."""
    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    points = np.array([(x, y) for y in ys for x in xs])

    def vid(i, j):
        return j * (nx + 1) + i

    cells = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            cells.append((a, b, c))
            cells.append((a, c, d))

    mesh = Mesh(dim=2, points=points, cells=cells)

    def tag_fn(centroid, lx=lx, ly=ly):
        x, y = centroid
        if x < 1e-9:
            return "left"
        if x > lx - 1e-9:
            return "right"
        if y < 1e-9:
            return "bottom"
        if y > ly - 1e-9:
            return "top"
        return "boundary"

    mesh.tag_boundary(tag_fn)
    return mesh


def generate_box(lx: float, ly: float, lz: float, nx: int, ny: int, nz: int) -> Mesh:
    """A structured 3D mesh (each grid cube split into 6 tetrahedra) over
    [0,lx] x [0,ly] x [0,lz]. Boundary faces are tagged x0/x1/y0/y1/z0/z1."""
    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    zs = np.linspace(0.0, lz, nz + 1)
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs])

    def vid(i, j, k):
        return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i

    # Canonical Kuhn (Freudenthal) triangulation of a cube into 6 tetrahedra,
    # all sharing the main diagonal (corner 0 <-> corner 6). This specific
    # decomposition is required (not an arbitrary choice of diagonals) so
    # that neighboring cubes triangulate every SHARED face with the same
    # diagonal -- otherwise adjacent cubes' tet facets don't match up and
    # leave spurious "boundary" faces in the interior of the mesh.
    tet_corner_indices = [
        (0, 1, 2, 6), (0, 1, 5, 6), (0, 3, 2, 6),
        (0, 3, 7, 6), (0, 4, 5, 6), (0, 4, 7, 6),
    ]

    cells = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                corners = [
                    vid(i, j, k), vid(i + 1, j, k), vid(i + 1, j + 1, k), vid(i, j + 1, k),
                    vid(i, j, k + 1), vid(i + 1, j, k + 1), vid(i + 1, j + 1, k + 1), vid(i, j + 1, k + 1),
                ]
                for tet in tet_corner_indices:
                    cells.append(tuple(corners[t] for t in tet))

    mesh = Mesh(dim=3, points=points, cells=cells)

    def tag_fn(centroid, lx=lx, ly=ly, lz=lz):
        x, y, z = centroid
        if x < 1e-9:
            return "x0"
        if x > lx - 1e-9:
            return "x1"
        if y < 1e-9:
            return "y0"
        if y > ly - 1e-9:
            return "y1"
        if z < 1e-9:
            return "z0"
        if z > lz - 1e-9:
            return "z1"
        return "boundary"

    mesh.tag_boundary(tag_fn)
    return mesh
