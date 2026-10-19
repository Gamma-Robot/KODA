"""Interaction-mesh Laplacian and neighborhood mesh pull-out helpers.

``create_interaction_mesh``, ``get_adjacency_list``, and
``calculate_laplacian_coordinates`` are ported from Holosoma
(amazon-far/holosoma ``holosoma_retargeting/src/utils.py``) without cvxpy or
the whole-body SQP.  The graph is built once on human landmarks plus object
points and reused for the robot; object vertices stay fixed.

Anti-penetration is not an SQP cost.  After retargeting, every stored
neighborhood object point is tested against the dense BrainCo visual mesh
and the wrist is translated toward the recorded human-hand landmarks so
interior points leave along the real-hand direction, not along palm geom z.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import Delaunay, cKDTree

try:
    from scipy.spatial import QhullError
except ImportError:  # pragma: no cover - older SciPy
    from scipy.spatial.qhull import QhullError


def create_interaction_mesh(vertices: np.ndarray):
    """Tetrahedralize ``vertices`` with Delaunay triangulation.

    Ported from Holosoma ``utils.create_interaction_mesh``.
    """
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tri = Delaunay(points)
    return points, np.asarray(tri.simplices, dtype=np.int64)


def get_adjacency_list(tetrahedra, num_vertices):
    """Undirected adjacency from tetrahedral edges.

    Ported from Holosoma ``utils.get_adjacency_list``.
    """
    adj = [set() for _ in range(int(num_vertices))]
    tets = np.asarray(tetrahedra, dtype=np.int64).reshape(-1, 4)
    for tet in tets:
        for i in range(4):
            for j in range(i + 1, 4):
                u = int(tet[i])
                v = int(tet[j])
                if u == v:
                    continue
                adj[u].add(v)
                adj[v].add(u)
    return [list(neighbours) for neighbours in adj]


def calculate_laplacian_coordinates(
    vertices, adj_list, epsilon=1e-6, uniform_weight=True
):
    """``Δ_i = v_i -`` weighted neighbour centroid.

    Ported from Holosoma ``utils.calculate_laplacian_coordinates``.
    """
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    laplacian = np.zeros_like(points)
    for i, neighbours in enumerate(adj_list):
        if not neighbours:
            continue
        vi = points[i]
        neighbor_positions = points[np.asarray(neighbours, dtype=np.int64)]
        distances = np.linalg.norm(vi - neighbor_positions, axis=1)
        if uniform_weight:
            weights = np.ones_like(distances)
        else:
            weights = 1.0 / (1.5 * distances + float(epsilon))
        sum_of_weights = float(np.sum(weights))
        if sum_of_weights <= 0.0:
            continue
        weighted_sum = np.sum(weights[:, None] * neighbor_positions, axis=0)
        laplacian[i] = vi - weighted_sum / sum_of_weights
    return laplacian


def calculate_laplacian_matrix(
    vertices, adj_list, epsilon=1e-6, uniform_weight=True
):
    """``(N, N)`` Laplacian so ``L @ vertices ≈ calculate_laplacian_coordinates``.

    Ported from Holosoma ``utils.calculate_laplacian_matrix``.  Uniform weights
    depend only on the (frozen) adjacency, so ``L`` is constant during an SQP
    inner loop.  Distance weights use the current vertex positions.
    """
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    n = len(points)
    laplacian_matrix = np.zeros((n, n), dtype=np.float64)
    for i, neighbours in enumerate(adj_list):
        if not neighbours:
            continue
        ids = np.asarray(neighbours, dtype=np.int64)
        if uniform_weight:
            weights = np.ones(len(ids), dtype=np.float64) / float(len(ids))
        else:
            distances = np.linalg.norm(points[i] - points[ids], axis=1)
            weights = 1.0 / (distances + float(epsilon))
            total = float(np.sum(weights))
            if total <= 0.0:
                continue
            weights = weights / total
        laplacian_matrix[i, i] = 1.0
        for weight, neighbor_idx in zip(weights, ids):
            laplacian_matrix[i, int(neighbor_idx)] = -float(weight)
    return laplacian_matrix


def _knn_adjacency(vertices: np.ndarray, k: int = 4) -> list[list[int]]:
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return [[]]
    kk = min(max(1, int(k)) + 1, n)
    tree = cKDTree(points)
    _, indices = tree.query(points, k=kk, workers=1)
    indices = np.atleast_2d(np.asarray(indices, dtype=np.int64))
    adj: list[list[int]] = []
    for i, row in enumerate(indices):
        neighbours = [int(j) for j in np.atleast_1d(row) if int(j) != i]
        adj.append(neighbours)
    return adj


def build_interaction_adjacency(vertices: np.ndarray, knn: int = 4) -> list[list[int]]:
    """Delaunay adjacency with a k-NN fallback when Qhull cannot tetrahedralize."""
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    n = len(points)
    if n < 4:
        return _knn_adjacency(points, k=min(knn, max(n - 1, 1)))
    try:
        _, tetrahedra = create_interaction_mesh(points)
        return get_adjacency_list(tetrahedra, n)
    except (QhullError, ValueError):
        return _knn_adjacency(points, k=knn)


def laplacian_residual(
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    adj_list: list[list[int]],
    *,
    uniform_weight: bool = True,
    weight: float = 1.0,
) -> np.ndarray:
    """Flattened ``Δ(target) - Δ(source)`` on a shared adjacency list."""
    source = np.asarray(source_vertices, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_vertices, dtype=np.float64).reshape(-1, 3)
    if source.shape != target.shape:
        raise ValueError(
            f"source/target Laplacian vertices must match, got {source.shape} vs {target.shape}"
        )
    delta = calculate_laplacian_coordinates(
        target, adj_list, uniform_weight=bool(uniform_weight)
    ) - calculate_laplacian_coordinates(
        source, adj_list, uniform_weight=bool(uniform_weight)
    )
    return float(weight) * delta.reshape(-1)


def _nearest_hand_signed_distance(
    object_points: np.ndarray,
    hand_samples: np.ndarray,
    hand_normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Signed distance of finite object points to the nearest finite hand sample.

    Returns ``(query, dist, signed, outward_normals)`` or ``None`` if empty.
    ``s = (p_obj - q_hand) · n_hand``; ``s < 0`` means the object point is on
    the interior side of the hand surface.
    """
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    hand = np.asarray(hand_samples, dtype=np.float64).reshape(-1, 3)
    normals = np.asarray(hand_normals, dtype=np.float64).reshape(-1, 3)
    if len(obj) == 0 or len(hand) == 0 or len(normals) == 0:
        return None
    if len(normals) != len(hand):
        raise ValueError("hand samples and normals must have the same length")
    finite_obj = np.isfinite(obj).all(axis=1)
    finite_hand = np.isfinite(hand).all(axis=1) & np.isfinite(normals).all(axis=1)
    if not np.any(finite_obj) or not np.any(finite_hand):
        return None
    query = obj[finite_obj]
    cloud = hand[finite_hand]
    cloud_n = normals[finite_hand]
    dist, idx = cKDTree(cloud).query(query, k=1, workers=1)
    dist = np.atleast_1d(np.asarray(dist, dtype=np.float64))
    idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
    q = cloud[idx]
    nrm = cloud_n[idx]
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-8)
    signed = np.sum((query - q) * nrm, axis=1)
    return query, dist, signed, nrm


def mesh_penetration_hinge(
    object_points: np.ndarray,
    hand_samples: np.ndarray,
    hand_normals: np.ndarray,
    *,
    thickness_m: float = 0.018,
    eps_m: float = 0.001,
    weight: float = 30.0,
) -> np.ndarray:
    """One-sided hinge residual for tests and diagnostics; not used in SQP.

    ``s = (p_obj - q_hand) · n_hand``.  A penalty is applied only when
    ``s < 0`` and the nearest hand sample is within ``thickness_m``.
    """
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    residual = np.zeros(len(obj), dtype=np.float64)
    packed = _nearest_hand_signed_distance(object_points, hand_samples, hand_normals)
    if packed is None:
        return residual
    _query, dist, signed, _nrm = packed
    finite_obj = np.isfinite(obj).all(axis=1)
    inside = (signed < 0.0) & (dist < float(thickness_m))
    scale = 1.0 / math.sqrt(float(len(obj)))
    values = np.zeros(int(np.count_nonzero(finite_obj)), dtype=np.float64)
    values[inside] = (
        float(weight) * scale * np.maximum(-signed[inside] - float(eps_m), 0.0)
    )
    residual[finite_obj] = values
    return residual


def retract_axis_toward_target(
    origin: np.ndarray,
    target: np.ndarray,
    *,
    min_norm_m: float = 1e-6,
) -> np.ndarray | None:
    """Unit vector from ``origin`` toward ``target``, or ``None`` if degenerate."""
    start = np.asarray(origin, dtype=np.float64).reshape(3)
    end = np.asarray(target, dtype=np.float64).reshape(3)
    if not np.isfinite(start).all() or not np.isfinite(end).all():
        return None
    delta = end - start
    norm = float(np.linalg.norm(delta))
    if norm < float(min_norm_m):
        return None
    return delta / norm


def retract_axis_away_from_object(
    axis: np.ndarray,
    origin: np.ndarray,
    object_points: np.ndarray,
) -> np.ndarray:
    """Unit wrist translation that backs the palm off the object cloud.

    Fallback when the human-hand target is missing.  ``axis`` is any direction
    candidate; the sign is chosen so the hand moves *away* from the object
    centroid.
    """
    direction = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        norm = 1.0
    direction = direction / norm
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(obj).all(axis=1)
    if not np.any(finite):
        return direction
    to_obj = obj[finite].mean(axis=0) - np.asarray(origin, dtype=np.float64).reshape(3)
    if float(np.dot(direction, to_obj)) > 0.0:
        direction = -direction
    return direction


def deepest_inside_world_push(
    object_points: np.ndarray,
    hand_samples: np.ndarray,
    hand_normals: np.ndarray,
    *,
    thickness_m: float = 0.018,
    clearance_m: float = 0.001,
    retract_axis: np.ndarray | None = None,
) -> tuple[np.ndarray, int, float]:
    """World translation of the hand that extracts the deepest interior object point.

    Uses every stored neighborhood object point against the dense hand samples,
    not the 16 retarget landmarks.  Depth still comes from nearest-sample
    signed distance.  The translation *direction* is ``retract_axis`` when
    given (already signed away from the object); otherwise
    ``t = -n (depth + clearance)`` for the deepest interior point.

    Returns ``(push_xyz, n_inside, max_depth_m)``.  ``push`` is zero when no
    neighborhood point is interior.
    """
    zero = np.zeros(3, dtype=np.float64)
    packed = _nearest_hand_signed_distance(object_points, hand_samples, hand_normals)
    if packed is None:
        return zero, 0, 0.0
    _query, dist, signed, nrm = packed
    inside = (signed < 0.0) & (dist < float(thickness_m))
    n_inside = int(np.count_nonzero(inside))
    if n_inside == 0:
        return zero, 0, 0.0
    depths = -signed[inside]
    deepest = int(np.argmax(depths))
    depth = float(depths[deepest])
    travel = depth + float(clearance_m)
    if retract_axis is not None:
        axis = np.asarray(retract_axis, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            return zero, n_inside, depth
        push = (axis / norm) * travel
    else:
        normal = nrm[inside][deepest]
        push = -normal * travel
    return np.asarray(push, dtype=np.float64), n_inside, depth
