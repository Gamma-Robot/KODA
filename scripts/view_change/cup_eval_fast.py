"""Fast CupModel metrics used by the contact-aware wrist/finger Powell loop.

The kernels reproduce ``point_group_metrics`` / ``evaluate_configuration``
numerically (linear percentile, first-argmin on ties) so Powell / DE search
paths stay the same.  The previous Python+NumPy path spent most of its time
in per-group ``np.percentile`` and dict assembly, at ~3 ms per evaluation;
BrainCo contact windows call this hundreds of thousands of times.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - CONTACT_PY ships numba
    njit = None  # type: ignore[assignment]


STAT_SURFACE_MIN = 0
STAT_CONTACT_POINTS = 1
STAT_CONTACT = 2
STAT_PENETRATING_POINTS = 3
STAT_PENETRATION_RATIO = 4
STAT_PENETRATION_MEAN = 5
STAT_PENETRATION_P95 = 6
STAT_PENETRATION_MAX = 7
STAT_CLOSEST_X = 8
STAT_CLOSEST_Y = 9
STAT_CLOSEST_Z = 10
STAT_NORMAL_X = 11
STAT_NORMAL_Y = 12
STAT_DIM = 13


def _linear_percentile_np(values: np.ndarray, q: float) -> float:
    n = int(values.shape[0])
    if n == 0:
        return 0.0
    ordered = np.sort(values)
    if n == 1:
        return float(ordered[0])
    pos = (n - 1) * (q / 100.0)
    lo = int(np.floor(pos))
    if lo >= n - 1:
        return float(ordered[n - 1])
    weight = pos - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[lo + 1] * weight)


def _world_tf_np(
    packed_local: np.ndarray, rotation: np.ndarray, translation: np.ndarray, world: np.ndarray
) -> None:
    np.dot(packed_local, rotation.T, out=world)
    world += translation


def _cup_stats_np(
    xyz: np.ndarray,
    offsets: np.ndarray,
    cx: float,
    cy: float,
    z_ref: float,
    r_ref: float,
    slope: float,
    z_min: float,
    z_max: float,
    thr: float,
    stats: np.ndarray,
    scratch: np.ndarray,
) -> tuple[float, float, float, float]:
    n_points = int(xyz.shape[0])
    n_groups = int(offsets.shape[0] - 1)
    positive_count = 0
    penetration_max = 0.0
    penetration_sum = 0.0
    for group in range(n_groups):
        start = int(offsets[group])
        end = int(offsets[group + 1])
        count = end - start
        if count <= 0:
            stats[group, :] = 0.0
            continue
        sl = xyz[start:end]
        dx = sl[:, 0] - cx
        dy = sl[:, 1] - cy
        z = sl[:, 2]
        rho = np.sqrt(dx * dx + dy * dy)
        lateral = rho - (r_ref + slope * (z - z_ref))
        within = (z >= z_min) & (z <= z_max)
        axial = np.maximum(np.maximum(z_min - z, z - z_max), 0.0)
        surface = np.sqrt(lateral * lateral + axial * axial)
        penetration = np.where(within, np.maximum(-lateral, 0.0), 0.0)
        scratch[start:end] = penetration
        eligible = np.flatnonzero(within)
        if eligible.size:
            best = int(eligible[np.argmin(np.abs(lateral[eligible]))])
            minimum = float(np.abs(lateral[best]))
            contact_count = int(np.count_nonzero(within & (np.abs(lateral) <= thr)))
        else:
            best = int(np.argmin(surface))
            minimum = float(surface[best])
            contact_count = 0
        radial_norm = float(np.hypot(dx[best], dy[best]))
        nx = float(dx[best] / radial_norm) if radial_norm > 1e-10 else 0.0
        ny = float(dy[best] / radial_norm) if radial_norm > 1e-10 else 0.0
        positive = penetration[penetration > 0.0]
        pos_n = int(positive.size)
        stats[group, STAT_SURFACE_MIN] = minimum
        stats[group, STAT_CONTACT_POINTS] = float(contact_count)
        stats[group, STAT_CONTACT] = 1.0 if contact_count >= 3 else 0.0
        stats[group, STAT_PENETRATING_POINTS] = float(pos_n)
        stats[group, STAT_PENETRATION_RATIO] = float(pos_n) / float(count)
        stats[group, STAT_PENETRATION_MEAN] = (
            float(np.mean(positive)) if pos_n else 0.0
        )
        stats[group, STAT_PENETRATION_P95] = (
            float(np.percentile(positive, 95)) if pos_n else 0.0
        )
        stats[group, STAT_PENETRATION_MAX] = float(np.max(penetration))
        stats[group, STAT_CLOSEST_X] = float(sl[best, 0])
        stats[group, STAT_CLOSEST_Y] = float(sl[best, 1])
        stats[group, STAT_CLOSEST_Z] = float(sl[best, 2])
        stats[group, STAT_NORMAL_X] = nx
        stats[group, STAT_NORMAL_Y] = ny
        if pos_n:
            positive_count += pos_n
            penetration_sum += float(np.sum(positive))
        group_max = float(np.max(penetration))
        if group_max > penetration_max:
            penetration_max = group_max
    if positive_count:
        positives = scratch[scratch > 0.0]
        p95 = float(np.percentile(positives, 95))
        mean_pos = penetration_sum / float(positive_count)
    else:
        p95 = 0.0
        mean_pos = 0.0
    ratio = float(positive_count) / float(n_points) if n_points else 0.0
    return p95, mean_pos, ratio, penetration_max


if njit is not None:

    @njit(cache=True)
    def _linear_percentile(values: np.ndarray, q: float) -> float:
        n = values.shape[0]
        if n == 0:
            return 0.0
        ordered = np.sort(values)
        if n == 1:
            return ordered[0]
        pos = (n - 1) * (q / 100.0)
        lo = int(np.floor(pos))
        if lo >= n - 1:
            return ordered[n - 1]
        weight = pos - lo
        return ordered[lo] * (1.0 - weight) + ordered[lo + 1] * weight

    @njit(cache=True)
    def _world_tf(
        packed_local: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
        world: np.ndarray,
    ) -> None:
        r00 = rotation[0, 0]
        r01 = rotation[0, 1]
        r02 = rotation[0, 2]
        r10 = rotation[1, 0]
        r11 = rotation[1, 1]
        r12 = rotation[1, 2]
        r20 = rotation[2, 0]
        r21 = rotation[2, 1]
        r22 = rotation[2, 2]
        tx = translation[0]
        ty = translation[1]
        tz = translation[2]
        for i in range(packed_local.shape[0]):
            x = packed_local[i, 0]
            y = packed_local[i, 1]
            z = packed_local[i, 2]
            world[i, 0] = x * r00 + y * r01 + z * r02 + tx
            world[i, 1] = x * r10 + y * r11 + z * r12 + ty
            world[i, 2] = x * r20 + y * r21 + z * r22 + tz

    @njit(cache=True)
    def _cup_stats(
        xyz: np.ndarray,
        offsets: np.ndarray,
        cx: float,
        cy: float,
        z_ref: float,
        r_ref: float,
        slope: float,
        z_min: float,
        z_max: float,
        thr: float,
        stats: np.ndarray,
        scratch: np.ndarray,
    ) -> tuple[float, float, float, float]:
        n_points = xyz.shape[0]
        n_groups = offsets.shape[0] - 1
        positive_count = 0
        penetration_max = 0.0
        penetration_sum = 0.0
        for group in range(n_groups):
            start = offsets[group]
            end = offsets[group + 1]
            count = end - start
            if count <= 0:
                for k in range(13):
                    stats[group, k] = 0.0
                continue
            has_eligible = False
            best = start
            best_abs = 1.0e300
            best_surface = 1.0e300
            best_surface_index = start
            contact_count = 0
            pos_n = 0
            pen_sum = 0.0
            pen_max = 0.0
            for i in range(start, end):
                dx = xyz[i, 0] - cx
                dy = xyz[i, 1] - cy
                z = xyz[i, 2]
                rho = np.sqrt(dx * dx + dy * dy)
                lateral = rho - (r_ref + slope * (z - z_ref))
                within = (z >= z_min) and (z <= z_max)
                axial = z_min - z
                above = z - z_max
                if above > axial:
                    axial = above
                if axial < 0.0:
                    axial = 0.0
                surface = np.sqrt(lateral * lateral + axial * axial)
                abs_lat = -lateral if lateral < 0.0 else lateral
                if within:
                    if (not has_eligible) or abs_lat < best_abs:
                        has_eligible = True
                        best_abs = abs_lat
                        best = i
                    if abs_lat <= thr:
                        contact_count += 1
                    pen = -lateral if lateral < 0.0 else 0.0
                else:
                    pen = 0.0
                if surface < best_surface:
                    best_surface = surface
                    best_surface_index = i
                scratch[i] = pen
                if pen > 0.0:
                    pos_n += 1
                    pen_sum += pen
                if pen > pen_max:
                    pen_max = pen
            if has_eligible:
                minimum = best_abs
            else:
                best = best_surface_index
                minimum = best_surface
                contact_count = 0
            dx = xyz[best, 0] - cx
            dy = xyz[best, 1] - cy
            radial_norm = np.sqrt(dx * dx + dy * dy)
            if radial_norm > 1e-10:
                nx = dx / radial_norm
                ny = dy / radial_norm
            else:
                nx = 0.0
                ny = 0.0
            if pos_n:
                compacted = np.empty(pos_n)
                write = 0
                for i in range(start, end):
                    if scratch[i] > 0.0:
                        compacted[write] = scratch[i]
                        write += 1
                p95 = _linear_percentile(compacted, 95.0)
                mean_pos = pen_sum / pos_n
            else:
                p95 = 0.0
                mean_pos = 0.0
            stats[group, 0] = minimum
            stats[group, 1] = float(contact_count)
            stats[group, 2] = 1.0 if contact_count >= 3 else 0.0
            stats[group, 3] = float(pos_n)
            stats[group, 4] = float(pos_n) / float(count)
            stats[group, 5] = mean_pos
            stats[group, 6] = p95
            stats[group, 7] = pen_max
            stats[group, 8] = xyz[best, 0]
            stats[group, 9] = xyz[best, 1]
            stats[group, 10] = xyz[best, 2]
            stats[group, 11] = nx
            stats[group, 12] = ny
            positive_count += pos_n
            penetration_sum += pen_sum
            if pen_max > penetration_max:
                penetration_max = pen_max
        if positive_count:
            compacted = np.empty(positive_count)
            write = 0
            for i in range(n_points):
                if scratch[i] > 0.0:
                    compacted[write] = scratch[i]
                    write += 1
            p95 = _linear_percentile(compacted, 95.0)
            mean_pos = penetration_sum / positive_count
        else:
            p95 = 0.0
            mean_pos = 0.0
        ratio = float(positive_count) / float(n_points) if n_points else 0.0
        return p95, mean_pos, ratio, penetration_max

    @njit(cache=True)
    def geoms_to_wrist_local(
        verts: np.ndarray,
        geom_ids: np.ndarray,
        geom_xpos: np.ndarray,
        geom_xmat: np.ndarray,
        wrist_pos: np.ndarray,
        wrist_mat: np.ndarray,
        out: np.ndarray,
    ) -> None:
        wr00 = wrist_mat[0, 0]
        wr01 = wrist_mat[0, 1]
        wr02 = wrist_mat[0, 2]
        wr10 = wrist_mat[1, 0]
        wr11 = wrist_mat[1, 1]
        wr12 = wrist_mat[1, 2]
        wr20 = wrist_mat[2, 0]
        wr21 = wrist_mat[2, 1]
        wr22 = wrist_mat[2, 2]
        wtx = wrist_pos[0]
        wty = wrist_pos[1]
        wtz = wrist_pos[2]
        for i in range(verts.shape[0]):
            gid = geom_ids[i]
            x = verts[i, 0]
            y = verts[i, 1]
            z = verts[i, 2]
            g00 = geom_xmat[gid, 0]
            g01 = geom_xmat[gid, 1]
            g02 = geom_xmat[gid, 2]
            g10 = geom_xmat[gid, 3]
            g11 = geom_xmat[gid, 4]
            g12 = geom_xmat[gid, 5]
            g20 = geom_xmat[gid, 6]
            g21 = geom_xmat[gid, 7]
            g22 = geom_xmat[gid, 8]
            wx = g00 * x + g01 * y + g02 * z + geom_xpos[gid, 0]
            wy = g10 * x + g11 * y + g12 * z + geom_xpos[gid, 1]
            wz = g20 * x + g21 * y + g22 * z + geom_xpos[gid, 2]
            dx = wx - wtx
            dy = wy - wty
            dz = wz - wtz
            out[i, 0] = dx * wr00 + dy * wr10 + dz * wr20
            out[i, 1] = dx * wr01 + dy * wr11 + dz * wr21
            out[i, 2] = dx * wr02 + dy * wr12 + dz * wr22

else:  # pragma: no cover

    def _world_tf(packed_local, rotation, translation, world):
        _world_tf_np(packed_local, rotation, translation, world)

    def _cup_stats(
        xyz, offsets, cx, cy, z_ref, r_ref, slope, z_min, z_max, thr, stats, scratch
    ):
        return _cup_stats_np(
            xyz, offsets, cx, cy, z_ref, r_ref, slope, z_min, z_max, thr, stats, scratch
        )

    def geoms_to_wrist_local(
        verts, geom_ids, geom_xpos, geom_xmat, wrist_pos, wrist_mat, out
    ):
        for i in range(verts.shape[0]):
            gid = int(geom_ids[i])
            rotation = geom_xmat[gid].reshape(3, 3)
            world = rotation @ verts[i] + geom_xpos[gid]
            out[i] = (world - wrist_pos) @ wrist_mat


_BUFFERS: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def acquire_eval_buffers(
    n_points: int, n_groups: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (int(n_points), int(n_groups))
    buffers = _BUFFERS.get(key)
    if buffers is None:
        buffers = (
            np.empty((n_points, 3), dtype=np.float64),
            np.empty((n_groups, STAT_DIM), dtype=np.float64),
            np.empty(n_points, dtype=np.float64),
        )
        _BUFFERS[key] = buffers
    return buffers


def pack_local_samples(
    local_samples: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    names = tuple(local_samples.keys())
    parts = [np.asarray(local_samples[name], dtype=np.float64) for name in names]
    packed = np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float64)
    offsets = np.ascontiguousarray(
        np.cumsum([0] + [part.shape[0] for part in parts], dtype=np.int64)
    )
    return packed, offsets, names


def stats_row_to_group(row: np.ndarray) -> dict[str, float | int | bool]:
    return {
        "surface_distance_min_m": float(row[STAT_SURFACE_MIN]),
        "contact_points": int(row[STAT_CONTACT_POINTS]),
        "contact": bool(row[STAT_CONTACT] >= 0.5),
        "penetrating_points": int(row[STAT_PENETRATING_POINTS]),
        "penetration_ratio": float(row[STAT_PENETRATION_RATIO]),
        "penetration_mean_positive_m": float(row[STAT_PENETRATION_MEAN]),
        "penetration_p95_m": float(row[STAT_PENETRATION_P95]),
        "penetration_max_m": float(row[STAT_PENETRATION_MAX]),
        "closest_point_x": float(row[STAT_CLOSEST_X]),
        "closest_point_y": float(row[STAT_CLOSEST_Y]),
        "closest_point_z": float(row[STAT_CLOSEST_Z]),
        "closest_radial_normal_x": float(row[STAT_NORMAL_X]),
        "closest_radial_normal_y": float(row[STAT_NORMAL_Y]),
    }


def split_world_samples(
    world: np.ndarray, offsets: np.ndarray, names: tuple[str, ...]
) -> dict[str, np.ndarray]:
    return {
        name: np.ascontiguousarray(world[int(offsets[i]) : int(offsets[i + 1])])
        for i, name in enumerate(names)
    }


def evaluate_packed_configuration(
    packed_local: np.ndarray,
    offsets: np.ndarray,
    names: tuple[str, ...],
    wrist_position: np.ndarray,
    wrist_rotation: np.ndarray,
    cup: Any,
    contact_threshold_m: float,
    penetration_tolerance_m: float,
    penetration_ratio_tolerance: float,
    *,
    return_world_samples: bool = True,
    world_buf: np.ndarray | None = None,
    stats_buf: np.ndarray | None = None,
    scratch_buf: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    packed = np.ascontiguousarray(packed_local, dtype=np.float64)
    offs = np.ascontiguousarray(offsets, dtype=np.int64)
    rotation = np.ascontiguousarray(wrist_rotation, dtype=np.float64)
    translation = np.ascontiguousarray(wrist_position, dtype=np.float64).reshape(3)
    n_points = int(packed.shape[0])
    n_groups = int(len(names))
    if world_buf is None or stats_buf is None or scratch_buf is None:
        world_buf, stats_buf, scratch_buf = acquire_eval_buffers(n_points, n_groups)
    world_view = world_buf[:n_points]
    stats_view = stats_buf[:n_groups]
    scratch_view = scratch_buf[:n_points]
    _world_tf(packed, rotation, translation, world_view)
    p95, mean_pos, ratio, pen_max = _cup_stats(
        world_view,
        offs,
        float(cup.center_xy[0]),
        float(cup.center_xy[1]),
        float(cup.z_reference),
        float(cup.radius_reference),
        float(cup.radius_slope),
        float(cup.z_min),
        float(cup.z_max),
        float(contact_threshold_m),
        stats_view,
        scratch_view,
    )
    groups = {names[i]: stats_row_to_group(stats_view[i]) for i in range(n_groups)}
    thumb = groups["thumb"]
    opposition = min(
        (groups["index"], groups["middle"]),
        key=lambda value: (
            not bool(value["contact"]),
            float(value["surface_distance_min_m"]),
        ),
    )
    normal_dot = float(
        thumb["closest_radial_normal_x"] * opposition["closest_radial_normal_x"]
        + thumb["closest_radial_normal_y"] * opposition["closest_radial_normal_y"]
    )
    bilateral = bool(
        thumb["contact"] and opposition["contact"] and normal_dot <= -0.20
    )
    thumb_distance = float(thumb["surface_distance_min_m"])
    opposition_distance = float(opposition["surface_distance_min_m"])
    valid_grasp = bool(
        bilateral
        and p95 <= penetration_tolerance_m
        and ratio <= penetration_ratio_tolerance
    )
    objective = (
        thumb_distance
        + opposition_distance
        + 2.0 * p95
        + 0.05 * ratio
        + 0.02 * max(normal_dot + 0.20, 0.0)
    )
    metric: dict[str, Any] = {
        "groups": groups,
        "thumb_surface_distance_m": thumb_distance,
        "opposition_surface_distance_m": opposition_distance,
        "opposition_normal_dot": normal_dot,
        "bilateral_contact": bilateral,
        "valid_grasp": valid_grasp,
        "hand_penetration_ratio": float(ratio),
        "hand_penetration_mean_positive_m": float(mean_pos),
        "hand_penetration_p95_m": float(p95),
        "hand_penetration_max_m": float(pen_max),
        "ranking_objective_m": float(objective),
    }
    world_samples = (
        split_world_samples(world_view, offs, names) if return_world_samples else {}
    )
    return metric, world_samples
