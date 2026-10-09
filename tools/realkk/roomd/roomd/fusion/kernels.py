"""Warp kernels for the dense TSDF volume.

Volume layout: tsdf/weight/last_seen are (nx, ny, nz) arrays, voxel (i, j, k) centred at
origin + (i, j, k) * voxel in Unity stage space. Integration threads own one (x, z) column
inside one chunk layer and walk its voxels in y, so the change odometer needs one atomic
per thread instead of one per voxel.
"""

import warp as wp


@wp.func
def unity_to_xr(p: wp.vec3):
    return wp.vec3(p[0], p[1], -p[2])


@wp.func
def project(p_xr: wp.vec3, rot_wc: wp.mat33, cam_pos: wp.vec3, intr: wp.vec4):
    """OpenXR world point -> (u, v, view depth); view depth <= 0 means behind the camera."""
    pc = rot_wc * (p_xr - cam_pos)
    zc = -pc[2]
    if zc <= 1.0e-4:
        return wp.vec3(0.0, 0.0, -1.0)
    return wp.vec3(intr[0] * pc[0] / zc + intr[2], intr[1] * (-pc[1]) / zc + intr[3], zc)


@wp.func
def backproject(u: int, v: int, d: float, intr: wp.vec4):
    return wp.vec3((float(u) + 0.5 - intr[2]) / intr[0] * d, -(float(v) + 0.5 - intr[3]) / intr[1] * d, -d)


@wp.kernel
def mask_depth(depth: wp.array3d(dtype=float), out: wp.array3d(dtype=float),
               intr: wp.array(dtype=wp.vec4), rot_cw: wp.array(dtype=wp.mat33),
               cam_pos: wp.array(dtype=wp.vec3), head: wp.vec3,
               crop_u: int, crop_v: int, min_d: float, max_d: float, cos_max_inc: float,
               body_r2: float, body_top: float, min_y: float):
    view, v, u = wp.tid()
    h = depth.shape[1]
    w = depth.shape[2]
    d = depth[view, v, u]
    keep = d > 0.0
    if u < crop_u or u >= w - crop_u or v < crop_v or v >= h - crop_v:
        keep = False
    if d < min_d or d > max_d:
        keep = False
    if keep:
        k = intr[view]
        p = backproject(u, v, d, k)
        # incidence from the right/down neighbours, falling back to left/up at the edge
        du = 1
        dv = 1
        if u + 1 >= w:
            du = -1
        if v + 1 >= h:
            dv = -1
        d1 = depth[view, v, u + du]
        d2 = depth[view, v + dv, u]
        if d1 <= 0.0 or d2 <= 0.0:
            keep = False
        else:
            n = wp.cross(backproject(u + du, v, d1, k) - p, backproject(u, v + dv, d2, k) - p)
            ln = wp.length(n)
            if ln <= 0.0 or wp.abs(wp.dot(n, p)) < cos_max_inc * ln * wp.length(p):
                keep = False
        if keep:
            wx = rot_cw[view] * p + cam_pos[view]
            dx = wx[0] - head[0]
            dz = -wx[2] - head[2]
            if dx * dx + dz * dz < body_r2 and wx[1] < body_top:
                keep = False
            # nothing exists below the stage floor; such samples would carve the floor away
            if wx[1] < min_y:
                keep = False
    if keep:
        out[view, v, u] = d
    else:
        out[view, v, u] = 0.0


@wp.kernel
def integrate_depth(tsdf: wp.array3d(dtype=float), weight: wp.array3d(dtype=float),
                    last_seen: wp.array3d(dtype=wp.int32), odometer: wp.array3d(dtype=float),
                    depth: wp.array3d(dtype=float), intr: wp.array(dtype=wp.vec4),
                    rot_wc: wp.array(dtype=wp.mat33), cam_pos: wp.array(dtype=wp.vec3),
                    n_views: int, origin: wp.vec3, voxel: float, cv: int, trunc: float,
                    max_w: float, frame: int, odo_eps: float, conflict_t: float,
                    conflict_decay: float):
    i, cj, k = wp.tid()
    h = depth.shape[1]
    w = depth.shape[2]
    acc = float(0.0)
    for jj in range(cv):
        j = cj * cv + jj
        p = unity_to_xr(origin + wp.vec3(float(i), float(j), float(k)) * voxel)
        t = tsdf[i, j, k]
        wt = weight[i, j, k]
        t0 = t
        seen = int(0)
        for view in range(n_views):
            uvz = project(p, rot_wc[view], cam_pos[view], intr[view])
            if uvz[2] > 0.0:
                iu = int(wp.floor(uvz[0]))
                iv = int(wp.floor(uvz[1]))
                if iu >= 0 and iu < w and iv >= 0 and iv < h:
                    d = depth[view, iv, iu]
                    if d > 0.0:
                        pc = rot_wc[view] * (p - cam_pos[view])
                        sdf = (d - uvz[2]) * wp.length(pc) / uvz[2]
                        if sdf >= -trunc:
                            obs = wp.min(1.0, sdf / trunc)
                            # the scene changed: a confident value contradicted by the opposite
                            # sign loses weight, so moved furniture appears / clears in a few frames
                            if (t > conflict_t and obs < 0.0) or (t < -conflict_t and obs > 0.0):
                                wt = wt * conflict_decay
                            t = (t * wt + obs) / (wt + 1.0)
                            wt = wp.min(wt + 1.0, max_w)
                            seen = 1
        if seen != 0:
            tsdf[i, j, k] = t
            weight[i, j, k] = wt
            last_seen[i, j, k] = frame
            dt = wp.abs(t - t0)
            if dt > odo_eps:
                acc += dt
    if acc > 0.0:
        wp.atomic_add(odometer, i / cv, cj, k / cv, acc)


@wp.kernel
def integrate_mesh_prior(tsdf: wp.array3d(dtype=float), weight: wp.array3d(dtype=float),
                         odometer: wp.array3d(dtype=float), mesh: wp.uint64, origin: wp.vec3,
                         voxel: float, cv: int, trunc: float, prior_w: float, max_w: float):
    i, cj, k = wp.tid()
    acc = float(0.0)
    for jj in range(cv):
        j = cj * cv + jj
        p = origin + wp.vec3(float(i), float(j), float(k)) * voxel
        q = wp.mesh_query_point_sign_normal(mesh, p, trunc)
        if q.result:
            closest = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
            s = q.sign * wp.length(p - closest)
            if s > -trunc and s < trunc:
                t = tsdf[i, j, k]
                wt = weight[i, j, k]
                t1 = (t * wt + s / trunc * prior_w) / (wt + prior_w)
                tsdf[i, j, k] = t1
                weight[i, j, k] = wp.min(wt + prior_w, max_w)
                dt = wp.abs(t1 - t)
                acc += dt
    if acc > 0.0:
        wp.atomic_add(odometer, i / cv, cj, k / cv, acc)


@wp.func
def column_crossing(tsdf: wp.array3d(dtype=float), weight: wp.array3d(dtype=float),
                    i: int, j: int, k: int):
    """Fraction in [0, 1) above voxel j where an upward surface crosses zero, or -1."""
    if weight[i, j, k] <= 0.0 or weight[i, j + 1, k] <= 0.0:
        return -1.0
    lo = tsdf[i, j, k]
    hi = tsdf[i, j + 1, k]
    if lo <= 0.0 and hi > 0.0 and hi - lo < 1.0:
        return -lo / (hi - lo)
    return -1.0


@wp.kernel
def upward_surface_points(tsdf: wp.array3d(dtype=float), weight: wp.array3d(dtype=float),
                          origin: wp.vec3, voxel: float, j_lo: int, j_hi: int, min_ny: float,
                          count: wp.array(dtype=wp.int32), points: wp.array(dtype=wp.vec3),
                          normals: wp.array(dtype=wp.vec3)):
    i, k = wp.tid()
    nx = tsdf.shape[0]
    nz = tsdf.shape[2]
    if i < 1 or k < 1 or i >= nx - 1 or k >= nz - 1:
        return
    for j in range(j_lo, j_hi):
        f = column_crossing(tsdf, weight, i, j, k)
        if f >= 0.0:
            n = wp.vec3(tsdf[i + 1, j, k] + tsdf[i + 1, j + 1, k] - tsdf[i - 1, j, k] - tsdf[i - 1, j + 1, k],
                        2.0 * (tsdf[i, j + 1, k] - tsdf[i, j, k]),
                        tsdf[i, j, k + 1] + tsdf[i, j + 1, k + 1] - tsdf[i, j, k - 1] - tsdf[i, j + 1, k - 1])
            ln = wp.length(n)
            if ln > 0.0:
                n = n / ln
                if n[1] >= min_ny:
                    slot = wp.atomic_add(count, 0, 1)
                    if slot < points.shape[0]:
                        points[slot] = origin + wp.vec3(float(i), float(j) + f, float(k)) * voxel
                        normals[slot] = n


@wp.kernel
def heightmap_cells(tsdf: wp.array3d(dtype=float), weight: wp.array3d(dtype=float),
                    origin: wp.vec3, voxel: float, hm_origin_x: float, hm_origin_z: float, cell: float,
                    floor_y: float, floor_tol: float, obs_min: float, obs_max: float, min_vox: int,
                    floor_out: wp.array2d(dtype=float), top_out: wp.array2d(dtype=float),
                    flags_out: wp.array2d(dtype=wp.uint8)):
    hz, hx = wp.tid()
    nx = tsdf.shape[0]
    ny = tsdf.shape[1]
    nz = tsdf.shape[2]
    x0 = hm_origin_x + float(hx) * cell
    z0 = hm_origin_z + float(hz) * cell
    i0 = wp.max(int(wp.ceil((x0 - origin[0]) / voxel - 1.0e-4)), 0)
    i1 = wp.min(int(wp.ceil((x0 + cell - origin[0]) / voxel - 1.0e-4)), nx)
    k0 = wp.max(int(wp.ceil((z0 - origin[2]) / voxel - 1.0e-4)), 0)
    k1 = wp.min(int(wp.ceil((z0 + cell - origin[2]) / voxel - 1.0e-4)), nz)
    j_lo = wp.max(int(wp.floor((floor_y - floor_tol - 0.05 - origin[1]) / voxel)), 0)
    j_hi = wp.min(int(wp.ceil((floor_y + obs_max - origin[1]) / voxel)), ny - 2)
    known = int(0)
    floor_n = int(0)
    floor_sum = float(0.0)
    occ = int(0)
    top = float(-1.0e9)
    for i in range(i0, i1):
        for k in range(k0, k1):
            for j in range(j_lo, j_hi + 1):
                y = origin[1] + float(j) * voxel
                if weight[i, j, k] > 0.0:
                    known = 1
                    if tsdf[i, j, k] <= 0.0 and y >= floor_y + obs_min:
                        occ += 1
                        top = wp.max(top, y)
                f = column_crossing(tsdf, weight, i, j, k)
                if f >= 0.0:
                    yc = y + f * voxel
                    if wp.abs(yc - floor_y) <= floor_tol:
                        floor_n += 1
                        floor_sum += yc
                    elif yc >= floor_y + obs_min and yc <= floor_y + obs_max:
                        occ += min_vox
                        top = wp.max(top, yc)
    fy = floor_y
    if floor_n > 0:
        fy = floor_sum / float(floor_n)
    flags = int(0)
    if known != 0:
        flags = flags | 1
    obstacle = occ >= min_vox
    if obstacle:
        flags = flags | 2
    elif known != 0 and floor_n > 0:
        flags = flags | 4
    floor_out[hz, hx] = fy
    if obstacle:
        top_out[hz, hx] = top
    else:
        top_out[hz, hx] = fy
    flags_out[hz, hx] = wp.uint8(flags)


@wp.kernel
def obb_counts(tsdf: wp.array3d(dtype=float), weight: wp.array3d(dtype=float),
               last_seen: wp.array3d(dtype=wp.int32), origin: wp.vec3, voxel: float,
               lo_idx: wp.vec3i, center: wp.vec3, rot_inv: wp.mat33, half: wp.vec3, min_frame: int,
               counts: wp.array(dtype=wp.int32)):
    a, b, c = wp.tid()
    i = lo_idx[0] + a
    j = lo_idx[1] + b
    k = lo_idx[2] + c
    if i < 0 or j < 0 or k < 0 or i >= tsdf.shape[0] or j >= tsdf.shape[1] or k >= tsdf.shape[2]:
        return
    p = origin + wp.vec3(float(i), float(j), float(k)) * voxel
    q = rot_inv * (p - center)
    if wp.abs(q[0]) > half[0] or wp.abs(q[1]) > half[1] or wp.abs(q[2]) > half[2]:
        return
    wp.atomic_add(counts, 0, 1)
    if weight[i, j, k] > 0.0:
        if tsdf[i, j, k] > 0.0:
            wp.atomic_add(counts, 1, 1)
        else:
            wp.atomic_add(counts, 2, 1)
    if last_seen[i, j, k] >= min_frame:
        wp.atomic_add(counts, 3, 1)


@wp.kernel
def copy_block(src: wp.array3d(dtype=float), dst: wp.array3d(dtype=float), i0: int, j0: int, k0: int):
    a, b, c = wp.tid()
    dst[a, b, c] = src[i0 + a, j0 + b, k0 + c]


@wp.kernel
def mesh_cells(src_t: wp.array3d(dtype=float), src_w: wp.array3d(dtype=float),
               field: wp.array3d(dtype=float), ok: wp.array3d(dtype=wp.uint8),
               i0: int, j0: int, k0: int, count: wp.array(dtype=wp.int32)):
    """Copy one chunk (+1 node seam) of the TSDF into field and flag the cells marching
    cubes may emit: all 8 corners observed and no truncation jump (-1 -> +1) inside."""
    a, b, c = wp.tid()
    field[a, b, c] = src_t[i0 + a, j0 + b, k0 + c]
    if a + 1 >= field.shape[0] or b + 1 >= field.shape[1] or c + 1 >= field.shape[2]:
        return
    lo = float(1.0e9)
    hi = float(-1.0e9)
    good = int(1)
    for d in range(8):
        i = i0 + a + d % 2
        j = j0 + b + (d / 2) % 2
        k = k0 + c + d / 4
        if src_w[i, j, k] <= 0.0:
            good = 0
        t = src_t[i, j, k]
        lo = wp.min(lo, t)
        hi = wp.max(hi, t)
    if good != 0 and hi - lo < 1.0 and lo <= 0.0 and hi > 0.0:
        ok[a, b, c] = wp.uint8(1)
        wp.atomic_add(count, 0, 1)
    else:
        ok[a, b, c] = wp.uint8(0)
