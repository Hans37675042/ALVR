import time

import numpy as np

from roomd.fusion import FusionConfig, TsdfFusion

from fusionkit import box_scene, first_frame, scan_poses


def test_integrate_latency_report():
    """Reports per-frame integration latency on the default volume; no pass/fail threshold."""
    fusion = TsdfFusion(FusionConfig())
    scene = box_scene()
    first_frame(fusion, scene)
    payloads = [scene.frame_payload(eye, target, width=320, height=320)
                for eye, target in scan_poses(60)]
    for p in payloads[:5]:  # warm-up
        fusion.integrate(p)
    wall = []
    for p in payloads:
        t0 = time.perf_counter()
        fusion.integrate(p)
        wall.append((time.perf_counter() - t0) * 1e3)
    t0 = time.perf_counter()
    out = fusion.snapshot_outputs(now=0.0, force=True)
    snap_ms = (time.perf_counter() - t0) * 1e3
    stats = fusion.stats()
    print("\nintegrate (decode+mask+tsdf, 2x320x320, %s voxels): p50 %.2f ms, p95 %.2f ms, max %.2f ms;"
          " stats p95 %.2f ms; full snapshot %d chunks %.1f ms"
          % (fusion.voxel_count(), np.percentile(wall, 50), np.percentile(wall, 95), max(wall),
             stats["integrateMsP95"], len(out.mesh_chunks), snap_ms))
    assert stats["integrateMsP95"] > 0
