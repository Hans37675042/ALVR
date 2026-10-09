import pytest

from roomd.fusion import FusionConfig, TsdfFusion

from fusionkit import box_scene, first_frame, scan, scan_poses


@pytest.fixture(scope="module")
def scanned():
    """A fusion volume that has seen the floor + box scene from 24 viewpoints."""
    fusion = TsdfFusion(FusionConfig())
    scene = box_scene()
    first_frame(fusion, scene)
    assert all(scan(fusion, scene, scan_poses()))
    return fusion, scene
