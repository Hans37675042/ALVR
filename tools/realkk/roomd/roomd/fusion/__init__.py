"""Depth fusion for roomd: MSG_DEPTH_FRAME_V2 -> dense TSDF -> mesh chunks, heightmap, queries."""

from .config import FusionConfig
from .decode import DepthFrame, DepthView, decode_depth_frame
from .outputs import (FloorPlane, FusionOutputs, Heightmap, MeshChunk, encode_mesh_chunk,
                      encode_nav_heightmap)
from .tsdf import Obb, TsdfFusion



def create_sink():
    """roomd --fusion roomd.fusion:create_sink (imports roomd.sink lazily)."""
    from .sink import create_sink as factory

    return factory()


__all__ = [
    "create_sink",
    "DepthFrame", "DepthView", "FloorPlane", "FusionConfig", "FusionOutputs", "Heightmap",
    "MeshChunk", "Obb", "TsdfFusion", "decode_depth_frame", "encode_mesh_chunk", "encode_nav_heightmap",
]
