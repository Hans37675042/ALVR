"""Depth sensor artefacts for synthetic frames (all on linear depth z, inf = invalid).

Quantisation is not simulated here: it happens for real when z is encoded to D16.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseModel:
    sigma_k: float = 0.0025  # sigma = k * z^2 (m): 1 cm at 2 m
    edge_jump: float = 0.08  # relative depth jump that counts as an edge
    edge_hole_p: float = 0.5  # edge pixel becomes a hole
    flying_p: float = 0.3  # remaining edge pixel gets a depth between fg and bg
    dropout_p: float = 0.003  # random isolated holes
    hand_p: float = 0.15  # probability a frame shows a hand blob
    hand_radius_px: float = 0.08  # fraction of the image width
    max_range: float = 5.0  # beyond this the sensor reports nothing

    def apply(self, z, rng, hand=None):
        """Return a noisy copy of z. hand: None = random by hand_p, True/False to force."""
        z = z.copy()
        valid = np.isfinite(z)
        z[valid] += rng.normal(0.0, 1.0, valid.sum()) * self.sigma_k * z[valid] ** 2
        edges = self._edges(z)
        holes = edges & (rng.random(z.shape) < self.edge_hole_p)
        flying = edges & ~holes & (rng.random(z.shape) < self.flying_p)
        if flying.any():
            far = self._neighbour_max(z)
            near = self._neighbour_min(z)
            mix = rng.random(z.shape)
            z[flying] = (near + (far - near) * mix)[flying]
        z[holes] = np.inf
        z[rng.random(z.shape) < self.dropout_p] = np.inf
        z[z > self.max_range] = np.inf
        if hand or (hand is None and rng.random() < self.hand_p):
            self._hand(z, rng)
        return z

    def _edges(self, z):
        zz = np.where(np.isfinite(z), z, 1e3)
        jump = np.zeros(z.shape, bool)
        for axis in (0, 1):
            d = np.abs(np.diff(zz, axis=axis))
            rel = d / np.minimum(np.delete(zz, -1, axis=axis), np.delete(zz, 0, axis=axis))
            e = rel > self.edge_jump
            if axis == 0:
                jump[:-1] |= e
                jump[1:] |= e
            else:
                jump[:, :-1] |= e
                jump[:, 1:] |= e
        return jump & np.isfinite(z)

    @staticmethod
    def _shifted(z):
        pad = np.pad(np.where(np.isfinite(z), z, np.nan), 1, mode="edge")
        h, w = z.shape
        return np.stack([pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])

    def _neighbour_max(self, z):
        return np.nan_to_num(np.nanmax(self._shifted(z), axis=0), nan=np.inf)

    def _neighbour_min(self, z):
        return np.nan_to_num(np.nanmin(self._shifted(z), axis=0), nan=np.inf)

    def _hand(self, z, rng):
        h, w = z.shape
        r = self.hand_radius_px * w
        cu = rng.uniform(0.25, 0.75) * w
        cv = rng.uniform(0.6, 0.95) * h
        dist = rng.uniform(0.3, 0.5)
        vv, uu = np.mgrid[0:h, 0:w]
        rr = np.hypot(uu + 0.5 - cu, vv + 0.5 - cv) / r
        inside = rr < 1.0
        z[inside] = np.minimum(z[inside], dist - 0.04 * np.sqrt(1 - rr[inside] ** 2))
