from typing import List

import numpy as np
import scipy.spatial.transform
from psygnal import EventedModel
from pydantic import validator
import einops
from scipy.interpolate import splprep, splev
import pandas as pd
from scipy.spatial.transform import Rotation

from .base import _Geometry


class _NDimensionalFilament(EventedModel):
    points: np.ndarray

    class Config:
        arbitrary_types_allowed = True

    @property
    def _ndim(self) -> int:
        return self.points.shape[-1]

    @validator('points')
    def is_coordinate_array(cls, v):
        points = np.asarray(v, dtype=float)
        if points.ndim != 2:
            raise ValueError('must be an (n, d) array')
        return points

    @property
    def length(self):
        """Approximate length of the spline."""
        backbone_points = self._get_n_points_along_spline(n_points=10000)
        # calculate sum of line-segment lengths
        deltas = np.diff(backbone_points, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)
        return np.sum(segment_lengths)

    def _unstack_points(self) -> List[np.ndarray]:
        """Unstack an (n, d) array into a list of (d, ) arrays."""
        return [point for point in einops.rearrange(self.points, 'n d -> d n')]

    def _get_n_points_along_spline(self, n_points: int, delta: float = 0) -> np.ndarray:
        """Get n points along a cubic B-spline fit to the filament."""
        tck, _ = splprep(self._unstack_points(), s=0, k=3)
        u = np.linspace(0, 1, n_points) + delta
        points = splev(u, tck)
        return np.stack(points, axis=-1)  # (n, d)

    def _get_points_along_spline(self, separation: float, delta: float = 0):
        """Get points along a filament with a defined separation."""
        n_points = self.length / separation
        return self._get_n_points_along_spline(n_points=n_points, delta=delta)

    def _get_directions_along_spline(self, separation: float) -> np.ndarray:
        points = self._get_points_along_spline(separation=separation)
        shifted_points = self._get_points_along_spline(separation=separation, delta=1e-2)
        directions = shifted_points - points
        return directions / np.linalg.norm(directions, axis=1)


class _Filament3D(_NDimensionalFilament):
    """3D filament model with data validation."""

    @validator('points')
    def is_3d_coordinate_array(cls, v):
        points = np.asarray(v, dtype=float)
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError('must be an (n, 3) array')
        return points


class SimpleHelix(_Filament3D, _Geometry):
    """3D filament model, points are extracted along the backbone."""
    rise: float = 1
    twist_degrees: float = 0

    @property
    def particles(self) -> pd.DataFrame:
        positions = self._get_points_along_spline(separation=self.rise)
        particle_z_vectors = self._get_directions_along_spline(separation=self.rise)

        in_plane_rotations = np.deg2rad(np.arange(len(positions)) * self.twist_degrees)
        x_values = np.sin(np.deg2rad(in_plane_rotations))
        x_vectors = np.zeros(shape=(len(positions), 3))
        x_vectors[:, 0] = x_values








