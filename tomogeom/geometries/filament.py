import numpy as np
from psygnal import EventedModel
from pydantic import validator
import einops
from scipy.interpolate import splprep, splev


class Filament(EventedModel):
    points: np.ndarray

    class Config:
        arbitrary_types_allowed = True

    @validator('points')
    def is_3d_coordinate_array(cls, v):
        points = np.asarray(v, dtype=float)
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError('must be an (n, 3) array')
        return points

    @property
    def length(self):
        """Approximate length of the spline."""
        # fit cubic B-spline with 10000 points
        x, y, z = einops.rearrange(self.points, 'b c -> c b')  # unstack
        tck, _ = splprep((x, y, z), s=0)
        x, y, z = splev(np.linspace(0, 1, 10000), tck)
        xyz = np.stack((x, y, z), axis=-1)  # (n, 3)
        # calculate sum of line-segment lengths
        deltas = np.diff(xyz, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)
        return np.sum(segment_lengths)



