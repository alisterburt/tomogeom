from typing import Tuple

from psygnal import EventedModel


class Sphere(EventedModel):
    """A sphere."""
    center: Tuple[float, float, float]
    radius: float
