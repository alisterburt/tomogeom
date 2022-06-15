import numpy as np
import pytest

from tomogeom.geometries.filament import Filament


@pytest.fixture
def filament_backbone_data():
    z = np.linspace(0, 2*np.pi, 200)
    x = np.cos(z)
    y = np.sin(z)
    return np.stack((x, y, z), axis=-1)


def test_filament_instantiation(filament_backbone_data):
    f = Filament(points=filament_backbone_data)
    with pytest.raises(ValueError):
        f = Filament(points=np.random.random(size=(3, 4, 5, 6)))  # 4d


def test_filament_length(filament_backbone_data):
    f = Filament(points=filament_backbone_data)
    np.testing.assert_array_almost_equal(f.length, 8.88, decimal=2)
