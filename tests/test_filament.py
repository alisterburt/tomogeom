import numpy as np
import pytest

from tomogeom.geometries.filament import Filament, extract_regularly_spaced_points


@pytest.fixture
def filament_backbone_data_3d():
    z = np.linspace(0, 2*np.pi, 200)
    x = np.cos(z)
    y = np.sin(z)
    return np.stack((x, y, z), axis=-1)


def test_filament_instantiation(filament_backbone_data_3d):
    f = Filament(points=filament_backbone_data_3d)
    with pytest.raises(ValueError):
        f = Filament(points=np.random.random(size=(3, 4, 5, 6)))  # 4d data


def test_filament_length(filament_backbone_data_3d):
    f = Filament(points=filament_backbone_data_3d)
    np.testing.assert_array_almost_equal(f.length, 8.88, decimal=2)

def test_point_extraction(filament_backbone_data_3d):

