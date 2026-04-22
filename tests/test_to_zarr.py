import itertools
import json
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import tensorstore as ts
import tifffile

from zarrify.to_zarr import to_zarr
from zarrify.utils.dask_utils import initialize_dask_client
from zarrify.utils.ts_utils import open_ts, zarr3_spec


@pytest.fixture
def create_test_file(tmp_path):
    """Factory fixture to create test files with specified format and dimensions."""
    def _create_file(ext: str, shape: tuple):
        data = np.random.rand(*shape).astype(np.uint8)
        file_path = tmp_path / f"test_image_{len(shape)}d.{ext}"

        if ext in ["tiff", "tif"]:
            tifffile.imwrite(file_path, data)
        elif ext == "mrc":
            with mrcfile.new(file_path, overwrite=True) as mrc:
                mrc.set_data(data.astype(data.dtype))
        else:
            raise ValueError("Unsupported file format")

        return file_path, data
    return _create_file


FORMATS = ['tif', 'tiff', 'mrc']
SHAPES = [(40, 50), (1, 30, 50)]  # 2D and 3D


@pytest.mark.parametrize("ext,shape", list(itertools.product(FORMATS, SHAPES)))
def test_to_zarr(create_test_file, ext, shape):
    src_path, expected_data = create_test_file(ext, shape)
    dest_path = Path(f"{src_path.with_suffix('')}.zarr")

    dask_client = initialize_dask_client('local')
    to_zarr(src_path, dest_path, dask_client)

    if src_path.suffix.lstrip('.') in ['tif', 'tiff']:
        src_data = tifffile.imread(src_path)
    elif src_path.suffix.lstrip('.') == 'mrc':
        with mrcfile.open(src_path, permissive=True) as mrc:
            src_data = mrc.data

    dest_data = open_ts(zarr3_spec(str(dest_path), 's0'))[:].read().result()
    assert np.array_equal(dest_data, src_data)
    assert np.array_equal(dest_data, expected_data)


def test_4d_tiff_rgb(tmp_path):
    """Test 4D TIFF with channel axis (c, z, y, x)."""
    shape = (3, 10, 64, 64)
    data = np.random.randint(0, 255, shape, dtype=np.uint8)

    src_path = tmp_path / "test_4d_rgb.tif"
    tifffile.imwrite(src_path, data)
    dest_path = tmp_path / "test_4d_rgb.zarr"

    dask_client = initialize_dask_client('local')
    to_zarr(str(src_path), str(dest_path), dask_client)

    src_data = tifffile.imread(src_path)
    dest_data = open_ts(zarr3_spec(str(dest_path), 's0'))[:].read().result()

    assert dest_data.shape == src_data.shape
    assert np.array_equal(dest_data, src_data)



def test_n5_to_zarr(tmp_path):
    """Test N5 group to zarr3 conversion."""
    shape = (10, 32, 32)
    chunks = (10, 32, 32)
    data = np.random.randint(0, 255, shape, dtype=np.uint8)

    # Create a minimal N5 store using TensorStore (zarr v3 dropped N5Store)
    n5_path = tmp_path / "test.n5"
    n5_arr_spec = {
        'driver': 'n5',
        'kvstore': {'driver': 'file', 'path': str(n5_path)},
        'path': 's0',
        'metadata': {
            'dataType': np.dtype(data.dtype).name,
            'dimensions': list(shape),
            'blockSize': list(chunks),
            'compression': {'type': 'raw'},
        },
        'create': True,
        'open': True,
    }
    ts.open(n5_arr_spec).result()[:].write(data).result()

    # Write N5 group and array attributes required by N5Group
    root_attrs = {
        'n5': '2.0.0',
        'axes': ['z', 'y', 'x'],
        'units': ['nm', 'nm', 'nm'],
        'scales': [[1.0, 1.0, 1.0]],
    }
    (n5_path / 'attributes.json').write_text(json.dumps(root_attrs))

    s0_attrs_path = n5_path / 's0' / 'attributes.json'
    s0_attrs = json.loads(s0_attrs_path.read_text()) if s0_attrs_path.exists() else {}
    s0_attrs['transform'] = {
        'scale': [1.0, 1.0, 1.0],
        'translate': [0.0, 0.0, 0.0],
        'units': ['nm', 'nm', 'nm'],
        'axes': ['z', 'y', 'x'],
    }
    s0_attrs_path.write_text(json.dumps(s0_attrs))

    dest_path = tmp_path / "test_n5.zarr"
    dask_client = initialize_dask_client('local')
    to_zarr(str(n5_path), str(dest_path), dask_client)

    dest_data = open_ts(zarr3_spec(str(dest_path), 's0'))[:].read().result()
    assert dest_data.shape == data.shape
    assert np.array_equal(dest_data, data)
