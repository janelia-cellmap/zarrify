from pathlib import Path

import pytest

import mrcfile
import tifffile
import zarr
import numpy as np
import os

from zarrify.utils.dask_utils import initialize_dask_client
from zarrify import to_zarr

@pytest.fixture
def create_test_file(tmp_path):
    """Factory fixture to create test files with specified format and dimensions"""
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

import itertools

# Test parameters
FORMATS = ['tif', 'tiff', 'mrc']
SHAPES = [(40, 50), (1, 30, 50)]  # 2D and 3D

@pytest.mark.parametrize("ext,shape", list(itertools.product(FORMATS, SHAPES)))
def test_to_zarr(create_test_file, ext, shape):
    
    src_path, expected_data = create_test_file(ext, shape)
    dest_path = Path(f"{src_path.with_suffix('')}.zarr")
    
    dask_client = initialize_dask_client('local')
    
    # convert to zarr
    to_zarr(src_path, dest_path, dask_client)
    
    if src_path.suffix.lstrip('.') in ['tif', 'tiff']:
        src_data = tifffile.imread(src_path)
    elif src_path.suffix.lstrip('.') == 'mrc':
        with mrcfile.open(src_path, permissive=True) as mrc:
            src_data = mrc.data
    
    # store array in s0 by convention
    dest_data = zarr.open(f'{dest_path}/s0', mode='r')
    assert np.array_equal(dest_data[:], src_data)
    assert np.array_equal(dest_data[:], expected_data)


def test_3d_tiff_rgb(tmp_path):
    """Test 3D TIFF with RGB channels"""
    
    # Create 3D RGB data: (channels, depth, height, width)
    shape = (3, 10, 64, 64)
    data = np.random.randint(0, 255, shape, dtype=np.uint8)
    
    # Create test file
    src_path = tmp_path / "test_3d_rgb.tif"
    tifffile.imwrite(src_path, data)
    dest_path = tmp_path / "test_3d_rgb.zarr"
    
    dask_client = initialize_dask_client('local')
    
    # Convert to zarr
    to_zarr(str(src_path), str(dest_path), dask_client)
    
    # Verify conversion
    src_data = tifffile.imread(src_path)
    dest_data = zarr.open(f'{dest_path}/s0', mode='r')
    
    assert dest_data.shape == src_data.shape
    assert np.array_equal(dest_data[:], src_data)
    
