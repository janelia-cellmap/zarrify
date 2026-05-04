"""Tests that zarrify writes correct OME-NGFF metadata.

zarr3 output → OME-NGFF 0.5 (under the "ome" key)
"""
import numpy as np
import pytest
import tifffile
import zarr

from zarrify.to_zarr import to_zarr
from zarrify.utils.dask_utils import initialize_dask_client


@pytest.fixture(scope="module")
def dask_client():
    client = initialize_dask_client("local")
    yield client
    client.close()


def test_tiff_to_zarr3_ome_metadata_version(tmp_path, dask_client):
    """Output store has ome.version == '0.5'."""
    data = np.random.randint(0, 255, (10, 32, 32), dtype=np.uint8)
    src = tmp_path / "img.tif"
    tifffile.imwrite(src, data)
    dest = tmp_path / "img.zarr"

    to_zarr(str(src), str(dest), dask_client,
            axes=["z", "y", "x"], scale=[1.0, 1.0, 1.0],
            translation=[0.0, 0.0, 0.0], units=["nanometer", "nanometer", "nanometer"],
            zarr_chunks=[10, 32, 32])

    root = zarr.open_group(zarr.storage.LocalStore(str(dest)), mode="r")
    assert "ome" in root.attrs, "expected 'ome' key for 0.5 metadata"
    assert root.attrs["ome"]["version"] == "0.5"
