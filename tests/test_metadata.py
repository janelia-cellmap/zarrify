"""Tests that zarrify writes correct OME-NGFF metadata.

zarr3 output → OME-NGFF 0.5 (under the "ome" key)
"""
import json

import numpy as np
import pytest
import tensorstore as ts
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


def test_tiff_to_zarr3_ome_multiscales_structure(tmp_path, dask_client):
    """ome.multiscales has axes and datasets with coordinateTransformations."""
    data = np.random.randint(0, 255, (10, 32, 32), dtype=np.uint8)
    src = tmp_path / "img.tif"
    tifffile.imwrite(src, data)
    dest = tmp_path / "img.zarr"

    scale = [4.0, 8.0, 8.0]
    translation = [0.0, 0.0, 0.0]
    axes = ["z", "y", "x"]
    units = ["nanometer", "nanometer", "nanometer"]

    to_zarr(str(src), str(dest), dask_client,
            axes=axes, scale=scale, translation=translation, units=units,
            zarr_chunks=[10, 32, 32])

    root = zarr.open_group(zarr.storage.LocalStore(str(dest)), mode="r")
    ms = root.attrs["ome"]["multiscales"][0]

    assert [a["name"] for a in ms["axes"]] == axes
    assert len(ms["datasets"]) == 1
    assert ms["datasets"][0]["path"] == "s0"
    cts = ms["datasets"][0]["coordinateTransformations"]
    assert any(ct["type"] == "scale" and ct["scale"] == scale for ct in cts)
    assert any(ct["type"] == "translation" and ct["translation"] == translation for ct in cts)


def test_n5_to_zarr3_ome_metadata_version(tmp_path, dask_client):
    """N5 conversion output has ome.version == '0.5'."""
    shape = (10, 32, 32)
    data = np.random.randint(0, 255, shape, dtype=np.uint8)

    n5_path = tmp_path / "test.n5"
    ts.open({
        "driver": "n5",
        "kvstore": {"driver": "file", "path": str(n5_path)},
        "path": "s0",
        "metadata": {
            "dataType": np.dtype(data.dtype).name,
            "dimensions": list(shape),
            "blockSize": list(shape),
            "compression": {"type": "raw"},
        },
        "create": True,
        "open": True,
    }).result()[:].write(data).result()

    (n5_path / "attributes.json").write_text(json.dumps({
        "n5": "2.0.0",
        "axes": ["z", "y", "x"],
        "units": ["nm", "nm", "nm"],
        "scales": [[1.0, 1.0, 1.0]],
    }))
    s0_attrs_path = n5_path / "s0" / "attributes.json"
    s0_attrs = json.loads(s0_attrs_path.read_text()) if s0_attrs_path.exists() else {}
    s0_attrs["transform"] = {
        "scale": [1.0, 1.0, 1.0],
        "translate": [0.0, 0.0, 0.0],
        "units": ["nm", "nm", "nm"],
        "axes": ["z", "y", "x"],
    }
    s0_attrs_path.write_text(json.dumps(s0_attrs))

    dest = tmp_path / "test_n5.zarr"
    to_zarr(str(n5_path), str(dest), dask_client)

    root = zarr.open_group(zarr.storage.LocalStore(str(dest)), mode="r")
    assert "ome" in root.attrs, "expected 'ome' key on N5 conversion output root"
    assert root.attrs["ome"]["version"] == "0.5"
    ms = root.attrs["ome"]["multiscales"][0]
    assert [a["name"] for a in ms["axes"]] == ["z", "y", "x"]
    assert any(d["path"] == "s0" for d in ms["datasets"])
