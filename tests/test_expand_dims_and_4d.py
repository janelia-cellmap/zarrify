"""Tests for expand_dims (3D to 4D casting) and native 4D dataset conversion."""
import json

import mrcfile
import numpy as np
import pytest
import tensorstore as ts
import tifffile
import zarr

from zarrify.to_zarr import to_zarr
from zarrify.utils.dask_utils import initialize_dask_client
from zarrify.utils.ts_utils import open_ts, zarr3_spec


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dask_client():
    return initialize_dask_client("local")


def _make_n5(tmp_path, shape, data, chunks=None):
    """Create a minimal N5 store with OME-style group attributes."""
    if chunks is None:
        chunks = shape
    n5_path = tmp_path / "test.n5"
    ndim = len(shape)
    axes = ["c", "z", "y", "x"][-ndim:]
    units = ["", "nm", "nm", "nm"][-ndim:]

    spec = {
        "driver": "n5",
        "kvstore": {"driver": "file", "path": str(n5_path)},
        "path": "s0",
        "metadata": {
            "dataType": np.dtype(data.dtype).name,
            "dimensions": list(shape),
            "blockSize": list(chunks),
            "compression": {"type": "raw"},
        },
        "create": True,
        "open": True,
    }
    ts.open(spec).result()[:].write(data).result()

    root_attrs = {
        "n5": "2.0.0",
        "axes": axes,
        "units": units,
        "scales": [[1.0] * ndim],
    }
    (n5_path / "attributes.json").write_text(json.dumps(root_attrs))

    s0_attrs_path = n5_path / "s0" / "attributes.json"
    s0_attrs = json.loads(s0_attrs_path.read_text()) if s0_attrs_path.exists() else {}
    s0_attrs["transform"] = {
        "scale": [1.0] * ndim,
        "translate": [0.0] * ndim,
        "units": units,
        "axes": axes,
    }
    s0_attrs_path.write_text(json.dumps(s0_attrs))
    return n5_path


def _make_zarr2(tmp_path, shape, data, chunks=None):
    """Create a minimal zarr v2 store with OME multiscales attributes."""
    if chunks is None:
        chunks = shape
    ndim = len(shape)
    axes = ["c", "z", "y", "x"][-ndim:]
    src_path = tmp_path / "src.zarr"

    spec = {
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(src_path)},
        "path": "s0",
        "metadata": {
            "dtype": np.dtype(data.dtype).str,
            "shape": list(shape),
            "chunks": list(chunks),
        },
        "create": True,
        "open": True,
    }
    ts.open(spec).result()[:].write(data).result()

    multiscales = [{
        "version": "0.4",
        "axes": [{"name": ax, "type": "space", "unit": "nanometer"} for ax in axes],
        "datasets": [{
            "path": "s0",
            "coordinateTransformations": [
                {"type": "scale", "scale": [1.0] * ndim},
                {"type": "translation", "translation": [0.0] * ndim},
            ],
        }],
    }]
    (src_path / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (src_path / ".zattrs").write_text(json.dumps({"multiscales": multiscales}))
    return src_path


def _make_tiff_stack(tmp_path, slices):
    """Write a directory of 2D TIFF tiles and return the directory path."""
    stack_dir = tmp_path / "stack"
    stack_dir.mkdir()
    for i, sl in enumerate(slices):
        tifffile.imwrite(stack_dir / f"slice_{i:04d}.tif", sl)
    return stack_dir


# ---------------------------------------------------------------------------
# expand_dims: Tiff, TiffStack, MRC
# ---------------------------------------------------------------------------

def test_expand_dims_tiff(tmp_path, dask_client):
    shape = (4, 8, 8)
    data = np.random.randint(0, 200, shape, dtype=np.uint8)
    src = tmp_path / "vol.tif"
    tifffile.imwrite(src, data)
    dest = tmp_path / "out.zarr"

    to_zarr(str(src), str(dest), dask_client, expand_dims=True)

    result = open_ts(zarr3_spec(str(dest), "s0"))[:].read().result()
    assert result.shape == (1, *shape)
    assert np.array_equal(result[0], data)

    root = zarr.open_group(zarr.storage.LocalStore(str(dest)), mode="r")
    ms_axes = root.attrs["multiscales"][0]["axes"]
    assert len(ms_axes) == 4
    assert ms_axes[0]["name"] == "c"
    assert ms_axes[0]["type"] == "channel"


def test_expand_dims_tiff_stack(tmp_path, dask_client):
    n_slices, h, w = 5, 8, 8
    data = np.random.randint(0, 200, (n_slices, h, w), dtype=np.uint8)
    stack_dir = _make_tiff_stack(tmp_path, data)
    dest = tmp_path / "out.zarr"

    to_zarr(str(stack_dir), str(dest), dask_client, expand_dims=True)

    result = open_ts(zarr3_spec(str(dest), "s0"))[:].read().result()
    assert result.shape == (1, n_slices, h, w)
    assert np.array_equal(result[0], data)

    root = zarr.open_group(zarr.storage.LocalStore(str(dest)), mode="r")
    ms_axes = root.attrs["multiscales"][0]["axes"]
    assert ms_axes[0]["name"] == "c"


def test_expand_dims_mrc(tmp_path, dask_client):
    shape = (4, 8, 8)
    data = np.random.randint(0, 200, shape, dtype=np.uint8)
    src = tmp_path / "vol.mrc"
    with mrcfile.new(src, overwrite=True) as mrc:
        mrc.set_data(data)
    dest = tmp_path / "out.zarr"

    to_zarr(str(src), str(dest), dask_client, expand_dims=True)

    result = open_ts(zarr3_spec(str(dest), "s0"))[:].read().result()
    assert result.shape == (1, *shape)
    assert np.array_equal(result[0], data)

    root = zarr.open_group(zarr.storage.LocalStore(str(dest)), mode="r")
    ms_axes = root.attrs["multiscales"][0]["axes"]
    assert ms_axes[0]["name"] == "c"


# ---------------------------------------------------------------------------
# expand_dims: Zarr2 - data and OME metadata
# ---------------------------------------------------------------------------

def test_expand_dims_zarr2(tmp_path, dask_client):
    shape = (4, 8, 8)
    data = np.random.randint(0, 200, shape, dtype=np.uint8)
    src = _make_zarr2(tmp_path, shape, data)
    dest = tmp_path / "out.zarr"

    to_zarr(str(src), str(dest), dask_client, expand_dims=True)

    result = open_ts(zarr3_spec(str(dest), "s0"))[:].read().result()
    assert result.shape == (1, *shape)
    assert np.array_equal(result[0], data)

    root = zarr.open_group(zarr.storage.LocalStore(str(dest)), mode="r")
    ms = root.attrs["multiscales"][0]
    assert len(ms["axes"]) == 4
    assert ms["axes"][0]["name"] == "c"
    scale = ms["datasets"][0]["coordinateTransformations"][0]["scale"]
    assert len(scale) == 4
    assert scale[0] == 1.0
    translation = ms["datasets"][0]["coordinateTransformations"][1]["translation"]
    assert len(translation) == 4
    assert translation[0] == 0.0


# ---------------------------------------------------------------------------
# expand_dims: N5 - data and OME metadata
# ---------------------------------------------------------------------------

def test_expand_dims_n5(tmp_path, dask_client):
    shape = (4, 8, 8)
    data = np.random.randint(0, 200, shape, dtype=np.uint8)
    n5_path = _make_n5(tmp_path, shape, data)
    dest = tmp_path / "out.zarr"

    to_zarr(str(n5_path), str(dest), dask_client, expand_dims=True)

    result = open_ts(zarr3_spec(str(dest), "s0"))[:].read().result()
    assert result.shape == (1, *shape)
    assert np.array_equal(result[0], data)

    root = zarr.open_group(zarr.storage.LocalStore(str(dest)), mode="r")
    ms = root.attrs["multiscales"][0]
    assert len(ms["axes"]) == 4
    assert ms["axes"][0]["name"] == "c"
    assert ms["axes"][0]["type"] == "channel"
    scale = ms["datasets"][0]["coordinateTransformations"][0]["scale"]
    assert len(scale) == 4
    assert scale[0] == 1.0
    translation = ms["datasets"][0]["coordinateTransformations"][1]["translation"]
    assert len(translation) == 4
    assert translation[0] == 0.0


# ---------------------------------------------------------------------------
# Native 4D datasets (no expand_dims)
# ---------------------------------------------------------------------------

def test_4d_n5(tmp_path, dask_client):
    shape = (2, 4, 8, 8)
    data = np.random.randint(0, 200, shape, dtype=np.uint8)
    n5_path = _make_n5(tmp_path, shape, data)
    dest = tmp_path / "out.zarr"

    to_zarr(str(n5_path), str(dest), dask_client)

    result = open_ts(zarr3_spec(str(dest), "s0"))[:].read().result()
    assert result.shape == shape
    assert np.array_equal(result, data)

    root = zarr.open_group(zarr.storage.LocalStore(str(dest)), mode="r")
    ms_axes = root.attrs["multiscales"][0]["axes"]
    assert len(ms_axes) == 4


def test_4d_zarr2(tmp_path, dask_client):
    shape = (2, 4, 8, 8)
    data = np.random.randint(0, 200, shape, dtype=np.uint8)
    src = _make_zarr2(tmp_path, shape, data)
    dest = tmp_path / "out.zarr"

    to_zarr(str(src), str(dest), dask_client)

    result = open_ts(zarr3_spec(str(dest), "s0"))[:].read().result()
    assert result.shape == shape
    assert np.array_equal(result, data)

    root = zarr.open_group(zarr.storage.LocalStore(str(dest)), mode="r")
    ms_axes = root.attrs["multiscales"][0]["axes"]
    assert len(ms_axes) == 4


def test_4d_tiff(tmp_path, dask_client):
    shape = (2, 4, 8, 8)
    data = np.random.randint(0, 200, shape, dtype=np.uint8)
    src = tmp_path / "vol_4d.tif"
    tifffile.imwrite(src, data)
    dest = tmp_path / "out.zarr"

    to_zarr(str(src), str(dest), dask_client)

    result = open_ts(zarr3_spec(str(dest), "s0"))[:].read().result()
    assert result.shape == shape
    assert np.array_equal(result, data)


def test_4d_mrc(tmp_path, dask_client):
    shape = (2, 4, 8, 8)
    data = np.random.randint(0, 200, shape, dtype=np.uint8)
    src = tmp_path / "vol_4d.mrc"
    with mrcfile.new(src, overwrite=True) as mrc:
        mrc.set_data(data)
    dest = tmp_path / "out.zarr"

    to_zarr(str(src), str(dest), dask_client)

    result = open_ts(zarr3_spec(str(dest), "s0"))[:].read().result()
    assert result.shape == shape
    assert np.array_equal(result, data)
