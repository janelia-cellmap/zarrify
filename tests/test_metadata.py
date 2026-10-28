"""Tests that zarrify writes correct OME-NGFF metadata.

zarr3 output → OME-NGFF 0.5 (under the "ome" key)
"""
import json

import numpy as np
import pytest
import tensorstore as ts
import zarr

from zarrify.to_zarr import to_zarr
from zarrify.utils.dask_utils import initialize_dask_client


@pytest.fixture(scope="module")
def dask_client():
    client = initialize_dask_client("local")
    yield client
    client.close()


def test_zarr2_to_zarr3_upgrades_to_05(tmp_path, dask_client):
    """Converting a zarr2 source with 0.4 metadata produces zarr3 with 0.5."""
    shape = (10, 32, 32)
    data = np.random.randint(0, 255, shape, dtype=np.uint8)

    src = tmp_path / "src.zarr"
    ts.open({
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(src)},
        "path": "s0",
        "metadata": {
            "dtype": np.dtype(data.dtype).str,
            "shape": list(shape),
            "chunks": list(shape),
        },
        "create": True,
        "open": True,
    }).result()[:].write(data).result()

    multiscales = [{
        "version": "0.4",
        "axes": [{"name": ax, "type": "space", "unit": "nanometer"} for ax in ["z", "y", "x"]],
        "datasets": [{"path": "s0", "coordinateTransformations": [
            {"type": "scale", "scale": [1.0, 1.0, 1.0]}
        ]}],
    }]
    (src / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (src / ".zattrs").write_text(json.dumps({"multiscales": multiscales}))

    dest = tmp_path / "dst.zarr"
    to_zarr(str(src), str(dest), dask_client,
            axes=["z", "y", "x"], scale=[1.0, 1.0, 1.0],
            translation=[0.0, 0.0, 0.0], units=["nanometer", "nanometer", "nanometer"],
            zarr_chunks=[10, 32, 32])

    root = zarr.open_group(zarr.storage.LocalStore(str(dest)), mode="r")
    assert "ome" in root.attrs, "zarr3 output should have 'ome' 0.5 key"
    assert "multiscales" not in root.attrs, "zarr3 output should not keep top-level 0.4 multiscales key"
    assert root.attrs["ome"]["version"] == "0.5"
    ms = root.attrs["ome"]["multiscales"][0]
    assert "version" not in ms, "0.5 puts version at the ome level, not in multiscale entries"
    assert [a["name"] for a in ms["axes"]] == ["z", "y", "x"]
