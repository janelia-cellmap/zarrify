"""One conversion per source format; each test checks values and OME-NGFF 0.5 metadata of its output."""
import mrcfile
import numpy as np
import ome_zarr_models.v05
import pytest
import tifffile
import zarr

from zarrify.to_zarr import to_zarr
from zarrify.utils.ts_utils import open_ts, zarr3_spec


def _read(dest, path='s0'):
    return open_ts(zarr3_spec(str(dest), path))[:].read().result()


def _root_attrs(dest):
    return zarr.open_group(zarr.storage.LocalStore(str(dest)), mode='r').attrs


def _dimension_names(dest, path='s0'):
    return list(zarr.open_array(zarr.storage.LocalStore(str(dest)), path=path, mode='r').metadata.dimension_names)


def test_tiff_to_zarr3(tmp_path, dask_client):
    """Values, OME 0.5 multiscales, transforms and dimension_names; output validates with ome-zarr-models."""
    data = np.random.randint(0, 255, (10, 32, 32), dtype=np.uint8)
    src = tmp_path / "img.tif"
    tifffile.imwrite(src, data)
    dest = tmp_path / "img.zarr"
    axes = ["z", "y", "x"]
    scale = [4.0, 8.0, 8.0]
    translation = [0.0, 0.0, 0.0]

    to_zarr(str(src), str(dest), dask_client, workers=1,
            axes=axes, scale=scale, translation=translation, units=["nanometer"] * 3,
            zarr_chunks=[10, 32, 32])

    assert np.array_equal(_read(dest), data)

    attrs = _root_attrs(dest)
    assert attrs["ome"]["version"] == "0.5"
    ms = attrs["ome"]["multiscales"][0]
    assert [a["name"] for a in ms["axes"]] == axes
    assert [d["path"] for d in ms["datasets"]] == ["s0"]
    cts = ms["datasets"][0]["coordinateTransformations"]
    assert any(ct["type"] == "scale" and ct["scale"] == scale for ct in cts)
    assert any(ct["type"] == "translation" and ct["translation"] == translation for ct in cts)
    assert _dimension_names(dest) == axes

    image = ome_zarr_models.v05.Image.from_zarr(zarr.open(zarr.storage.LocalStore(str(dest)), mode="r"))
    assert [a.name for a in image.ome_attributes.multiscales[0].axes] == axes


@pytest.mark.parametrize("ext,shape", [("tif", (40, 50)), ("mrc", (10, 30, 50))], ids=["tiff-2d", "mrc-3d"])
def test_tiff_2d_and_mrc_to_zarr3(tmp_path, dask_client, ext, shape):
    data = np.random.randint(0, 255, shape, dtype=np.uint8)
    src = tmp_path / f"img.{ext}"
    if ext == "mrc":
        with mrcfile.new(src) as mrc:
            mrc.set_data(data)
    else:
        tifffile.imwrite(src, data)
    dest = tmp_path / "img.zarr"
    ndim = len(shape)

    to_zarr(str(src), str(dest), dask_client, workers=1,
            axes=["z", "y", "x"][-ndim:], scale=[1.0] * ndim,
            translation=[0.0] * ndim, units=["nanometer"] * ndim,
            zarr_chunks=[64] * ndim)

    assert np.array_equal(_read(dest), data)


def test_4d_tiff_with_default_config(tmp_path, dask_client):
    """4D TIFF (c, z, y, x) with the to_zarr defaults."""
    data = np.random.randint(0, 255, (3, 10, 64, 64), dtype=np.uint8)
    src = tmp_path / "rgb.tif"
    tifffile.imwrite(src, data)
    dest = tmp_path / "rgb.zarr"

    to_zarr(str(src), str(dest), dask_client, workers=1)

    assert np.array_equal(_read(dest), data)


def test_n5_to_zarr3(tmp_path, make_n5_source, dask_client):
    """Values, OME 0.5 multiscales; dimension_names come from the N5 'axes', not the config axes."""
    data = np.random.randint(1, 255, (4, 32, 32), dtype=np.uint8)
    src = make_n5_source(data, data.shape, axes=("c", "y", "x"))
    dest = tmp_path / "out.zarr"

    to_zarr(str(src), str(dest), dask_client, workers=1,
            axes=["z", "y", "x"], scale=[1.0, 1.0, 1.0],
            translation=[0.0, 0.0, 0.0], units=["nanometer"] * 3,
            zarr_chunks=[4, 32, 32])

    assert np.array_equal(_read(dest), data)

    attrs = _root_attrs(dest)
    assert attrs["ome"]["version"] == "0.5"
    ms = attrs["ome"]["multiscales"][0]
    assert [a["name"] for a in ms["axes"]] == ["c", "y", "x"]
    assert any(d["path"] == "s0" for d in ms["datasets"])
    assert _dimension_names(dest) == ["c", "y", "x"]


def test_zarr2_to_zarr3(tmp_path, make_zarr2_source, dask_client):
    """Source is a multiscales group inside the store; 0.4 metadata is upgraded to 0.5 and
    dimension_names come from the source multiscales, not the config axes."""
    data = np.random.randint(1, 255, (4, 32, 32), dtype=np.uint8)
    src = make_zarr2_source(data, data.shape, axes=("c", "y", "x"), group="em/fibsem")
    dest = tmp_path / "out.zarr"

    to_zarr(str(src), str(dest), dask_client, workers=1,
            axes=["z", "y", "x"], scale=[1.0, 1.0, 1.0],
            translation=[0.0, 0.0, 0.0], units=["nanometer"] * 3,
            zarr_chunks=[4, 32, 32])

    assert np.array_equal(_read(dest), data)

    attrs = _root_attrs(dest)
    assert "multiscales" not in attrs, "top-level 0.4 multiscales must not be kept"
    assert attrs["ome"]["version"] == "0.5"
    ms = attrs["ome"]["multiscales"][0]
    assert "version" not in ms, "0.5 puts version at the ome level, not in multiscale entries"
    assert [a["name"] for a in ms["axes"]] == ["c", "y", "x"]
    assert _dimension_names(dest) == ["c", "y", "x"]
