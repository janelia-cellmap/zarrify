"""N5 and zarr v2 sources take dimension_names from their own multiscales metadata."""
import numpy as np
import pytest
import zarr

from zarrify.to_zarr import to_zarr
from zarrify.utils.ts_utils import open_ts, zarr3_spec


SHAPE = (4, 32, 32)


CONFIG = dict(axes=['z', 'y', 'x'], scale=[1.0, 1.0, 1.0],
              translation=[0.0, 0.0, 0.0], units=['nanometer'] * 3,
              zarr_chunks=[4, 32, 32], workers=1)


def _dimension_names(dest, path='s0'):
    arr = zarr.open_array(zarr.storage.LocalStore(str(dest)), path=path, mode='r')
    return list(arr.metadata.dimension_names)


def test_zarr2_dimension_names_from_source(tmp_path, make_zarr2_source, dask_client):
    """Source multiscales axes win over the config axes."""
    data = np.random.randint(1, 255, SHAPE, dtype=np.uint8)
    src = make_zarr2_source(data, SHAPE, axes=('c', 'y', 'x'))
    dest = tmp_path / 'out.zarr'

    to_zarr(str(src), str(dest), dask_client, **CONFIG)

    assert _dimension_names(dest) == ['c', 'y', 'x']


@pytest.mark.parametrize("axes", [None, ('y', 'x')], ids=["no-multiscales", "rank-mismatch"])
def test_zarr2_without_matching_axes_fails_before_write(tmp_path, make_zarr2_source, dask_client, axes):
    data = np.random.randint(1, 255, SHAPE, dtype=np.uint8)
    src = make_zarr2_source(data, SHAPE, axes=axes)
    dest = tmp_path / 'out.zarr'

    with pytest.raises(ValueError, match="Could not find OME multiscales axes matching rank 3"):
        to_zarr(str(src), str(dest), dask_client, **CONFIG)
    assert not dest.exists()


@pytest.mark.parametrize("axes", [None, ('y', 'x')], ids=["no-axes", "rank-mismatch"])
def test_n5_without_matching_axes_fails_before_write(tmp_path, make_n5_source, dask_client, axes):
    data = np.random.randint(1, 255, SHAPE, dtype=np.uint8)
    src = make_n5_source(data, SHAPE, axes=axes)
    dest = tmp_path / 'out.zarr'

    with pytest.raises(ValueError, match="Could not find N5 multiscales axes matching rank 3"):
        to_zarr(str(src), str(dest), dask_client, **CONFIG)
    assert not dest.exists()


def test_zarr2_source_can_be_group_inside_store(tmp_path, make_zarr2_source, dask_client):
    """src may point at a multiscales group below the .zarr root."""
    data = np.random.randint(1, 255, SHAPE, dtype=np.uint8)
    src = make_zarr2_source(data, SHAPE, group='em/fibsem')
    dest = tmp_path / 'out.zarr'

    to_zarr(str(src), str(dest), dask_client, **CONFIG)

    out = open_ts(zarr3_spec(str(dest), 's0'))[:].read().result()
    assert np.array_equal(out, data)
    assert _dimension_names(dest) == ['z', 'y', 'x']
