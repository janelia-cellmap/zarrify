"""N5 and zarr v2 sources fail before any write when their multiscales axes are missing or do not match the rank.

The success case (dimension_names from the source) is checked in test_to_zarr.py.
"""
import numpy as np
import pytest

from zarrify.to_zarr import to_zarr

SHAPE = (4, 32, 32)
CONFIG = dict(axes=['z', 'y', 'x'], scale=[1.0, 1.0, 1.0],
              translation=[0.0, 0.0, 0.0], units=['nanometer'] * 3,
              zarr_chunks=[4, 32, 32], workers=1)


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
