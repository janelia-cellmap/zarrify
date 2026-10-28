"""Smoke test on a dask cluster with a real worker process.

The other tests share a threaded cluster (processes=False) for speed, so tasks
are not pickled. This test sends tasks to a separate process, as LSF runs do,
so a non-picklable task or spec fails here.
"""
import numpy as np
from dask.distributed import Client, LocalCluster

from zarrify.to_zarr import to_zarr
from zarrify.utils.ts_utils import open_ts, zarr3_spec


def test_zarr2_to_zarr3_on_process_worker(tmp_path, make_zarr2_source):
    data = np.random.randint(1, 255, (4, 32, 32), dtype=np.uint8)
    src = make_zarr2_source(data, (4, 16, 16))
    dest = tmp_path / "out.zarr"

    with Client(LocalCluster(n_workers=1, processes=True, dashboard_address=":0")) as client:
        to_zarr(str(src), str(dest), client, workers=1,
                axes=["z", "y", "x"], scale=[1.0, 1.0, 1.0],
                translation=[0.0, 0.0, 0.0], units=["nanometer"] * 3,
                zarr_chunks=[4, 16, 16], shard_shape=[4, 32, 32])

    assert np.array_equal(open_ts(zarr3_spec(str(dest), "s0"))[:].read().result(), data)
