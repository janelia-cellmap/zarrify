"""Shared fixtures: a module-scoped dask client and builders for small N5 / zarr v2 sources."""
import json

import numpy as np
import pytest
import tensorstore as ts

from zarrify.utils.dask_utils import initialize_dask_client


@pytest.fixture(scope="module")
def dask_client():
    client = initialize_dask_client("local")
    yield client
    client.close()


@pytest.fixture
def make_zarr2_source(tmp_path):
    """Factory: write *data* as array 's0' of a zarr v2 group.

    *axes* go into an OME-NGFF 0.4 multiscales entry in the group's .zattrs;
    pass None to write no multiscales. *group* places the multiscales group
    below the store root (e.g. "em/fibsem"). Returns the group path.
    """
    def _make(data, chunks, axes=("z", "y", "x"), name="src.zarr", group=""):
        root = tmp_path / name
        group_path = root / group if group else root
        ts.open({
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': str(root)},
            'path': f"{group}/s0" if group else 's0',
            'metadata': {
                'dtype': np.dtype(data.dtype).str,
                'shape': list(data.shape),
                'chunks': list(chunks),
            },
            'create': True,
            'open': True,
        }).result()[:].write(data).result()

        (root / '.zgroup').write_text(json.dumps({'zarr_format': 2}))
        (group_path / '.zgroup').write_text(json.dumps({'zarr_format': 2}))
        if axes is not None:
            multiscales = [{
                'version': '0.4',
                'axes': [
                    {'name': ax, 'type': 'channel'} if ax == 'c'
                    else {'name': ax, 'type': 'space', 'unit': 'nanometer'}
                    for ax in axes
                ],
                'datasets': [{'path': 's0', 'coordinateTransformations': [
                    {'type': 'scale', 'scale': [1.0] * len(axes)},
                ]}],
            }]
            (group_path / '.zattrs').write_text(json.dumps({'multiscales': multiscales}))
        return group_path
    return _make


@pytest.fixture
def make_n5_source(tmp_path):
    """Factory: write *data* as array 's0' of an N5 group.

    *axes* go into the root attributes.json; pass None to leave them out.
    Returns the N5 root path.
    """
    def _make(data, chunks, axes=("z", "y", "x"), name="src.n5"):
        n5_path = tmp_path / name
        ts.open({
            'driver': 'n5',
            'kvstore': {'driver': 'file', 'path': str(n5_path)},
            'path': 's0',
            'metadata': {
                'dataType': np.dtype(data.dtype).name,
                'dimensions': list(data.shape),
                'blockSize': list(chunks),
                'compression': {'type': 'raw'},
            },
            'create': True,
            'open': True,
        }).result()[:].write(data).result()

        ndim = data.ndim
        root_attrs = {'n5': '2.0.0', 'units': ['nm'] * ndim, 'scales': [[1.0] * ndim]}
        if axes is not None:
            root_attrs['axes'] = list(axes)
        (n5_path / 'attributes.json').write_text(json.dumps(root_attrs))

        s0_attrs_path = n5_path / 's0' / 'attributes.json'
        s0_attrs = json.loads(s0_attrs_path.read_text())
        s0_attrs['transform'] = {
            'scale': [1.0] * ndim,
            'translate': [0.0] * ndim,
            'units': ['nm'] * ndim,
            'axes': list(axes) if axes is not None else ['z', 'y', 'x'][-ndim:],
        }
        s0_attrs_path.write_text(json.dumps(s0_attrs))
        return n5_path
    return _make
