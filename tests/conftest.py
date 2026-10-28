"""Shared fixtures: a session-scoped 1-worker dask client and builders for small N5 / zarr v2 / zarr v3 sources."""
import json

import numpy as np
import pytest
import tensorstore as ts

from dask.distributed import Client, LocalCluster


@pytest.fixture(scope="session")
def dask_client():
    client = Client(LocalCluster(n_workers=1, processes=False))
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


@pytest.fixture
def make_zarr3_source(tmp_path):
    """Factory: write *levels* ({'s0': array, 's1': array, ...}) as unsharded arrays of a zarr v3 group.

    *ome* selects the group attributes: "0.5" writes the 'ome' wrapper,
    "0.4" writes top-level multiscales. *group* places the multiscales group
    below the store root. Returns the group path.
    """
    def _make(levels, chunks, axes=("z", "y", "x"), ome="0.5", name="src.zarr", group=""):
        root = tmp_path / name
        group_path = root / group if group else root
        for path, data in levels.items():
            ts.open({
                'driver': 'zarr3',
                'kvstore': {'driver': 'file', 'path': str(root)},
                'path': f"{group}/{path}" if group else path,
                'metadata': {
                    'shape': list(data.shape),
                    'chunk_grid': {'name': 'regular', 'configuration': {'chunk_shape': list(chunks)}},
                    'data_type': np.dtype(data.dtype).name,
                    'codecs': [{'name': 'bytes', 'configuration': {'endian': 'little'}},
                               {'name': 'zstd', 'configuration': {'level': 3}}],
                    'dimension_names': list(axes),
                },
                'create': True,
                'open': True,
            }).result()[:].write(data).result()

        multiscale = {
            'axes': [{'name': ax, 'type': 'space', 'unit': 'nanometer'} for ax in axes],
            'datasets': [
                {'path': path, 'coordinateTransformations': [
                    {'type': 'scale', 'scale': [float(2 ** i)] * len(axes)},
                ]}
                for i, path in enumerate(levels)
            ],
        }
        if ome == "0.5":
            attrs = {'ome': {'version': '0.5', 'multiscales': [multiscale]}}
        else:
            attrs = {'multiscales': [{'version': '0.4', **multiscale}]}

        group_meta = {'zarr_format': 3, 'node_type': 'group', 'attributes': {}}
        (root / 'zarr.json').write_text(json.dumps(group_meta))
        (group_path / 'zarr.json').write_text(json.dumps({**group_meta, 'attributes': attrs}))
        return group_path
    return _make
