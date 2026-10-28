"""zarr v3 sources are rewritten with new chunks and shards; levels, values and metadata are kept."""
import json

import numpy as np
import pytest
import zarr

from zarrify.to_zarr import init_dataset, to_zarr
from zarrify.utils.ts_utils import open_ts, zarr3_spec

CONFIG = dict(axes=['z', 'y', 'x'], scale=[1.0, 1.0, 1.0],
              translation=[0.0, 0.0, 0.0], units=['nanometer'] * 3,
              zarr_chunks=[8, 8, 8], shard_shape=[16, 16, 16], workers=1)


def _levels():
    s0 = np.random.randint(1, 2**16, (32, 32, 32), dtype=np.uint16)
    return {'s0': s0, 's1': s0[::2, ::2, ::2].copy()}


@pytest.mark.parametrize("ome", ["0.5", "0.4"])
def test_zarr3_to_zarr3_reshards_every_level(tmp_path, make_zarr3_source, dask_client, ome):
    levels = _levels()
    src = make_zarr3_source(levels, chunks=(32, 32, 32), ome=ome)
    dest = tmp_path / 'out.zarr'

    to_zarr(str(src), str(dest), dask_client, **CONFIG)

    for path, data in levels.items():
        out = open_ts(zarr3_spec(str(dest), path))[:].read().result()
        assert np.array_equal(out, data)

        meta = json.loads((dest / path / 'zarr.json').read_text())
        assert meta['chunk_grid']['configuration']['chunk_shape'] == [16, 16, 16]
        sharding = meta['codecs'][0]
        assert sharding['name'] == 'sharding_indexed'
        assert sharding['configuration']['chunk_shape'] == [8, 8, 8]
        assert meta['dimension_names'] == ['z', 'y', 'x']

    root = zarr.open_group(zarr.storage.LocalStore(str(dest)), mode='r')
    assert 'multiscales' not in root.attrs
    assert root.attrs['ome']['version'] == '0.5'
    ms = root.attrs['ome']['multiscales'][0]
    assert [d['path'] for d in ms['datasets']] == ['s0', 's1']
    assert ms['datasets'][1]['coordinateTransformations'][0]['scale'] == [2.0, 2.0, 2.0]


def test_zarr3_source_can_be_group_inside_store(tmp_path, make_zarr3_source, dask_client):
    levels = _levels()
    src = make_zarr3_source(levels, chunks=(16, 16, 16), group='labels/mito')
    dest = tmp_path / 'out.zarr'

    to_zarr(str(src), str(dest), dask_client, **CONFIG)

    out = open_ts(zarr3_spec(str(dest), 's0'))[:].read().result()
    assert np.array_equal(out, levels['s0'])


def test_zarr3_array_as_source_is_rejected(make_zarr3_source):
    src = make_zarr3_source(_levels(), chunks=(16, 16, 16))

    with pytest.raises(ValueError, match="zarr source must be a group"):
        init_dataset(str(src / 's0'), **{k: CONFIG[k] for k in ('axes', 'scale', 'translation', 'units')})
