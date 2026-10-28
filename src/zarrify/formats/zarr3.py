import json
import os

import numpy as np

from zarrify.formats.zarr2 import Zarr2Group
from zarrify.utils.ts_utils import zarr3_spec


class Zarr3Group(Zarr2Group):
    """zarr v3 group source, rewritten to zarr v3 with new chunks, shards and codec.

    Reuses the Zarr2Group conversion; only the metadata reads and the source
    spec differ. Group attributes may carry OME-NGFF 0.5 ('ome' wrapper, copied
    as is) or 0.4-style top-level multiscales (upgraded to 0.5).
    """

    format_name = 'zarr v3'

    def _read_node(self, rel_path: str) -> dict:
        """Return the zarr.json of the node at *rel_path*, or {} if absent or unreadable."""
        try:
            with open(os.path.join(self.src_path, rel_path, 'zarr.json')) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _walk_nodes(self, node_type: str) -> list[str]:
        """Return store-relative paths ('.' for the root) of all nodes of *node_type*.

        The walk does not descend into arrays, so chunk directories are never listed.
        """
        paths = []
        for dirpath, dirnames, filenames in os.walk(self.src_path):
            if 'zarr.json' not in filenames:
                continue
            rel = os.path.relpath(dirpath, self.src_path)
            found = self._read_node(rel).get('node_type')
            if found == 'array':
                dirnames[:] = []
            if found == node_type:
                paths.append(rel)
        return paths

    def _read_attrs(self, rel_path: str) -> dict:
        return self._read_node(rel_path).get('attributes') or {}

    def _read_array_meta(self, rel_path: str) -> tuple[tuple[int, ...], np.dtype]:
        meta = self._read_node(rel_path)
        return tuple(meta['shape']), np.dtype(meta['data_type'])

    def _src_spec(self, rel_path: str) -> dict:
        return zarr3_spec(self.src_path, rel_path)

    def _find_arrays(self) -> list[str]:
        return self._walk_nodes('array')

    def _find_groups(self) -> list[str]:
        return self._walk_nodes('group')
