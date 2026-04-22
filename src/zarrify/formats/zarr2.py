import json
import logging
import os

import zarr

from zarrify.utils.volume import Volume

logger = logging.getLogger(__name__)


class Zarr2Group(Volume):

    def __init__(
        self,
        src_path: str,
        axes: list[str],
        scale: list[float],
        translation: list[float],
        units: list[str],
    ):
        super().__init__(src_path, axes, scale, translation, units)

    def _find_arrays(self) -> list[str]:
        """Return store-relative paths of all zarr v2 arrays (dirs containing .zarray)."""
        array_paths = []
        for dirpath, _, filenames in os.walk(self.src_path):
            if '.zarray' in filenames:
                array_paths.append(os.path.relpath(dirpath, self.src_path))
        return array_paths

    def _copy_group_attrs(self, dst_root: zarr.Group) -> None:
        """Copy zarr v2 group-level .zattrs to the output zarr3 groups."""
        for dirpath, _, filenames in os.walk(self.src_path):
            if '.zarray' in filenames:
                continue
            if '.zgroup' not in filenames and '.zattrs' not in filenames:
                continue
            attrs = {}
            zattrs_path = os.path.join(dirpath, '.zattrs')
            if os.path.exists(zattrs_path):
                with open(zattrs_path) as f:
                    try:
                        attrs = json.load(f)
                    except json.JSONDecodeError:
                        pass
            rel = os.path.relpath(dirpath, self.src_path)
            target = dst_root if rel == '.' else dst_root.require_group(rel)
            for k, v in attrs.items():
                target.attrs[k] = v

    def _copy_array_attrs(self, dest: str, array_paths: list[str]) -> None:
        """Copy zarr v2 array .zattrs to the output zarr3 arrays."""
        dst_store = zarr.storage.LocalStore(dest)
        for rel_path in array_paths:
            zattrs_path = os.path.join(self.src_path, rel_path, '.zattrs')
            if not os.path.exists(zattrs_path):
                continue
            with open(zattrs_path) as f:
                try:
                    attrs = json.load(f)
                except json.JSONDecodeError:
                    continue
            dst_arr = zarr.open_array(store=dst_store, path=rel_path, mode='a')
            for k, v in attrs.items():
                dst_arr.attrs[k] = v
