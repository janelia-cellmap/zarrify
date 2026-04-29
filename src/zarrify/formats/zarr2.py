import json
import logging
import os
import time
from typing import Tuple

import numpy as np
import zarr
from dask.array.core import normalize_chunks, slices_from_chunks
from dask.distributed import Client, wait
from toolz import partition_all

from zarrify.utils.dask_utils import check_shardslab_fits_in_ram, raise_on_task_errors
from zarrify.utils.ts_utils import align_shard_to_chunks, zarr2_spec, zarr3_spec, open_ts, zstd_codec
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

    def _update_multiscales_expand_dims(self, dest: str) -> None:
        """Prepend a channel axis to any OME multiscales metadata in the output store.

        Called after group/array attrs have been copied from the zarr2 source,
        when expand_dims=True has added a leading size-1 dimension to every array.
        """
        dst_store = zarr.storage.LocalStore(dest)
        dst_root = zarr.open_group(store=dst_store, mode='a')
        self._patch_group_multiscales(dst_root)

    def _patch_group_multiscales(self, group: zarr.Group) -> None:
        for key in group.keys():
            node = group[key]
            if isinstance(node, zarr.Group):
                self._patch_group_multiscales(node)

        if 'multiscales' not in group.attrs:
            return
        ms = list(group.attrs['multiscales'])
        for m in ms:
            if 'axes' in m:
                m['axes'] = [{'name': 'c', 'type': 'channel'}] + list(m['axes'])
            for ct in m.get('coordinateTransformations', []):
                if ct['type'] == 'scale':
                    ct['scale'] = [1.0] + list(ct['scale'])
                elif ct['type'] == 'translation':
                    ct['translation'] = [0.0] + list(ct['translation'])
            for ds in m.get('datasets', []):
                for ct in ds.get('coordinateTransformations', []):
                    if ct['type'] == 'scale':
                        ct['scale'] = [1.0] + list(ct['scale'])
                    elif ct['type'] == 'translation':
                        ct['translation'] = [0.0] + list(ct['translation'])
        group.attrs['multiscales'] = ms

    def write_to_zarr(
        self,
        dest: str,
        client: Client,
        chunk_shape: list[int],
        shard_shape: list[int] | None = None,
        codec: dict = zstd_codec(level=6),
        expand_dims: bool = False,
    ) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        array_paths = self._find_arrays()
        logger.info(f"Found {len(array_paths)} zarr v2 arrays: {array_paths}")

        dst_store = zarr.storage.LocalStore(dest)
        dst_root = zarr.open_group(store=dst_store, mode='a')
        self._copy_group_attrs(dst_root)

        for rel_path in array_paths:
            logger.info(f"Processing array: {rel_path}")
            zarray_path = os.path.join(self.src_path, rel_path, '.zarray')
            with open(zarray_path) as f:
                zarray_meta = json.load(f)

            shape = tuple(zarray_meta['shape'])
            dtype = np.dtype(zarray_meta['dtype'])
            out_shape = (1, *shape) if expand_dims else shape

            arr_chunk_shape_base = [min(c, s) for c, s in zip(list(chunk_shape)[-len(shape):], shape)]
            arr_chunk_shape = ([1] + arr_chunk_shape_base) if expand_dims else arr_chunk_shape_base
            arr_shard_shape = (
                align_shard_to_chunks(
                    [min(s, dim) for s, dim in zip(list(shard_shape)[-len(out_shape):], out_shape)],
                    arr_chunk_shape,
                )
                if shard_shape is not None else None
            )
            if arr_shard_shape is not None:
                check_shardslab_fits_in_ram(arr_shard_shape, dtype, arr_chunk_shape, client)

            src_spec = zarr2_spec(self.src_path, rel_path)
            dst_spec = zarr3_spec(
                store_path=dest,
                array_path=rel_path,
                shape=out_shape,
                dtype=dtype,
                chunk_shape=arr_chunk_shape,
                shard_shape=arr_shard_shape,
                codec=codec,
                create=True,
            )

            dest_arr = open_ts(dst_spec)
            dest_chunks = dest_arr.chunk_layout.write_chunk.shape

            logger.info(f"{rel_path}: shape={out_shape}, dtype={dtype}, chunk={arr_chunk_shape}, shard={arr_shard_shape}")
            out_slices = slices_from_chunks(normalize_chunks(dest_chunks, shape=out_shape))
            out_slices_partitioned = tuple(partition_all(100000, out_slices))
            logger.info(f"{rel_path}: {len(out_slices)} total slices, {len(out_slices_partitioned)} batch(es)")

            for idx, part in enumerate(out_slices_partitioned):
                logger.info(f"{rel_path}: {idx + 1} / {len(out_slices_partitioned)}")
                start = time.time()
                fut = client.map(
                    lambda v: save_chunk(src_spec, dst_spec, v, expand_dims), part
                )
                logger.info(f"Submitted {len(part)} tasks in {time.time() - start:.2f}s")
                wait(fut)
                raise_on_task_errors(fut)
                logger.info(f"Completed {len(part)} tasks in {time.time() - start:.2f}s")

        self._copy_array_attrs(dest, array_paths)
        if expand_dims:
            self._update_multiscales_expand_dims(dest)


def save_chunk(
    src_spec: dict,
    dst_spec: dict,
    chunk_slice: Tuple[slice, ...],
    expand_dims: bool = False,
) -> None:
    src = open_ts(src_spec)
    src_slice = chunk_slice[1:] if expand_dims else chunk_slice
    data = src[src_slice].read().result()
    if (data == 0).all():
        return
    dest = open_ts(dst_spec)
    dest[chunk_slice].write(data[np.newaxis] if expand_dims else data).result()
