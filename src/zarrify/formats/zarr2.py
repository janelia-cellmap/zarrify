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

from zarrify.utils.dask_utils import check_shardslab_fits_in_ram
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

    def write_to_zarr(
        self,
        dest: str,
        client: Client,
        chunk_shape: list[int],
        shard_shape: list[int] | None = None,
        codec: dict = zstd_codec(level=6),
    ) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        array_paths = self._find_arrays()

        dst_store = zarr.storage.LocalStore(dest)
        dst_root = zarr.open_group(store=dst_store, mode='a')
        self._copy_group_attrs(dst_root)

        for rel_path in array_paths:
            zarray_path = os.path.join(self.src_path, rel_path, '.zarray')
            with open(zarray_path) as f:
                zarray_meta = json.load(f)

            shape = tuple(zarray_meta['shape'])
            dtype = np.dtype(zarray_meta['dtype'])

            arr_chunk_shape = [min(c, s) for c, s in zip(list(chunk_shape)[-len(shape):], shape)]
            arr_shard_shape = (
                align_shard_to_chunks(
                    [min(s, dim) for s, dim in zip(list(shard_shape)[-len(shape):], shape)],
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
                shape=shape,
                dtype=dtype,
                chunk_shape=arr_chunk_shape,
                shard_shape=arr_shard_shape,
                codec=codec,
                create=True,
            )

            dest_arr = open_ts(dst_spec)
            dest_chunks = dest_arr.chunk_layout.write_chunk.shape

            out_slices = slices_from_chunks(normalize_chunks(dest_chunks, shape=shape))
            out_slices_partitioned = tuple(partition_all(100000, out_slices))

            for idx, part in enumerate(out_slices_partitioned):
                logger.info(f"{rel_path}: {idx + 1} / {len(out_slices_partitioned)}")
                start = time.time()
                fut = client.map(
                    lambda v: save_chunk(src_spec, dst_spec, v), part
                )
                logger.info(f"Submitted {len(part)} tasks in {time.time() - start:.2f}s")
                wait(fut)
                logger.info(f"Completed {len(part)} tasks in {time.time() - start:.2f}s")

        self._copy_array_attrs(dest, array_paths)


def save_chunk(
    src_spec: dict,
    dst_spec: dict,
    chunk_slice: Tuple[slice, ...],
) -> None:
    src = open_ts(src_spec)
    data = src[chunk_slice].read().result()
    if (data == 0).all():
        return
    dest = open_ts(dst_spec)
    dest[chunk_slice].write(data).result()
