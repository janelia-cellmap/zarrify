import ctypes
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


def _multiscales(attrs: dict) -> list:
    """Return the OME-NGFF multiscales list from *attrs* (0.5 'ome' wrapper or 0.4 top level)."""
    return (attrs.get('ome') or {}).get('multiscales') or attrs.get('multiscales') or []


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

    def _find_multiscales_axes(self, rel_path: str) -> list[str] | None:
        """Walk up from *rel_path* and return axes from the nearest multiscales group.

        Returns the names of the axes in the first ancestor group attributes
        that contain an OME-NGFF `multiscales` entry. Returns None if no such
        ancestor is found.
        """
        current = os.path.dirname(rel_path)
        while True:
            ms_list = _multiscales(self._read_attrs(current or '.'))
            if ms_list:
                axes_meta = ms_list[0].get('axes') or []
                if axes_meta:
                    return [a['name'] if isinstance(a, dict) else a for a in axes_meta]
            if not current:
                return None
            current = os.path.dirname(current)

    def _read_attrs(self, rel_path: str) -> dict:
        """Return the .zattrs of the node at *rel_path*, or {} if absent or unreadable."""
        zattrs_path = os.path.join(self.src_path, rel_path, '.zattrs')
        try:
            with open(zattrs_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _read_array_meta(self, rel_path: str) -> tuple[tuple[int, ...], np.dtype]:
        """Return (shape, dtype) of the array at *rel_path* from its .zarray."""
        with open(os.path.join(self.src_path, rel_path, '.zarray')) as f:
            zarray_meta = json.load(f)
        return tuple(zarray_meta['shape']), np.dtype(zarray_meta['dtype'])

    def _find_arrays(self) -> list[str]:
        """Return store-relative paths of all zarr v2 arrays (dirs containing .zarray)."""
        array_paths = []
        for dirpath, _, filenames in os.walk(self.src_path):
            if '.zarray' in filenames:
                array_paths.append(os.path.relpath(dirpath, self.src_path))
        return array_paths

    def _find_groups(self) -> list[str]:
        """Return store-relative paths ('.' for the root) of dirs with .zgroup or .zattrs that are not arrays."""
        group_paths = []
        for dirpath, _, filenames in os.walk(self.src_path):
            if '.zarray' in filenames:
                continue
            if '.zgroup' not in filenames and '.zattrs' not in filenames:
                continue
            group_paths.append(os.path.relpath(dirpath, self.src_path))
        return group_paths

    def _copy_group_attrs(self, dst_root: zarr.Group) -> None:
        """Copy source group attributes to the output zarr3 groups.

        OME-NGFF 0.4 multiscales are upgraded to the 0.5 'ome' wrapper.
        """
        for rel in self._find_groups():
            attrs = self._read_attrs(rel)
            target = dst_root if rel == '.' else dst_root.require_group(rel)
            for k, v in attrs.items():
                if k == 'multiscales':
                    target.attrs['ome'] = {
                        'version': '0.5',
                        'multiscales': [
                            {mk: mv for mk, mv in m.items() if mk not in ('version', 'name')}
                            for m in v
                        ],
                    }
                else:
                    target.attrs[k] = v

    def _copy_array_attrs(self, dest: str, array_paths: list[str]) -> None:
        """Copy source array attributes to the output zarr3 arrays."""
        dst_store = zarr.storage.LocalStore(dest)
        for rel_path in array_paths:
            attrs = self._read_attrs(rel_path)
            if not attrs:
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
        invert: bool = False,
    ) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        array_paths = self._find_arrays()
        logger.info(f"Found {len(array_paths)} zarr v2 arrays: {array_paths}")

        # Resolve dimension_names for every array up front (from each array's
        # enclosing multiscales group) and fail fast if any array's rank does
        # not match its multiscales axes. Multiscale datasets must have
        # homogeneous rank — this catches mismatches before any writes start.
        array_dim_names: dict[str, list[str]] = {}
        for rel_path in array_paths:
            with open(os.path.join(self.src_path, rel_path, '.zarray')) as f:
                shape_len = len(json.load(f)['shape'])
            axes = self._find_multiscales_axes(rel_path)
            if axes is None or len(axes) != shape_len:
                raise ValueError(
                    f"Could not find OME multiscales axes matching rank {shape_len} "
                    f"for array '{rel_path}'. Source must carry multiscales "
                    f"metadata at an enclosing group."
                )
            array_dim_names[rel_path] = axes

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
            dim_names = array_dim_names[rel_path]

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
                dimension_names=dim_names,
                create=True,
                # skipped all-zero source chunks must read back as inverted zeros
                fill_value=np.invert(dtype.type(0)).item() if invert else 0,
            )

            dest_arr = open_ts(dst_spec)
            dest_chunks = dest_arr.chunk_layout.write_chunk.shape

            logger.info(f"{rel_path}: shape={shape}, dtype={dtype}, chunk={arr_chunk_shape}, shard={arr_shard_shape}")
            out_slices = slices_from_chunks(normalize_chunks(dest_chunks, shape=shape))
            out_slices_partitioned = tuple(partition_all(1000, out_slices))
            logger.info(f"{rel_path}: {len(out_slices)} total slices, {len(out_slices_partitioned)} batch(es)")

            for idx, part in enumerate(out_slices_partitioned):
                logger.info(f"{rel_path}: {idx + 1} / {len(out_slices_partitioned)}")
                start = time.time()
                fut = client.map(
                    lambda v: save_chunk(src_spec, dst_spec, v, invert=invert), part
                )
                logger.info(f"Submitted {len(part)} tasks in {time.time() - start:.2f}s")
                wait(fut)
                raise_on_task_errors(fut)
                logger.info(f"Completed {len(part)} tasks in {time.time() - start:.2f}s")

        self._copy_array_attrs(dest, array_paths)


try:
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError:  # not glibc (e.g. macOS)
    _LIBC = None


def _release_memory_to_os() -> None:
    """Force glibc to hand freed heap pages back to the OS.

    Freed chunk buffers return to the malloc arena, which can keep those pages
    mapped indefinitely, so worker RSS creeps up across many sequential chunk
    tasks even though nothing is actually leaked at the Python level. Dask's
    MALLOC_TRIM_THRESHOLD_ does not cover this for large (shard-sized)
    buffers. malloc_trim forces the pages back, keeping steady-state memory
    close to a single chunk's footprint.

    No gc.collect() here: save_chunk creates no reference cycles, so a full
    collection per task only adds cost (~5x slower on small chunks).
    """
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


def save_chunk(
    src_spec: dict,
    dst_spec: dict,
    chunk_slice: Tuple[slice, ...],
    invert: bool = False,
) -> None:
    src = open_ts(src_spec)
    data = src[chunk_slice].read().result()
    try:
        if not data.any():
            return
        if invert:
            data = np.invert(data)
        dest = open_ts(dst_spec)
        dest[chunk_slice].write(data).result()
    finally:
        del data
        _release_memory_to_os()
