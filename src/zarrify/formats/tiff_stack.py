import json
import logging
import os
import time
from glob import glob

import numpy as np
from dask.distributed import Client, wait
from natsort import natsorted
from tifffile import imread

from zarrify.utils.ts_utils import open_ts
from zarrify.utils.volume import Volume


class TiffStack(Volume):

    def __init__(
        self,
        src_path: str,
        axes: list[str],
        scale: list[float],
        translation: list[float],
        units: list[str],
    ):
        """Initialize a TIFF stack reader from a directory of 2-D TIFF tiles.

        Parameters
        ----------
        src_path:
            Path to the directory containing the TIFF files.
        axes:
            Axis names (e.g. ["z", "y", "x"]).
        scale:
            Voxel size along each axis.
        translation:
            Spatial offset along each axis.
        units:
            Physical unit for each axis (e.g. ["nm", "nm", "nm"]).
        """
        super().__init__(src_path, axes, scale, translation, units)

        self.stack_list = natsorted(glob(os.path.join(src_path, "*.tif*")))
        probe_store = imread(os.path.join(src_path, self.stack_list[0]), aszarr=True)
        tile_meta = json.loads(probe_store[".zarray"])
        probe_store.close()

        self.dtype = np.dtype(tile_meta["dtype"])
        self.shape = tuple(np.squeeze([len(self.stack_list)] + tile_meta["shape"]))
        self.ndim = len(self.shape)

    def write_to_zarr(self, dst_spec: dict, client: Client) -> None:
        """Assemble tiles into slabs and write each slab to a zarr3 array via TensorStore.

        Parameters
        ----------
        dst_spec:
            TensorStore zarr3 spec dict for the destination array, as returned
            by :func:`~zarrify.utils.ts_utils.zarr3_spec`.
        client:
            Dask distributed client used to parallelise slab writes.
        """
        # TODO: With large shard shapes (e.g. 1024^3) the slab thickness along
        # z becomes 1024 voxels, which may exhaust worker memory when assembling
        # tiles. Implement a strobing write pattern: first pass writes tiles
        # 0-255, second pass 256-511, etc., so each task assembles a sub-slab
        # and multiple passes run concurrently without holding the full shard
        # in memory at once.

        dest_arr = open_ts(dst_spec)
        tiff_chunkdim = dest_arr.chunk_layout.write_chunk.shape[0]
        chunks_list = np.arange(0, dest_arr.shape[0], tiff_chunkdim)

        start = time.time()
        fut = client.map(
            lambda v: write_tile_slab(v, dst_spec, self.stack_list), chunks_list
        )
        logging.info(
            f"Submitted {len(chunks_list)} tasks to the scheduler in {time.time() - start:.4f}s"
        )
        wait(fut)
        logging.info(f"Completed {len(chunks_list)} tasks in {time.time() - start:.2f}s")


def write_tile_slab(chunk_num: int, dst_spec: dict, src_volume: list) -> None:
    """Assemble one slab from individual TIFF tiles and write it to a zarr3 TensorStore array.

    Parameters
    ----------
    chunk_num:
        Index of the first tile (along z) in this slab.
    dst_spec:
        TensorStore zarr3 spec dict for the destination array.
    src_volume:
        Ordered list of TIFF file paths comprising the full stack.
    """
    dest_arr = open_ts(dst_spec)
    total_z = dest_arr.shape[0]
    tiff_chunkdim = dest_arr.chunk_layout.write_chunk.shape[0]

    actual_thickness = min(tiff_chunkdim, total_z - chunk_num)
    slab_shape = [actual_thickness] + list(dest_arr.shape[1:])
    np_slab = np.empty(slab_shape, dtype=np.dtype(dest_arr.dtype.numpy_dtype))

    for slab_index in range(chunk_num, chunk_num + actual_thickness):
        try:
            np_slab[slab_index - chunk_num] = imread(src_volume[slab_index])
        except Exception:
            logging.info(f"Tiff tile with index {slab_index} is not present in tiff stack.")

    dest_arr[chunk_num: chunk_num + actual_thickness].write(np_slab).result()
