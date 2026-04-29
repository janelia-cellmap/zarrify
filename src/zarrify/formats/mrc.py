import copy
import logging
import time
from typing import Tuple

import mrcfile
import numpy as np
from dask.array.core import normalize_chunks, slices_from_chunks
from dask.distributed import Client, wait
from toolz import partition_all

from zarrify.utils.ts_utils import open_ts
from zarrify.utils.volume import Volume


class Mrc3D(Volume):

    def __init__(
        self,
        src_path: str,
        axes: list[str],
        scale: list[float],
        translation: list[float],
        units: list[str],
    ):
        """Initialize an MRC volume reader.

        Parameters
        ----------
        src_path:
            Path to the source MRC file.
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

        self.memmap = mrcfile.mmap(self.src_path, mode="r")
        self.ndim = self.memmap.data.ndim
        self.shape = np.squeeze(self.memmap.data.shape)
        self.dtype = self.memmap.data.dtype

    def write_to_zarr(self, dst_spec: dict, client: Client, expand_dims: bool = False) -> None:
        """Read the MRC file in chunks and write each chunk to a zarr3 array via TensorStore.

        Chunks that are entirely zero are skipped to avoid unnecessary writes.

        Parameters
        ----------
        dst_spec:
            TensorStore zarr3 spec dict for the destination array, as returned
            by :func:`~zarrify.utils.ts_utils.zarr3_spec`.
        client:
            Dask distributed client used to parallelise chunk writes.
        expand_dims:
            When True, the destination array has a leading size-1 channel
            dimension; the leading slice is stripped when reading the source.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        src_path = copy.copy(self.src_path)
        dest_arr = open_ts(dst_spec)

        # Use write_chunk (shard shape when sharding, chunk shape otherwise) so
        # each Dask task writes one complete on-disk object.
        dest_chunks = dest_arr.chunk_layout.write_chunk.shape

        out_slices = slices_from_chunks(
            normalize_chunks(dest_chunks, shape=tuple(dest_arr.shape))
        )
        out_slices_partitioned = tuple(partition_all(100000, out_slices))

        for idx, part in enumerate(out_slices_partitioned):
            logging.info(f"{idx + 1} / {len(out_slices_partitioned)}")
            start = time.time()
            fut = client.map(lambda v: save_chunk(src_path, dst_spec, v, expand_dims), part)
            logging.info(
                f"Submitted {len(part)} tasks to the scheduler in {time.time() - start:.2f}s"
            )
            wait(fut)
            logging.info(f"Completed {len(part)} tasks in {time.time() - start:.2f}s")


def save_chunk(src_path: str, dst_spec: dict, chunk_slice: Tuple[slice, ...],
               expand_dims: bool = False) -> None:
    """Copy one chunk from an MRC file into a zarr3 TensorStore array.

    Chunks that are entirely zero are skipped to avoid unnecessary I/O.

    Parameters
    ----------
    src_path:
        Path to the source MRC file.
    dst_spec:
        TensorStore zarr3 spec dict for the destination array.
    chunk_slice:
        The slice tuple identifying the chunk region in the destination array.
    expand_dims:
        When True, strip the leading slice when reading the source (which has
        no channel dimension) and prepend np.newaxis before writing.
    """
    mrc_file = mrcfile.mmap(src_path, mode="r")
    src_slice = chunk_slice[1:] if expand_dims else chunk_slice
    data = mrc_file.data[src_slice]
    if not (data == 0).all():
        dest_arr = open_ts(dst_spec)
        dest_arr[chunk_slice].write(data[np.newaxis] if expand_dims else data).result()
