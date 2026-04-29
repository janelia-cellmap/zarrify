import copy
import json
import logging
import os
import time

import numpy as np
from dask.array.core import normalize_chunks, slices_from_chunks
from dask.distributed import Client, wait
from tifffile import imread

from zarrify.utils.dask_utils import raise_on_task_errors
from zarrify.utils.ts_utils import open_ts
from zarrify.utils.volume import Volume

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class Tiff(Volume):

    def __init__(
        self,
        src_path: str,
        axes: list[str],
        scale: list[float],
        translation: list[float],
        units: list[str],
        optimize_reads: bool = False,
    ):
        """Initialize an N-D TIFF volume reader.

        Parameters
        ----------
        src_path:
            Path to the source TIFF file.
        axes:
            Axis names (e.g. ["z", "y", "x"]).
        scale:
            Voxel size along each axis.
        translation:
            Spatial offset along each axis.
        units:
            Physical unit for each axis (e.g. ["nm", "nm", "nm"]).
        optimize_reads:
            When True, align read slabs with TIFF internal chunk boundaries to
            reduce redundant decompression during ingestion.
        """
        super().__init__(src_path, axes, scale, translation, units)

        # Read array metadata directly from tifffile's zarr store without
        # involving zarr.open (zarr v3 no longer accepts ZarrTiffStore).
        _tiff_store = imread(os.path.join(src_path), aszarr=True)
        _meta = json.loads(_tiff_store['.zarray'])
        _tiff_store.close()

        self.shape = tuple(_meta['shape'])
        self.dtype = np.dtype(_meta['dtype'])
        self.ndim = len(self.shape)
        self._tiff_chunks = tuple(_meta['chunks'])
        self.optimize_reads = optimize_reads

        # Trim metadata to match actual data dimensionality.
        self.metadata["axes"] = list(self.metadata["axes"])[-self.ndim:]
        self.metadata["scale"] = self.metadata["scale"][-self.ndim:]
        self.metadata["translation"] = self.metadata["translation"][-self.ndim:]
        self.metadata["units"] = self.metadata["units"][-self.ndim:]

    def write_to_zarr(self, dst_spec: dict, client: Client, expand_dims: bool = False) -> None:
        """Read the TIFF file in slabs and write each slab to a zarr3 array via TensorStore.

        Parameters
        ----------
        dst_spec:
            TensorStore zarr3 spec dict for the destination array, as returned
            by :func:`~zarrify.utils.ts_utils.zarr3_spec`.
        client:
            Dask distributed client used to parallelise slab writes.
        expand_dims:
            When True, the destination array has a leading size-1 channel
            dimension; each slab is written with an extra np.newaxis prepended.
        """
        # TODO: With large shard shapes (e.g. 1024^3) the slab thickness along
        # the slab axis becomes 1024 voxels, potentially exhausting worker memory.
        # Implement a strobing write pattern: first pass writes voxels 0-255,
        # second pass 256-511, etc., so each task handles a sub-slab and workers
        # can process multiple passes concurrently without holding the full shard
        # in memory at once.

        # axes in metadata are already trimmed to source ndim; use them to find z.
        src_axes = self.metadata["axes"][1:] if expand_dims else self.metadata["axes"]
        slab_axis = src_axes.index("z") if "z" in src_axes else 0

        dest_arr = open_ts(dst_spec)
        # dest_chunks includes the leading 1 when expand_dims; strip it for
        # slice computation which is done against the source shape.
        dest_chunks = list(dest_arr.chunk_layout.write_chunk.shape)
        src_chunks = dest_chunks[1:] if expand_dims else dest_chunks

        slice_chunks = src_chunks
        if self.optimize_reads:
            logger.info("Optimizing read chunking...")
            logger.info(f"Output zarr3 write-chunk shape: {dest_chunks}")
            logger.info(f"Input TIFF chunk shape: {self._tiff_chunks}")
            slice_chunks = src_chunks[: slab_axis + 1].copy()

            for dest_chunkdim, tiff_chunkdim, tiff_dim in zip(
                src_chunks[slab_axis + 1:],
                self._tiff_chunks[slab_axis + 1:],
                self.shape[slab_axis + 1:],
            ):
                if tiff_chunkdim < dest_chunkdim:
                    slice_chunks.append(dest_chunkdim)
                elif tiff_chunkdim / tiff_dim < 0.5:
                    slice_chunks.append(int(tiff_chunkdim / dest_chunkdim) * dest_chunkdim)
                else:
                    slice_chunks.append(tiff_dim)

            logger.info(f"Adjusted slice chunks: {slice_chunks}")

        slab_size_bytes = int(np.prod(slice_chunks)) * np.dtype(self.dtype).itemsize
        logger.info(f"Slab size: {slab_size_bytes / 1e9:.3f} GB")

        normalized_chunks = normalize_chunks(slice_chunks, shape=self.shape)
        slice_tuples = slices_from_chunks(normalized_chunks)
        src_path = copy.copy(self.src_path)

        start = time.time()
        fut = client.map(
            lambda v: write_volume_slab(v, dst_spec, src_path, expand_dims), slice_tuples
        )
        logger.info(
            f"Submitted {len(slice_tuples)} tasks to the scheduler in "
            f"{time.time() - start:.4f}s"
        )
        wait(fut)
        raise_on_task_errors(fut)
        logger.info(f"Completed {len(slice_tuples)} tasks in {time.time() - start:.2f}s")


def write_volume_slab(slice_tuple: tuple, dst_spec: dict, src_path: str,
                      expand_dims: bool = False) -> None:
    """Copy one slab from a TIFF file into a zarr3 TensorStore array.

    The TIFF is read directly via tifffile into NumPy (no TensorStore TIFF
    driver exists), then written to the destination via TensorStore.

    Parameters
    ----------
    slice_tuple:
        The slice tuple identifying the slab region in the source array.
    dst_spec:
        TensorStore zarr3 spec dict for the destination array.
    src_path:
        Path to the source TIFF file.
    expand_dims:
        When True, prepend a size-1 channel dimension to the data before
        writing and offset the destination slice accordingly.
    """
    data = np.asarray(imread(src_path)[slice_tuple])
    dest_arr = open_ts(dst_spec)
    if expand_dims:
        dest_arr[(slice(0, 1), *slice_tuple)].write(data[np.newaxis]).result()
    else:
        dest_arr[slice_tuple].write(data).result()
