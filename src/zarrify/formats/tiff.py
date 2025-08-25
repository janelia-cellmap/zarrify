from tifffile import imread
import numpy as np
import zarr
import os
from dask.distributed import Client, wait
import time
import copy
from zarrify.utils.volume import Volume
from abc import ABCMeta
from numcodecs import Zstd
import logging
from dask.array.core import slices_from_chunks, normalize_chunks



class Tiff(Volume):

    def __init__(
        self,
        src_path: str,
        axes: list[str],
        scale: list[float],
        translation: list[float],
        units: list[str],
    ):
        """Construct all the necessary attributes for the proper conversion of tiff to OME-NGFF Zarr.

        Args:
            input_filepath (str): path to source tiff file.
        """
        super().__init__(src_path, axes, scale, translation, units)

        self.zarr_store = imread(os.path.join(src_path), aszarr=True)
        self.zarr_arr = zarr.open(self.zarr_store)

        self.shape = self.zarr_arr.shape
        self.dtype = self.zarr_arr.dtype
        self.ndim = self.zarr_arr.ndim
        
        # Scale metadata parameters to match data dimensionality
        self.metadata["axes"] = list(self.metadata["axes"])[-self.ndim:]
        self.metadata["scale"] = self.metadata["scale"][-self.ndim:]
        self.metadata["translation"] = self.metadata["translation"][-self.ndim:]
        self.metadata["units"] = self.metadata["units"][-self.ndim:]

    def write_to_zarr(self,
        zarr_array: zarr.Array,
        client: Client,
        ):
        
        # Find slab axis based on metadata axes - use z axis for slabbing
        slab_axis = self.metadata["axes"].index('z')
            
        z_arr = zarr_array
        
        #slicing
        #(c, z, y, x) or (z, y, x) - combine (c, z) from zarr_chunks and (y, x) from tiff chunking
        slice_chunks = list(z_arr.chunks[:slab_axis+1]).copy()
        slice_chunks.extend(self.zarr_arr.chunks[slab_axis+1:])
        print(slice_chunks)
        
        normalized_chunks = normalize_chunks(slice_chunks, shape=self.zarr_arr.shape)
        slice_tuples = slices_from_chunks(normalized_chunks)

        
        
        src_path = copy.copy(self.src_path)

        start = time.time()
        fut = client.map(
            lambda v: write_volume_slab_to_zarr(v, z_arr, src_path), slice_tuples
        )
        logging.info(
            f"Submitted {len(slice_tuples)} tasks to the scheduler in {time.time()- start}s"
        )

        # wait for all the futures to complete
        result = wait(fut)
        logging.info(f"Completed {len(slice_tuples)} tasks in {time.time() - start}s")

        return 0


def write_volume_slab_to_zarr(slice: slice, zarray: zarr.Array, src_path: str):

    tiff_store = imread(src_path, aszarr=True)
    src_tiff_arr = zarr.open(tiff_store, mode='r')
    zarray[slice] = src_tiff_arr[slice]
