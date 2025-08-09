from tifffile import imread
import numpy as np
import zarr
import os
from dask.distributed import Client, wait
import time
import dask.array as da
import copy
from zarrify.utils.volume import Volume
from abc import ABCMeta
from numcodecs import Zstd
import logging


class Tiff3D(Volume):

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
        self.metadata["axes"] = self.metadata["axes"][-self.ndim:]
        self.metadata["scale"] = self.metadata["scale"][-self.ndim:]
        self.metadata["translation"] = self.metadata["translation"][-self.ndim:]
        self.metadata["units"] = self.metadata["units"][-self.ndim:]

    def write_to_zarr(self,
        dest: str,
        client: Client,
        zarr_chunks : list[int],
        comp : ABCMeta = Zstd(level=6),
        ):
        
        # reshape chunk shape to align with arr shape
        if len(zarr_chunks) != self.shape:
           zarr_chunks = self.reshape_to_arr_shape(zarr_chunks, self.shape)
             
        z_arr = self.get_output_array(dest, zarr_chunks, comp)
        chunks_list = np.arange(0, z_arr.shape[0], z_arr.chunks[0])

        src_path = copy.copy(self.src_path)

        start = time.time()
        fut = client.map(
            lambda v: write_volume_slab_to_zarr(v, z_arr, src_path), chunks_list
        )
        logging.info(
            f"Submitted {len(chunks_list)} tasks to the scheduler in {time.time()- start}s"
        )

        # wait for all the futures to complete
        result = wait(fut)
        logging.info(f"Completed {len(chunks_list)} tasks in {time.time() - start}s")

        return 0


def write_volume_slab_to_zarr(chunk_num: int, zarray: zarr.Array, src_path: str):

    # check if the slab is at the array boundary or not
    if chunk_num + zarray.chunks[0] > zarray.shape[0]:
        slab_thickness = zarray.shape[0] - chunk_num
    else:
        slab_thickness = zarray.chunks[0]

    tiff_store = imread(src_path, aszarr=True)
    src_zarr_arr = zarr.open(tiff_store, mode = 'r')
    tiff_slab = src_zarr_arr[chunk_num:chunk_num + slab_thickness, :, :]

    # write a tiff stack slab into zarr array
    zarray[chunk_num : chunk_num + slab_thickness, :, :] = tiff_slab
