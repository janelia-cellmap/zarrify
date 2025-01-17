import zarr 
import mrcfile
import os
from typing import Tuple 
from dask.array.core import slices_from_chunks, normalize_chunks
from dask.distributed import Client, wait
from toolz import partition_all
import time
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
        """Construct all the necessary attributes for the proper conversion of tiff to OME-NGFF Zarr.

        Args:
            input_filepath (str): path to source tiff file.
        """
        super().__init__(src_path, axes, translation, scale, units)

        self.memmap = mrcfile.mmap(self.src_path, mode='r')
        self.ndim = self.memmap.data.ndim
        self.shape =self.memmap.shape
        self.dtype = self.memmap.data.dtype
     
    def save_chunk(self,
                    z_arr: zarr.core.Array,
                    chunk_slice : Tuple[slice, ...]):
        """Copies data from a particular part of the input mrc array into a specific chunk of the output zarr array.

        Args:
            z_arr (zarr.core.Array): output zarr array object
            chunk_slice (Tuple[slice, ...]): slice of the mrc array to copy. 
        """
        mrc_file = mrcfile.mmap(self.src_path, mode='r')

        if not (mrc_file.data[chunk_slice] == 0).all():
            z_arr[chunk_slice] = mrc_file.data[chunk_slice]
    

    def write_to_zarr(self,
                    z_arr : zarr.Array,
                    client: Client,
                    ):
        """Use mrcfile memmap to access small parts of the mrc file and write them into zarr chunks.

        Args:
            dest_path (str): path to the zarr group where the output dataset is stored.
            client (Client): instance of a dask client
        """
            
        out_slices = slices_from_chunks(normalize_chunks(z_arr.chunks, shape=z_arr.shape))
        out_slices_partitioned = tuple(partition_all(100000, out_slices))

        for idx, part in enumerate(out_slices_partitioned):
            
            print(f'{idx + 1} / {len(out_slices_partitioned)}')
            start = time.time()
            fut = client.map(lambda v: self.save_chunk(z_arr, v), part)
            print(f'Submitted {len(part)} tasks to the scheduler in {time.time()- start}s')
            # wait for all the futures to complete
            result = wait(fut)
            print(f'Completed {len(part)} tasks in {time.time() - start}s')
                    
    

    