import zarr
from abc import ABCMeta


def create_output_array(dest: str, shape: tuple, dtype, chunks: list[int], comp: ABCMeta) -> zarr.Array:
    """Create a zarr array at the specified destination with given parameters.
    
    Args:
        dest (str): Output zarr group location
        shape (tuple): Shape of the array
        dtype: Data type of the array
        chunks (list[int]): Chunk sizes for the array
        comp (ABCMeta): Compressor to use
        
    Returns:
        zarr.Array: The created zarr array
    """
    z_store = zarr.NestedDirectoryStore(dest)
    z_root = zarr.open(store=z_store, mode="a")
    
    return z_root.require_dataset(
        name="s0",
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        compressor=comp,
    )