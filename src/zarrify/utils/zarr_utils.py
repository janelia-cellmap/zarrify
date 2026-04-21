import numpy as np

from zarrify.utils.ts_utils import zarr3_spec, open_ts, zstd_codec


def create_output_array(
    store_path: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
    chunk_shape: list[int],
    shard_shape: list[int] | None = None,
    codec: dict = zstd_codec(level=6),
    array_path: str = "s0",
) -> object:
    """Create and open a zarr3 output array via TensorStore.

    Parameters
    ----------
    store_path:
        Absolute path to the zarr store root directory on the local filesystem.
    shape:
        Array dimensions.
    dtype:
        NumPy dtype for the array.
    chunk_shape:
        Inner chunk shape. When *shard_shape* is also given this is the inner
        chunk stored inside each shard; otherwise it is the chunk grid directly.
    shard_shape:
        Outer shard shape. When provided, enables sharding.
    codec:
        Compression codec dict. Defaults to zstd_codec(level=6).
    array_path:
        Path of the array within the store. Defaults to "s0".

    Returns
    -------
    ts.TensorStore
        The created and opened TensorStore array.
    """
    spec = zarr3_spec(
        store_path=store_path,
        array_path=array_path,
        shape=shape,
        dtype=dtype,
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        codec=codec,
        create=True,
    )
    return open_ts(spec)
