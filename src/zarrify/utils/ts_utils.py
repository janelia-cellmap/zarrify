"""TensorStore spec builders and helpers for zarrify.

TensorStore specs are plain dicts, making them safely picklable for Dask workers.
Each worker constructs its own TensorStore handle from the spec rather than sharing
a single handle across process boundaries.
"""

from __future__ import annotations

import numpy as np
import tensorstore as ts


# ---------------------------------------------------------------------------
# Codec constructors — return TensorStore zarr3 codec dicts
# ---------------------------------------------------------------------------

def zstd_codec(level: int = 6) -> dict:
    """Return a TensorStore zarr3 Zstandard codec spec.

    Parameters
    ----------
    level:
        Compression level (1–22). Higher values compress more but are slower.
    """
    return {"name": "zstd", "configuration": {"level": level}}


def blosc_codec(
    cname: str = "lz4",
    clevel: int = 5,
    shuffle: str = "shuffle",
    blocksize: int = 0,
) -> dict:
    """Return a TensorStore zarr3 Blosc codec spec.

    Parameters
    ----------
    cname:
        Internal compressor. One of "lz4", "lz4hc", "blosclz",
        "zstd", "snappy", "zlib".
    clevel:
        Compression level (0–9).
    shuffle:
        Byte-shuffle filter. One of "noshuffle", "shuffle", "bitshuffle".
    blocksize:
        Block size in bytes. 0 lets Blosc choose automatically.
    """
    return {
        "name": "blosc",
        "configuration": {
            "cname": cname,
            "clevel": clevel,
            "shuffle": shuffle,
            "blocksize": blocksize,
        },
    }


def gzip_codec(level: int = 6) -> dict:
    """Return a TensorStore zarr3 Gzip codec spec.

    Parameters
    ----------
    level:
        Compression level (0–9).
    """
    return {"name": "gzip", "configuration": {"level": level}}


# ---------------------------------------------------------------------------
# Internal codec helpers
# ---------------------------------------------------------------------------

_BYTES_CODEC = {"name": "bytes", "configuration": {"endian": "little"}}
_CRC32C_CODEC = {"name": "crc32c"}


def _build_codecs(codec: dict, chunk_shape: list[int] | None) -> list[dict]:
    """Return the zarr3 codecs list, wrapping with sharding_indexed when *chunk_shape* is given."""
    data_codecs = [_BYTES_CODEC, codec]

    if chunk_shape is None:
        return data_codecs

    return [
        {
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": chunk_shape,
                "codecs": data_codecs,
                "index_codecs": [_BYTES_CODEC, _CRC32C_CODEC],
            },
        }
    ]


# ---------------------------------------------------------------------------
# Spec builders
# ---------------------------------------------------------------------------

def zarr3_spec(
    store_path: str,
    array_path: str,
    shape: tuple[int, ...] | None = None,
    dtype: np.dtype | None = None,
    chunk_shape: list[int] | None = None,
    shard_shape: list[int] | None = None,
    codec: dict | None = None,
    *,
    create: bool = False,
) -> dict:
    """Build a TensorStore zarr3 driver spec.

    When *create* is True, *shape*, *dtype*, and *chunk_shape* are required.
    The spec always sets open=True so an existing array is opened rather than
    raising an error.

    Sharding is enabled by passing *shard_shape*. In that case *shard_shape*
    defines the outer chunk grid — each shard is a single object on disk —
    while *chunk_shape* defines the inner data chunks stored inside each shard.
    Without *shard_shape*, *chunk_shape* is used as the chunk grid directly.

    Parameters
    ----------
    store_path:
        Absolute path to the zarr store root directory on the local filesystem.
    array_path:
        Path of the array relative to *store_path* (e.g. "s0").
    shape:
        Array dimensions. Required when *create* is True.
    dtype:
        NumPy dtype for the array. Required when *create* is True.
    chunk_shape:
        Chunk shape. Required when *create* is True. When *shard_shape* is
        also given, this is the inner chunk stored inside each shard; otherwise
        it is the chunk grid directly.
    shard_shape:
        Outer shard shape. When provided, enables sharding: each shard covers
        this region on disk and is subdivided into *chunk_shape* inner chunks.
    codec:
        Compression codec dict as returned by :func:`zstd_codec`,
        :func:`blosc_codec`, or :func:`gzip_codec`. Defaults to
        zstd_codec(level=6).
    create:
        When True, embed creation metadata so ts.open(spec) creates the array
        on disk if absent.

    Returns
    -------
    dict
        A picklable TensorStore spec suitable for passing to ts.open().
    """
    spec: dict = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": store_path},
        "path": array_path,
        "open": True,
    }

    if create:
        if shape is None or dtype is None or chunk_shape is None:
            raise ValueError("shape, dtype, and chunk_shape are required when create=True")

        outer_shape = shard_shape if shard_shape is not None else chunk_shape
        resolved_codec = codec if codec is not None else zstd_codec()

        spec["create"] = True
        spec["metadata"] = {
            "shape": list(shape),
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": outer_shape},
            },
            "data_type": np.dtype(dtype).name,
            "codecs": _build_codecs(resolved_codec, chunk_shape if shard_shape is not None else None),
            "fill_value": 0,
        }

    return spec


def n5_spec(store_path: str, array_path: str) -> dict:
    """Build a TensorStore N5 driver spec for reading an N5 array.

    Parameters
    ----------
    store_path:
        Absolute path to the N5 store root directory on the local filesystem.
    array_path:
        Dataset path relative to *store_path* (e.g. "volumes/raw/s0").

    Returns
    -------
    dict
        A picklable TensorStore spec suitable for passing to ts.open().
    """
    return {
        "driver": "n5",
        "kvstore": {"driver": "file", "path": store_path},
        "path": array_path,
        "open": True,
    }


def zarr2_spec(store_path: str, array_path: str) -> dict:
    """Build a TensorStore zarr (v2) driver spec for reading an existing array.

    Parameters
    ----------
    store_path:
        Absolute path to the zarr v2 store root directory on the local filesystem.
    array_path:
        Path of the array relative to *store_path* (e.g. "s0").

    Returns
    -------
    dict
        A picklable TensorStore spec suitable for passing to ts.open().
    """
    return {
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": store_path},
        "path": array_path,
        "open": True,
    }


def open_ts(spec: dict) -> ts.TensorStore:
    """Open a TensorStore array synchronously from *spec*.

    Parameters
    ----------
    spec:
        A TensorStore spec dict as returned by :func:`zarr3_spec`,
        :func:`n5_spec`, or :func:`zarr2_spec`.

    Returns
    -------
    ts.TensorStore
        The opened (or created) TensorStore array.
    """
    return ts.open(spec).result()
