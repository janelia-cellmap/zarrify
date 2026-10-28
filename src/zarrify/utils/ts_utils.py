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


def _build_codecs(
    codec: dict,
    chunk_shape: list[int] | None,
    checksum: bool = True,
) -> list[dict]:
    """Return the zarr3 codecs list, wrapping with sharding_indexed when *chunk_shape* is given.

    When *checksum* is True a CRC32C codec is appended to the data codec chain
    so that corruption in the stored (compressed) bytes is detected on read.
    The shard index always includes CRC32C regardless of this flag.
    """
    data_codecs: list[dict] = [_BYTES_CODEC, codec]
    if checksum:
        data_codecs.append(_CRC32C_CODEC)

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
# Shard alignment helper
# ---------------------------------------------------------------------------

def align_shard_to_chunks(shard_shape: list[int], chunk_shape: list[int]) -> list[int]:
    """Round each shard dim down to the nearest multiple of the corresponding chunk dim.

    zarr3 requires shard dims to be exact multiples of inner chunk dims. When a
    shard dim is clamped to the array extent it may no longer satisfy this, so
    this re-aligns it. Each dim is floored to at least one chunk.
    """
    return [max(c, (s // c) * c) for s, c in zip(shard_shape, chunk_shape)]


# ---------------------------------------------------------------------------
# Spec builders
# ---------------------------------------------------------------------------

# Forces TensorStore's internal thread pools to be fully serial, so the only
# concurrency in the pipeline is the Dask worker's own thread count. Without
# this, "shared" pools default to the host's total CPU count (which on a
# shared HPC node can be far larger than the cores actually granted to the
# job), letting a single chunk task fan out into extra concurrent buffers on
# top of Dask's own per-worker concurrency.
_DEFAULT_CONTEXT: dict = {
    "data_copy_concurrency": {"limit": 1},
    "file_io_concurrency": {"limit": 1},
    "cache_pool": {"total_bytes_limit": 0},
}


def zarr3_spec(
    store_path: str,
    array_path: str,
    shape: tuple[int, ...] | None = None,
    dtype: np.dtype | None = None,
    chunk_shape: list[int] | None = None,
    shard_shape: list[int] | None = None,
    codec: dict | None = None,
    checksum: bool = True,
    dimension_names: list[str] | None = None,
    *,
    create: bool = False,
    fill_value: int | float | bool = 0,
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
    checksum:
        When True (default), appends a CRC32C codec to the data codec chain so
        that corruption in stored bytes is detected on read. The shard index
        always includes CRC32C regardless of this flag.
    create:
        When True, embed creation metadata so ts.open(spec) creates the array
        on disk if absent.
    fill_value:
        Value that unwritten chunks read as. Only used when *create* is True.

    Returns
    -------
    dict
        A picklable TensorStore spec suitable for passing to ts.open().
    """
    spec: dict = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(store_path)},
        "path": array_path,
        "open": True,
        "context": dict(_DEFAULT_CONTEXT),
    }

    if create:
        if shape is None or dtype is None or chunk_shape is None:
            raise ValueError("shape, dtype, and chunk_shape are required when create=True")
        if dimension_names is None:
            raise ValueError(
                "dimension_names is required when create=True (OME-NGFF 0.5 spec)"
            )
        if len(dimension_names) != len(shape):
            raise ValueError(
                f"dimension_names length ({len(dimension_names)}) must match "
                f"shape length ({len(shape)})"
            )

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
            "codecs": _build_codecs(resolved_codec, chunk_shape if shard_shape is not None else None, checksum),
            "fill_value": fill_value,
            "dimension_names": list(dimension_names),
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
        "kvstore": {"driver": "file", "path": str(store_path)},
        "path": array_path,
        "open": True,
        "context": dict(_DEFAULT_CONTEXT),
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
        "kvstore": {"driver": "file", "path": str(store_path)},
        "path": array_path,
        "open": True,
        "context": dict(_DEFAULT_CONTEXT),
    }


_CODEC_DEFAULTS = {"zstd": 3, "gzip": 6, "blosc": 5}


def build_codec(
    name: str,
    level: int | None = None,
    cname: str | None = None,
    shuffle: str | None = None,
) -> dict:
    """Return a TensorStore zarr3 codec dict from a codec name and compression level.

    When *level* is None the codec's own default is used:
    zstd → 3, gzip → 6, blosc → 5.

    Byte-shuffling (a preprocessing step that improves compression of
    structured numeric data, e.g. fibsem images) is only available through
    Blosc in the zarr v3 codec spec — the plain "zstd"/"gzip" codecs have no
    shuffle field. To get "zstd with shuffle", pick codec="blosc" and
    cname="zstd".

    Parameters
    ----------
    name:
        Codec name. One of "zstd", "gzip", "blosc".
    level:
        Compression level. When None, a sensible per-codec default is applied.
    cname:
        Blosc-only: internal compressor ("lz4", "lz4hc", "blosclz", "zstd",
        "snappy", "zlib"). Ignored for "zstd"/"gzip". Defaults to
        blosc_codec's own default ("lz4") when None.
    shuffle:
        Blosc-only: byte-shuffle filter ("noshuffle", "shuffle", "bitshuffle").
        Ignored for "zstd"/"gzip". Defaults to blosc_codec's own default
        ("shuffle") when None.
    """
    if name not in _CODEC_DEFAULTS:
        raise ValueError(f"Unsupported codec: {name!r}. Choose from {list(_CODEC_DEFAULTS)}.")
    resolved = level if level is not None else _CODEC_DEFAULTS[name]
    if name == "zstd":
        return zstd_codec(resolved)
    elif name == "gzip":
        return gzip_codec(resolved)
    else:
        blosc_kwargs = {"clevel": resolved}
        if cname is not None:
            blosc_kwargs["cname"] = cname
        if shuffle is not None:
            blosc_kwargs["shuffle"] = shuffle
        return blosc_codec(**blosc_kwargs)


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
