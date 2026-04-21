"""TensorStore spec builders and helpers for zarrify.

TensorStore specs are plain dicts, making them safely picklable for Dask workers.
Each worker constructs its own TensorStore handle from the spec rather than sharing
a single handle across process boundaries.
"""

from __future__ import annotations


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
        Internal compressor. One of ``"lz4"``, ``"lz4hc"``, ``"blosclz"``,
        ``"zstd"``, ``"snappy"``, ``"zlib"``.
    clevel:
        Compression level (0–9).
    shuffle:
        Byte-shuffle filter. One of ``"noshuffle"``, ``"shuffle"``,
        ``"bitshuffle"``.
    blocksize:
        Block size in bytes. ``0`` lets Blosc choose automatically.
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
