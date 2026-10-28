import json

import numpy as np
import pytest
from pydantic import ValidationError

from zarrify.to_zarr import to_zarr
from zarrify.utils.pydantic_models import ZarrifyConfig
from zarrify.utils.ts_utils import build_codec, open_ts, zarr3_spec


def test_build_codec_blosc_options():
    cfg = build_codec("blosc", 7, cname="zstd", shuffle="bitshuffle")["configuration"]
    assert (cfg["cname"], cfg["clevel"], cfg["shuffle"]) == ("zstd", 7, "bitshuffle")


def test_build_codec_blosc_defaults():
    cfg = build_codec("blosc")["configuration"]
    assert (cfg["cname"], cfg["clevel"], cfg["shuffle"]) == ("lz4", 5, "shuffle")


@pytest.mark.parametrize("codec", ["zstd", "gzip"])
def test_build_codec_ignores_blosc_options_for_other_codecs(codec):
    assert build_codec(codec, cname="zstd", shuffle="bitshuffle") == build_codec(codec)


@pytest.mark.parametrize("option", [{"blosc_cname": "zstd"}, {"shuffle": "bitshuffle"}])
def test_config_rejects_blosc_options_without_blosc(tmp_path, option):
    with pytest.raises(ValidationError, match="only apply when codec='blosc'"):
        ZarrifyConfig(src=str(tmp_path), dest=str(tmp_path / "out.zarr"), codec="zstd", **option)


def test_config_accepts_blosc_options(tmp_path):
    cfg = ZarrifyConfig(src=str(tmp_path), dest=str(tmp_path / "out.zarr"),
                        codec="blosc", blosc_cname="zstd", shuffle="bitshuffle")
    assert (cfg.blosc_cname, cfg.shuffle) == ("zstd", "bitshuffle")


def _find_codec(codecs, name):
    """Return the first codec called *name*, looking inside sharding codecs too."""
    for c in codecs:
        if c["name"] == name:
            return c
        inner = c.get("configuration", {}).get("codecs")
        if inner and (found := _find_codec(inner, name)):
            return found
    return None


def test_zarr2_to_zarr3_writes_blosc_options(tmp_path, make_zarr2_source, dask_client):
    data = np.random.randint(1, 255, (4, 32, 32), dtype=np.uint8)
    src = make_zarr2_source(data, (4, 32, 32))
    dest = tmp_path / "out.zarr"

    to_zarr(str(src), str(dest), dask_client,
            axes=["z", "y", "x"], scale=[1.0, 1.0, 1.0],
            translation=[0.0, 0.0, 0.0], units=["nanometer"] * 3,
            zarr_chunks=[4, 32, 32], codec="blosc", blosc_cname="zstd", shuffle="bitshuffle")

    meta = json.loads((dest / "s0" / "zarr.json").read_text())
    blosc = _find_codec(meta["codecs"], "blosc")
    assert blosc is not None
    assert blosc["configuration"]["cname"] == "zstd"
    assert blosc["configuration"]["shuffle"] == "bitshuffle"
    assert np.array_equal(open_ts(zarr3_spec(str(dest), "s0"))[:].read().result(), data)
