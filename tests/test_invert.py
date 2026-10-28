"""invert option for N5 and zarr v2 sources, and CLI flag defaults."""
import importlib

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from zarrify.to_zarr import to_zarr
from zarrify.utils.ts_utils import open_ts, zarr3_spec

to_zarr_module = importlib.import_module("zarrify.to_zarr")

SHAPE = (4, 32, 32)
CONFIG = dict(axes=['z', 'y', 'x'], scale=[1.0, 1.0, 1.0],
              translation=[0.0, 0.0, 0.0], units=['nanometer'] * 3,
              zarr_chunks=[4, 32, 32], workers=1)


@pytest.mark.parametrize("source", ["zarr2", "n5"])
@pytest.mark.parametrize("invert", [True, False])
def test_invert(tmp_path, make_zarr2_source, make_n5_source, dask_client, source, invert):
    data = np.random.randint(1, 255, SHAPE, dtype=np.uint8)
    make = make_zarr2_source if source == "zarr2" else make_n5_source
    src = make(data, SHAPE)
    dest = tmp_path / 'out.zarr'

    to_zarr(str(src), str(dest), dask_client, invert=invert, **CONFIG)

    out = open_ts(zarr3_spec(str(dest), 's0'))[:].read().result()
    assert np.array_equal(out, np.invert(data) if invert else data)


@pytest.mark.parametrize("source", ["zarr2", "n5"])
def test_invert_all_zero_chunk_reads_as_inverted_zero(tmp_path, make_zarr2_source, make_n5_source,
                                                      dask_client, source):
    """An all-zero source chunk is skipped, and reads back as np.invert(0) through fill_value."""
    data = np.random.randint(1, 255, SHAPE, dtype=np.uint8)
    data[:, :16, :16] = 0  # exactly one 4x16x16 chunk is all zero
    make = make_zarr2_source if source == "zarr2" else make_n5_source
    src = make(data, (4, 16, 16))
    dest = tmp_path / 'out.zarr'

    to_zarr(str(src), str(dest), dask_client, invert=True, **{**CONFIG, 'zarr_chunks': [4, 16, 16]})

    out = open_ts(zarr3_spec(str(dest), 's0'))[:].read().result()
    assert np.array_equal(out, np.invert(data))
    chunks_dir = dest / 's0' / 'c' / '0'
    assert (chunks_dir / '0' / '1').exists()
    assert not (chunks_dir / '0' / '0').exists()  # empty chunk was not written


@pytest.fixture
def run_cli(tmp_path, monkeypatch):
    """Run the zarrify CLI with a config file; return the kwargs it passes to to_zarr."""
    captured = {}
    monkeypatch.setattr(to_zarr_module, "initialize_dask_client", lambda *a, **k: None)
    monkeypatch.setattr(to_zarr_module, "to_zarr", lambda **kw: captured.update(kw))

    def _run(config, *flags):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({"src": str(tmp_path), "dest": str(tmp_path / "out.zarr"),
                                            "cluster": "local", **config}))
        result = CliRunner().invoke(to_zarr_module.cli, ["--config", str(cfg_path), *flags])
        assert result.exit_code == 0, result.output
        return captured
    return _run


def test_cli_keeps_config_values_when_flags_absent(run_cli):
    kwargs = run_cli({"invert": True, "optimize_reads": False})
    assert kwargs["invert"] is True
    assert kwargs["optimize_reads"] is False


def test_cli_flags_override_config(run_cli):
    kwargs = run_cli({"invert": False, "optimize_reads": False}, "--invert", "--optimize_reads")
    assert kwargs["invert"] is True
    assert kwargs["optimize_reads"] is True
