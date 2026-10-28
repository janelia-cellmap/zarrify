# Zarrify

Convert TIFF, TIFF stacks, MRC, N5, zarr v2 and zarr v3 files to OME-Zarr v3. A zarr v3 source is rewritten with the new chunks, shards and codec; its existing pyramid levels are kept.

## Install

Requires [Pixi](https://pixi.sh/latest/installation/).

```bash
git clone <repo>
cd zarrify
pixi run dev-install
```

## Run

```bash
pixi run zarrify --config config.yaml
```

Copy a config template from `config_examples/` and edit for your data.

## Config

```yaml
src: "/path/to/input"         # tiff, tiff stack dir, .mrc, .n5, or zarr v2/v3 .zarr group
dest: "/path/to/output.zarr"

cluster: lsf                  # local | lsf
workers: 100
log_dir: "/path/to/lsf/logs"
extra_directives: ["-P myproject"]

zarr_chunks: [32, 32, 32]     # inner chunk shape
shard_shape: [1024, 1024, 1024]  # outer shard — null to disable
codec: zstd                   # zstd | gzip | blosc
codec_level: 3                # null = per-codec default (zstd:3, gzip:6, blosc:5)
blosc_cname: zstd             # blosc only: blosclz | lz4 | lz4hc | snappy | zlib | zstd
shuffle: shuffle               # blosc only: noshuffle | shuffle | bitshuffle (byte-shuffle before compressing)

axes: [z, y, x]
scale: [8.0, 8.0, 8.0]
translation: [0.0, 0.0, 0.0]
units: [nanometer, nanometer, nanometer]

optimize_reads: false         # chunk-aligned TIFF reads (single TIFF only)
invert: false                 # bitwise-invert voxel data (N5, zarr v2 and zarr v3 sources only)

multiscale: false             # build a multiscale pyramid
ms_workers: 100
data_origin: raw              # raw | labels
antialiasing: false
normalize_voxel_size: false
custom_scale_factors: null    # e.g. [[1,1,1],[2,2,2],[4,4,4]]
```

CLI flags override config values:

```bash
pixi run zarrify --config config.yaml --workers 200 --scale 1.0 1.0 0.058 0.058
```

## LSF example

```bash
bsub -n 4 -P myproject -o out.log -e err.log \
  "pixi run zarrify --config config.yaml"
```

## Python API

```python
from zarrify.to_zarr import to_zarr
from zarrify.utils.dask_utils import initialize_dask_client

client = initialize_dask_client("local")
to_zarr("input.mrc", "output.zarr", client)
```
