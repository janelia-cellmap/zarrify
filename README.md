# Zarrify

Convert TIFF/TIFF Stacks, MRC, and N5 files to Zarr (v2) format with OME-NGFF 0.4 metadata.

## Prerequisites

You must [install Pixi](https://pixi.sh/latest/installation/) to run this tool.

## Install into local env

After you check out the repo, run Pixi to download dependencies and create the local environment:

```bash
pixi run dev-install
```

## Usage

### Step 1: Convert image to OME-Zarr format

To convert an OME-TIFF image with axes ZCYX:

```
pixi run zarrify --src=/path/to/input --dest=/path/to/output.zarr   --axes z c y x --units nanometer '' nanometer nanometer --zarr_chunks 64 3 128 128
```

### Step 2: Add multiscale pyramid

To add multiscale pyramids to the above OME-Zarr:

```
pixi run zarr-multiscale --config=/path/to/config.yaml
```

Where `config.yaml` contains the parameters, e.g.:

```yaml
src: "/path/to/output.zarr"
workers: 50
data_origin: "raw"
cluster: "lsf"
log_dir: "/path/to/logs/workers"
antialiasing: false
high_aspect_ratio: false
custom_scale_factors: 
  - [1, 1, 1, 1]
  - [1, 1, 2, 2]
  - [1, 1, 4, 4]
  - [1, 1, 8, 8]
  - [1, 2, 16, 16]
  - [1, 4, 32, 32]
  - [1, 8, 64, 64]
  - [1, 16, 128, 128]
  - [1, 32, 256, 256]
  - [1, 64, 512, 512]
  - [1, 128, 1024, 1024]
```

### IBM Platform LSF 

You can run the pipeline on an LSF cluster by specifying the `--cluster=lsf` parameter. For example:

```
export INPUT=/path/to/input
export OUTPUT=/path/to/output.zarr
export LOGDIR=`pwd`
bsub -n 4 -P myproject -o $LOGDIR/jobout.log -e $LOGDIR/joberr.log "/misc/sc/pixi run zarrify --src=$INPUT --dest=$OUTPUT --cluster $CLUSTER --log_dir $LOGDIR/workers --workers 500 --zarr_chunks 3 128 128 128 --extra_directives='-P myproject' --scale 1.0 1.0 0.116 0.116 --optimize_reads"
```


## Python API

Integrating conversion to zarr into the python code

```python
import zarrify
from zarrify.utils.dask_utils import initialize_dask_client

client = initialize_dask_client("local") # or "lsf"
zarrify.to_zarr("input.mrc", "output.zarr", client, workers=20)
```

## Supported formats

- TIFF stacks
- 3D TIFF files
- MRC files  
- N5 containers
