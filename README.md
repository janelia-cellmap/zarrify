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

### Using Configuration Files (Recommended)

Create a YAML configuration file with your conversion parameters:

```yaml
# config.yaml
src: "/path/to/input/file"
dest: "/path/to/output.zarr"
workers: 100
cluster: "local"  # or "lsf"

# Zarr array configuration
zarr_chunks: [3, 64, 128, 128]
axes: ["c", "z", "y", "x"]
translation: [0.0, 0.0, 0.0, 0.0]
scale: [1.0, 1.0, 0.116, 0.116]
units: ["", "nanometer", "nanometer", "nanometer"]
optimize_reads: true

# Multiscale options (optional)
multiscale: false
ms_workers: 100
data_origin: "raw"  # or "labels"
antialiasing: false
normalize_voxel_size: false
custom_scale_factors : null

# LSF cluster options (if using cluster: "lsf")
log_dir: "/path/to/log/directory"
extra_directives: ["-P myproject"]
```

Then run:

```bash
pixi run zarrify --config config.yaml
```

You can also override specific config values from the command line:

```bash
pixi run zarrify --config config.yaml --workers 200 --scale 1.0 1.0 0.058 0.058
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
