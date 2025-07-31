# zarrify

Convert TIFF/TIFF Stacks, MRC, and N5 files to Zarr (v2) format with ome-ngff (0.4) metadata.

## Install

```bash
pip install zarrify
```

## Usage

### Local processing
```bash
zarrify --src input.tiff --dest output.zarr --cluster local --workers 20
```

### LSF cluster processing
```bash
bsub -n 1 -J to_zarr 'zarrify --src input.tiff --dest output.zarr --cluster lsf --workers 20'
```

## Python API

Integrating conversion to zarr into python script

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
