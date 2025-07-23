# zarrify

Convert TIFF, MRC, and N5 files to OME-Zarr format.

## Install

```bash
pip install zarrify
```

## Usage

```bash
zarrify --src input.tiff --dest output.zarr --cluster local
```

## Python API

```python
import zarrify
from zarrify.utils.dask_utils import initialize_dask_client

client = initialize_dask_client("local")
zarrify.to_zarr("input.tiff", "output.zarr", client)
```

## Supported formats

- TIFF stacks
- 3D TIFF files
- MRC files  
- N5 containers
