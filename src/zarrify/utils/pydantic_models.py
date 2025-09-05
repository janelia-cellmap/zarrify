from pydantic import BaseModel, field_validator, model_validator
from typing import Optional, Literal, List, Tuple
import math
import zarr
import click
import yaml
import pint


ureg = pint.UnitRegistry()

class ZarrifyConfig(BaseModel):
    src: str
    dest: str
    workers: int = 100
    cluster: Optional[str] = None
    log_dir: Optional[str] = None
    extra_directives: Optional[List[str]] = None
    
    zarr_chunks: List[int] = [3, 64, 128, 128]
    axes: List[str] = ["c", "z", "y", "x"]
    translation: List[float] = [0.0, 0.0, 0.0, 0.0]
    scale: List[float] = [1.0, 1.0, 1.0, 1.0]
    units: List[str] = ["", "nanometer", "nanometer", "nanometer"]
    optimize_reads: bool = True
    
    multiscale: bool = False
    ms_workers: int = 100
    data_origin: Literal['raw', 'labels']
    antialiasing: bool = False
    normalize_voxel_size: bool = False
    custom_scale_factors:  Optional[List[List[float]]] = None
    

    @field_validator('src')
    def src_exists(cls, v):
        import os
        if not os.path.exists(v):
            raise ValueError(f"Source path does not exist: {v}")
        return v

    @field_validator('dest')
    def non_empty_dest_path(cls, v):
        dest = v.replace(' ', '') if isinstance(v, str) else v
        if dest in ['', None]:
            raise ValueError(f"Empty destination path")
        return v

    # dask configs
    @field_validator('workers')
    def workers_positive(cls, v):
        if v <= 0:
            raise ValueError("Number of workers must be > 0")
        return v

    # output zarr arra configs
    @field_validator('zarr_chunks')
    def zarr_chunks_dtype(cls, v):
        if not all(isinstance(x, int) for x in v):
            raise TypeError("All zarr chunks dimensions must be integers")
        return v

    @field_validator('axes')
    def axes_dtype(cls, v):
        if not all(isinstance(x, str) for x in v):
            raise TypeError("All axes names must be strings")
        return v

    @field_validator('translation')
    def translation_dtype(cls, v):
        if not all(isinstance(x, float) for x in v):
            raise TypeError("All translation values must be floats")
        return v

    @field_validator('scale')
    def scale_length(cls, v):
        if not all(isinstance(x, float) for x in v):
            raise TypeError("All scale values must be floats")
        return v
    
    @field_validator("units")
    def units_and_valid(cls, v):
        for u in v:
            try:
                _ = ureg.Unit(u)  # check if it’s a valid unit
            except Exception:
                raise ValueError(f"Invalid physical unit: {u}")
        return v
    
    @field_validator('ms_workers')
    def ms_workers_positive(cls, v):
        if v <= 0:
            raise ValueError('Number of workers must be > 0')
        return v

    
    @model_validator(mode="before")
    def check_dimensions(cls, values):
                
        params = ["zarr_chunks",
                  "axes",
                  "units",
                  "scale",
                  "translation",
                  ]
        
        lengths = {
            f: len(values[f]) 
            for f in params 
            if f in values and values[f] is not None
        }

        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"Length mismatch among fields: {lengths}"
            )

        # custom scale factors            
        scale_factors = values.get('custom_scale_factors')
        if scale_factors:
            # check that each scale is a factor of 2:
            if not all([x > 0 and abs(math.log2(x) - round(math.log2(x))) < 1e-10 for xs in scale_factors for x in xs]):
                raise ValueError(f'Some of the factors are not a power of 2.')
            
            # check that multiscale factor windows shape==len(array.chunksize) (preliminary check)
            for window in scale_factors:
                if len(window) != len(values.get('zarr_chunks')):
                    raise ValueError(f'number of dimensions of the window and array dimensions must match')

        return values
    
def validate_config(config, **kwargs):
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Override config with CLI arguments (CLI takes precedence)
        for key, value in kwargs.items():
            if value is not None: 
                config_data[key] = value
        
        # Validate config
        try:
            return ZarrifyConfig(**config_data).model_dump()
        except Exception as e:
            click.echo(f"Configuration validation error: {e}", err=True)
            raise click.Abort()