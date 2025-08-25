from pydantic import BaseModel, validator, model_validator
from typing import Optional, Literal, List
import math
import zarr
import click
import yaml

class Config(BaseModel):
    src: str
    workers: int = 100
    data_origin: Literal['raw', 'segmentations']
    cluster: Optional[str] = None
    log_dir: Optional[str] = None
    antialiasing: bool = False
    high_aspect_ratio: bool = False
    custom_scale_factors:  Optional[List[List[float]]] = None
    
    @validator('src')
    def src_must_exist(cls, v):
        import os
        if not os.path.exists(v):
            raise ValueError(f'Source path {v} does not exist')
        return v
    
    @validator('workers')
    def workers_positive(cls, v):
        if v <= 0:
            raise ValueError('Workers must be positive')
        return v
    
    @model_validator(mode="before")
    def check_dimensions(cls, values):
        src = values.get('src')
        scale_factors = values.get('custom_scale_factors')
        z_root = zarr.open(src)
        
        try:
            s0 = z_root['s0']
        except:
            raise ValueError(f"No s0 array found in {src}")
        
        s0_shape = s0.shape
        
        for window in scale_factors:
            if len(window) != len(s0_shape):
                raise ValueError(f'number of dimensions of the window and array shape must match')
         
        # check that each scale is a factor of 2:
        if not all([x > 0 and abs(math.log2(x) - round(math.log2(x))) < 1e-10 for xs in scale_factors for x in xs]):
            raise ValueError(f'Some of the factors are not a power of 2.')
        
        if any([int(s0_dim/ sc) > 32 for s0_dim, sc in zip(s0_shape, scale_factors[-1])]):
            raise ValueError('not enough custom scale levels to generate full multiscale pyramid')

        return values
    
def validate_config(config, **kwargs):
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Override config with CLI arguments (CLI takes precedence)
        for key, value in kwargs.items():
            if value is not None:  # Only override if CLI arg was provided
                config_data[key] = value
        
        # Validate config
        try:
            return Config(**config_data).dict()
        except Exception as e:
            click.echo(f"Configuration validation error: {e}", err=True)
            raise click.Abort()