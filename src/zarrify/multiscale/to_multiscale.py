import os
from typing import Tuple, List
from xarray_multiscale import windowed_mean, windowed_mode
#from xarray_multiscale.reducers import windowed_mode_countless, windowed_mode_scipy
import zarr
import time
from dask.array.core import slices_from_chunks, normalize_chunks
from dask.distributed import Client, wait
from toolz import partition_all
import click
import math
import scipy.ndimage as ndi
from dask_utils import initialize_dask_client
from pydantic_models import validate_config

def upscale_slice(slc: slice, factor: int):
    """
    Returns input slice coordinates. 

    Args:
        slc : output slice
        factor : upsampling factor
    Returns:
        slice: source array slice
    """
    return slice(slc.start * factor, slc.stop * factor, slc.step)

def downsample_save_chunk_mode(
        source: zarr.Array, 
        dest: zarr.Array, 
        out_slices: Tuple[slice, ...],
        downsampling_factors: Tuple[int, ...],
        data_origin: str,
        antialiasing : bool):
    
    """
    Downsamples source array slice and writes into destination array. 

    Args:
        source : source zarr array that needs to be downsampled
        dest : destination zarr array that contains downsampled data
        out_slices : part of the destination array that would contain the output data
        downsampling_factors : tuple that contains downsampling factors. dim(downsampling_factors) must match the shape of the source array
        data_origin : affects which downsampling method is used. Accepts two values: segmentation or raw
    """
    
    in_slices = tuple(upscale_slice(out_slice, fact) for out_slice, fact in zip(out_slices, downsampling_factors))
    source_data = source[in_slices]
    # only downsample source_data if it is not all 0s
    if not (source_data == 0).all():
        if data_origin == 'segmentations':
            ds_data = windowed_mode(source_data, window_size=downsampling_factors)
        elif data_origin == 'raw':
            if antialiasing:
                # blur data in chunk before downsampling to reduce aliasing of the image 
                # conservative Gaussian blur coeff: 2/2.5 = 0.8
                sigma = [0 if factor == 1 else factor/2.5 for factor in downsampling_factors]
                filtered_data = ndi.gaussian_filter(source_data, sigma=sigma)
                ds_data = windowed_mean(filtered_data, window_size=downsampling_factors)
            else:
                ds_data = windowed_mean(source_data, window_size=downsampling_factors)

        dest[out_slices] = ds_data
    return 0

def get_downsampling_factors(shape, axes_order, min_ratio=0.5, max_ratio=2.0, high_aspect_ratio=False):
    """
    Calculate adaptive downsampling factors based on aspect ratios.
    
    Args:
        shape: Array shape (c, z, y, x)
        axes_order: List of axis names ['c', 'z', 'y', 'x']
        min_ratio: Minimum allowed ratio (default 0.5)
        max_ratio: Maximum allowed ratio (default 2.0)
    
    Returns:
        List of downsampling factors
    """
    if high_aspect_ratio==False:
        return [ 1 if axis in ['c', 't'] else 2 for axis in axes_order]
    else:
        # Get spatial dimensions (skip channel and time)
        spatial_dims = list((axis, shape[i]) for i, axis in enumerate(axes_order) if axis not in ['c', 't'])
        
        axes, dimensions = zip(*spatial_dims)
    
        if len(dimensions) == 1:
            factors = [2]
        else:
            ratios = []
            for i, dim in enumerate(dimensions):
                # Calculate ratios of current dimension to all others
                # for example for 3D:  [(z/y, z/x),(y/z, y/x),(x/z, x/y)]
                dim_ratios = tuple(dim / dimensions[j] for j in range(len(dimensions)) if j != i)
                ratios.append(dim_ratios)
            
            # Determine downsampling factors for each spatial dimension
            factors = []
            for (i,dim_ratios) in enumerate(ratios):
                # Check if both ratios are within acceptable range
                if all(ratio >= max_ratio for ratio in dim_ratios):
                    factors = [1,]*len(ratios)
                    factors[i] = 2
                    break
                elif all(ratio <= min_ratio for ratio in dim_ratios):
                    factors = [2,]*len(ratios)
                    factors[i] = 1
                    break
                else:
                    factors.append(2)
        
        spatial_factors = {k : v for k,v in zip(axes, factors)}
        return tuple(1 if axis in ['c', 't'] else spatial_factors[axis] for axis in axes_order)

def create_multiscale(z_root: zarr.Group,
                      client: Client,
                      data_origin: str,
                      antialiasing : bool,
                      high_aspect_ratio : bool,
                      custom_scale_factors : List[List[float]]
                      ):
    """
    Creates multiscale pyramid and write corresponding metadata into .zattrs 

    Args:
        z_root : parent group for source zarr array
        client : Dask client instance
        num_workers : Number of dask workers
        data_origin : affects which downsampling method is used. Accepts two values: 'segmentation' or 'raw'
    """
    # store original array in a new .zarr file as an arr_name scale
    z_attrs = z_root.attrs.asdict() 
    scn_level_up = z_attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']
    trn_level_up = z_attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][1]['translation']
        
    level = 1
    source_shape = z_root[f's{level-1}'].shape
    axes_order = [axis['name'].lower() for axis in z_attrs['multiscales'][0]['axes']]
    spatial_shape = list(source_shape[i] for i, axis in enumerate(axes_order) if axis not in ['c', 't'])
    
    #continue downsampling if output array dimensions > 32 
    while all([dim > 32 for dim in spatial_shape]):
        print(f'{level=}')
        source_arr = z_root[f's{level-1}']
        
        if custom_scale_factors:
            scaling_factors = tuple(int(sc_cur/sc_prev) for sc_prev, sc_cur in zip(custom_scale_factors[level-1], custom_scale_factors[level]))
        else:
            scaling_factors = get_downsampling_factors(source_arr.shape, axes_order, high_aspect_ratio=high_aspect_ratio)

        dest_shape = [math.floor(dim / scaling) for dim, scaling in zip(source_arr.shape, scaling_factors)]

        # initialize output array
        dest_arr = z_root.require_dataset(
            f's{level}', 
            shape=dest_shape, 
            chunks=source_arr.chunks, 
            dtype=source_arr.dtype, 
            compressor=source_arr.compressor, 
            dimension_separator='/',
            fill_value=0,
            exact=True)
                
        assert dest_arr.chunks == source_arr.chunks
        out_slices = slices_from_chunks(normalize_chunks(source_arr.chunks, shape=dest_arr.shape))
        
        #break the slices up into batches, to make things easier for the dask scheduler
        out_slices_partitioned = tuple(partition_all(100000, out_slices))
        for idx, part in enumerate(out_slices_partitioned):
            print(f'{idx + 1} / {len(out_slices_partitioned)}')
            start = time.time()
            fut = client.map(lambda v: downsample_save_chunk_mode(source_arr, dest_arr, v, scaling_factors, data_origin, antialiasing), part)
            print(f'Submitted {len(part)} tasks to the scheduler in {time.time()- start}s')
            
            # wait for all the futures to complete
            result = wait(fut)
            print(f'Completed {len(part)} tasks in {time.time() - start}s')
            
        # calculate scale and transalation for n-th scale
        sn = [sc*sn_prev for sn_prev, sc in zip(scn_level_up, scaling_factors)]
        trn = [(sc -1)*sn_prev/(sc) + trn_prev for sn_prev, trn_prev, sc
               in zip(scn_level_up, trn_level_up, scaling_factors)]
                
        # Convert datasets list to dict for easier lookup/replacement
        datasets = z_attrs['multiscales'][0]['datasets']
        datasets_dict = {d['path']: d for d in datasets}

        # Update or add
        datasets_dict[f's{level}'] = {
            'coordinateTransformations': [
                {"type": "scale", "scale": sn}, 
                {"type": "translation", "translation": trn}
            ],
            'path': f's{level}'
        }
        # Convert back to list (maintaining order)
        z_attrs['multiscales'][0]['datasets'] = list(datasets_dict.values())
        
        #prepare data for downsampling next level
        level += 1
        scn_level_up = sn
        trn_level_up = trn
        spatial_shape = list(dest_shape[i] for i, axis in enumerate(axes_order) if axis not in ['c', 't'])
    
    #write multiscale metadata into .zattrs
    z_root.attrs['multiscales'] = z_attrs['multiscales']
        
@click.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--src','-s',type=click.Path(exists = True),help='Input .zarr file location.')
@click.option('--workers','-w',type=click.INT, help = "Number of dask workers")
@click.option('--data_origin','-do',type=click.STRING,help='Different data requires different type of interpolation. Raw fibsem data - use \'raw\', for segmentations - use \'segmentations\'')
@click.option('--cluster', '-cl',type=click.STRING, help="Which instance of dask client to use. Local client - 'local', cluster 'lsf'")
@click.option('--log_dir', type=click.STRING,
    help="The path of the parent directory for all LSF worker logs.  Omit if you want worker logs to be emailed to you.")
@click.option('--antialiasing', '-aa', type=click.BOOL, help='Reduce aliasing of the image by blurring it with Gaussian filter before downsampling. Default: False')
@click.option('--high_aspect_ratio', '--har', type=click.BOOL, help='Reduce aspect ratio of the data array more and more with each downsampling level. Metadata in zattrs is being recomputed accordingly. Default: False.')
def cli(config, **kwargs):
    
    if config:
        config_dict = validate_config(config, **kwargs)
    else:
        config_dict = {k: v for k, v in kwargs.items() if v is not None}
    
    src_store = zarr.NestedDirectoryStore(config_dict['src'])
    src_root = zarr.open_group(store=src_store, mode = 'a')

    client = initialize_dask_client(cluster_type=config_dict['cluster'],
                                    log_dir=config_dict.get('log_dir', None))
    client.cluster.scale(int(config_dict['workers']))
    create_multiscale(z_root=src_root,
                      client=client,
                      data_origin=config_dict['data_origin'],
                      antialiasing=config_dict.get('antialiasing', False),
                      high_aspect_ratio=config_dict.get('high_aspect_ratio', False),
                      custom_scale_factors = config_dict.get('custom_scale_factors', None)
                      )
    
if __name__ == '__main__':
    cli()

