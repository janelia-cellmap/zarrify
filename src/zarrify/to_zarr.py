import zarr
from numcodecs import Zstd
from pathlib import Path
import click
import sys
from dask.distributed import Client
import time
from zarrify.formats.tiff_stack import TiffStack
from zarrify.formats.tiff import Tiff
from zarrify.formats.mrc import Mrc3D
from zarrify.formats.n5 import N5Group
from zarrify.utils.dask_utils import initialize_dask_client
from zarrify.utils.zarr_utils import create_output_array
from zarrify.utils.pydantic_models import validate_config
from typing import Union
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

def init_dataset(src :str,
                 axes : list[str],
                 scale : list[float],
                 translation : list[float],
                 units : list[str],
                 optimize_reads : bool = False) -> Union[TiffStack, Tiff, N5Group, Mrc3D]:
    """Returns an instance of a dataset class (TiffStack, N5Group, Mrc3D, or Tiff), depending on the input file format.

    Args:
        src (str): source file/container location
        axes (list[str]): axis order (for ome-zarr metadata)
        scale (list[float]): voxel size (for ome-zarr metadata)
        translation (list[float]): offset (for ome-zarr metadata)
        units (list[str]): physical units (for ome-zarr metadata)
        optimize_reads (bool): enable optimized TIFF loading using chunk-aligned reads.
    Raises:
        ValueError: return value error if the input file format not in the list.

    Returns:
        Union[TiffStack, Tiff, N5Group, Mrc3D]: return a file format object depending on the input file format. 
        \n All different file formats objects have identical instance methods (write_to_zarr, add_ome_metadata) to emulate API abstraction. 
    """
    
    src_path = Path(src)
    params = (src, axes, scale, translation, units)
    
    ext = src_path.suffix.lower()

    if ext=='.n5':
        return N5Group(*params)
    elif src_path.is_dir():
        return TiffStack(*params)
    elif ext == ".mrc":
        return Mrc3D(*params)
    elif ext in (".tif", ".tiff"):
        return Tiff(*params, optimize_reads)
    
    raise ValueError(f"Unsupported source type: {src}")

def to_zarr(src : str,
            dest: str,
            client : Client,
            workers : int = 20,
            zarr_chunks : list[int] = [3, 128, 128, 128],
            axes : list[str] = ['c', 'z', 'y', 'x'],
            scale : list[float] = [1.0,]*4,
            translation : list[float] = [0.0,]*4,
            units: list[str] = ['']+['nanometer',]*3,
            optimize_reads : bool = False,
            multiscale : bool = False,
            ms_workers: int = None,
            data_origin : str = None,
            antialiasing : bool = False,
            normalize_voxel_size : bool = False,
            custom_scale_factors : str = None
            ):
    """Convert Tiff stack, 3D Tiff, N5, or MRC file to OME-Zarr.

    Args:
        src (str): input data location.
        dest (str): output zarr group location.
        client (Client): dask client instance.
        workers (int, optional): Number of dask workers. Defaults to 20.
        zarr_chunks (list[int], optional): _description_. Defaults to [128,]*4.
        axes (list[str], optional): axis order. Defaults to ['c', 'z', 'y', 'x'].
        scale (list[float], optional): voxel size (in physical units). Defaults to [1.0,]*4.
        translation (list[float], optional): offset (in physical units). Defaults to [0.0,]*4.
        units (list[str], optional): physical units. Defaults to ['']+['nanometer']*3.
    """
    dataset = init_dataset(src, axes, scale, translation, units, optimize_reads)
    
    # Handle N5Group separately as it has custom zarr creation logic
    if isinstance(dataset, N5Group):
        # N5 handles zarr creation internally due to tree structure complexity
        client.cluster.scale(workers)
        dataset.write_to_zarr(dest, client, zarr_chunks)
        client.cluster.scale(0)

        # populate zarr metadata
        dataset.add_ome_metadata(dest)
        return
    else:
        logger.info(f"Input dataset: {type(dataset)}")
        logger.info(f"Input dataset shape: {dataset.shape}")
        logger.info(f"Input dataset dtype: {dataset.dtype}")
        logger.info(f"Input dataset ndim: {dataset.ndim}")
        # Reshape chunks to match data dimensionality
        logger.info(f"Zarr chunks: {zarr_chunks}")
        if len(zarr_chunks) != len(dataset.shape):
            logger.info(f"Reshaping chunks to match data dimensionality")
            zarr_chunks = dataset.reshape_to_arr_shape(zarr_chunks, dataset.shape)
            logger.info(f"Reshaped chunks: {zarr_chunks}")
            
        # check multiscale custom scale parameters before expensive data copying
        if multiscale:
            if not ms_workers:
                ms_workers = workers
            if not data_origin:
                raise ValueError('Data origin (raw/labels) is not specified')
            
            if custom_scale_factors:    
                if any([int(s0_dim/ sc) > 32 for s0_dim, sc in zip(dataset.shape, custom_scale_factors[-1])]):
                    raise ValueError('Not enough custom scale levels to generate full multiscale pyramid')

        z_store = zarr.NestedDirectoryStore(dest)
        z_root = zarr.open(store=z_store, mode="a")

        # Create zarr array externally
        zarr_array = create_output_array(z_root, dataset.shape, dataset.dtype, zarr_chunks, Zstd(level=6))
        logger.info(f"Created output Zarr: {zarr_array}")

        # Populate zarr metadata
        full_scale_group_name = zarr_array.name.lstrip('/')
        dataset.add_ome_metadata(dest, full_scale_group_name)
        logger.info(f"Added OME-Zarr metadata")

        # Write data using new signature
        logger.info(f"Writing data to Zarr arrays...")
        client.cluster.scale(workers)
        dataset.write_to_zarr(zarr_array, client)
        client.cluster.scale(0)
        logger.info(f"Completed writing all data to Zarr arrays")
        
        # create multiscale
        if multiscale:
            logger.info(f"create multiscale pyramid")
            client.cluster.scale(ms_workers)
            dataset.create_multiscale(z_root,
                                    client,
                                    data_origin,
                                    antialiasing,
                                    normalize_voxel_size,
                                    custom_scale_factors)
            client.cluster.scale(0)
            logger.info(f"Completed multiscal pyramid creation")
        
    

@click.command("zarrify")
@click.option(
    "--config",
    "-cfg",
    type=click.Path(exists=True),
    help="Path to YAML config file with parameters.",
)
@click.option(
    "--src",
    "-s",
    type=click.Path(exists=True),
    help="Input file/directory location",
)
@click.option("--dest", "-d", type=click.STRING, help="Output .zarr file path.")
@click.option(
    "--workers", "-w", type=click.INT, help="Number of dask workers"
)
@click.option(
    "--cluster",
    "-c",
    type=click.STRING,
    help="Which instance of dask client to use. Local client - 'local', cluster 'lsf'",
)
@click.option(
    "--zarr_chunks",
    "-zc",
    nargs=4,
    type=click.INT,
    help="Chunk size. For 3D: (z, y, x), for 4D RGB: (z, y, x, c). Examples: -zc 64 128 128 or -zc 10 64 128 3",
)
@click.option(
    "--axes",
    "-a",
    nargs=4,
    type=str,
    help="Metadata axis names. Order matters. \n Example: -a z y x",
)
@click.option(
    "--translation",
    "-t",
    nargs=4,
    type=float,
    help="Metadata translation(offset) value. Order matters. \n Example: -t 1.0 2.0 3.0",
)
@click.option(
    "--scale",
    "-sc",
    nargs=4,
    type=float,
    help="Metadata scale value. Order matters. \n Example: --scale 1.0 2.0 3.0",
)
@click.option(
    "--units",
    "-u",
    nargs=4,
    type=str,
    help="Metadata unit names. Order matters. \n Example: -u nanometer nanometer nanometer",
)
@click.option(
    "--log_dir",
    "-l",
    type=click.STRING,
    help="The path of the parent directory for all LSF worker logs. Omit if you want worker logs to be emailed to you.",
)
@click.option(
    "--extra_directives",
    "-e",
    type=click.STRING,
    multiple=True,
    help="Additional LSF job directives (e.g., -P project_name). Can be specified multiple times.",
)
@click.option(
    "--optimize_reads",
    is_flag=True,
    help="Enable optimized image loading using chunk-aligned reads.",
)

#def cli(src, dest, workers, cluster, zarr_chunks, axes, translation, scale, units, log_dir, extra_directives, optimize_reads):
def cli(config, **kwargs):
    logger.info(f"Starting Zarrify...")
    if config:
        configs = validate_config(config, **kwargs)
    else:
        configs = {k: v for k, v in kwargs.items() if v not in (None, (), [], {})}
    
    # create a dask client to submit tasks
    client = initialize_dask_client(configs['cluster'],
                                    configs.get('log_dir', None),
                                    configs.get('extra_directives', None))
    
    # convert src dataset(n5, tiff, mrc) to zarr ome dataset 
    logger.info(configs)
    to_zarr(src=configs['src'],
            dest=configs['dest'],
            client=client,
            workers=configs['workers'],
            zarr_chunks=configs['zarr_chunks'],
            axes=configs['axes'], 
            scale=configs['scale'],
            translation=configs['translation'],
            units=configs['units'],
            optimize_reads=configs['optimize_reads'],
            multiscale=configs['multiscale'],
            ms_workers=configs['ms_workers'],
            data_origin=configs['data_origin'],
            antialiasing=configs['antialiasing'],
            normalize_voxel_size=configs['normalize_voxel_size'],
            custom_scale_factors=configs['custom_scale_factors']
    )
    
if __name__ == "__main__":
    cli()
