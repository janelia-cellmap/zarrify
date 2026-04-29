import logging
import time
from pathlib import Path
from typing import Union

import click
import zarr
from dask.distributed import Client

from zarrify.formats.mrc import Mrc3D
from zarrify.formats.n5 import N5Group
from zarrify.formats.tiff import Tiff
from zarrify.formats.tiff_stack import TiffStack
from zarrify.formats.zarr2 import Zarr2Group
from zarrify.utils.dask_utils import initialize_dask_client, check_shardslab_fits_in_ram
from zarrify.utils.pydantic_models import validate_config
from zarrify.utils.ts_utils import align_shard_to_chunks, build_codec, zarr3_spec
from zarrify.utils.zarr_utils import create_output_array

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

    if any(part.endswith('.n5') for part in src_path.parts):
        return N5Group(*params)
    elif ext == '.zarr':
        if (src_path / '.zgroup').exists():
            return Zarr2Group(*params)
        raise ValueError(f"zarr v3 as source is not supported: {src}")
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
            shard_shape : list[int] | None = None,
            axes : list[str] = ['c', 'z', 'y', 'x'],
            scale : list[float] = [1.0,]*4,
            translation : list[float] = [0.0,]*4,
            units: list[str] = ['']+['nanometer',]*3,
            optimize_reads : bool = False,
            expand_dims : bool = False,
            multiscale : bool = False,
            ms_workers: int = None,
            data_origin : str = None,
            antialiasing : bool = False,
            normalize_voxel_size : bool = False,
            custom_scale_factors : str = None,
            codec : str = 'zstd',
            codec_level : int | None = None,
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
        shard_shape (list[int] | None, optional): outer shard shape for sharded zarr3 arrays.
        expand_dims (bool, optional): prepend a size-1 channel dimension to the output array.
        codec (str, optional): compression codec name ('zstd', 'gzip', 'blosc'). Defaults to 'zstd'.
        codec_level (int | None, optional): codec compression level. None uses per-codec default.
    """
    logger.info(f"Initializing dataset from {src}")
    dataset = init_dataset(src, axes, scale, translation, units, optimize_reads)
    logger.info(f"Dataset type: {type(dataset).__name__}")

    # Handle N5Group separately as it has custom zarr creation logic
    codec_dict = build_codec(codec, codec_level)

    if isinstance(dataset, N5Group):
        logger.info("Detected N5Group — scaling workers and starting write")
        client.cluster.scale(workers)
        dataset.write_to_zarr(dest, client, zarr_chunks, shard_shape=shard_shape, codec=codec_dict,
                              expand_dims=expand_dims)
        client.cluster.scale(0)
        return

    if isinstance(dataset, Zarr2Group):
        logger.info("Detected Zarr2Group — scaling workers and starting write")
        client.cluster.scale(workers)
        dataset.write_to_zarr(str(dest), client, zarr_chunks, shard_shape=shard_shape, codec=codec_dict,
                              expand_dims=expand_dims)
        client.cluster.scale(0)
        logger.info("Zarr2Group write complete")
        return
    else:
        logger.info(f"Input dataset: {type(dataset)}")
        logger.info(f"Input dataset shape: {dataset.shape}")
        logger.info(f"Input dataset dtype: {dataset.dtype}")
        logger.info(f"Input dataset ndim: {dataset.ndim}")

        # Tiff trims metadata to source ndim in __init__; TiffStack and MRC do not.
        # Normalise here so all formats are consistent before the expand step.
        for key in ('axes', 'units', 'scale', 'translation'):
            dataset.metadata[key] = list(dataset.metadata[key])[-len(dataset.shape):]

        # When expanding dims, prepend a channel axis to OME metadata so that
        # add_ome_metadata writes axes/scale/translation/units for ndim+1.
        if expand_dims:
            dataset.metadata['axes'] = ['c'] + list(dataset.metadata['axes'])
            dataset.metadata['units'] = [''] + list(dataset.metadata['units'])
            dataset.metadata['scale'] = [1.0] + list(dataset.metadata['scale'])
            dataset.metadata['translation'] = [0.0] + list(dataset.metadata['translation'])

        out_shape = (1, *dataset.shape) if expand_dims else dataset.shape

        # Reshape chunks and shard to match output dimensionality
        logger.info(f"Zarr chunks: {zarr_chunks}")
        if len(zarr_chunks) != len(out_shape):
            logger.info(f"Reshaping chunks to match data dimensionality")
            zarr_chunks = dataset.reshape_to_arr_shape(zarr_chunks, out_shape)
            logger.info(f"Reshaped chunks: {zarr_chunks}")
        if shard_shape is not None:
            shard_shape = list(shard_shape)[-len(out_shape):]
            shard_shape = align_shard_to_chunks(
                [min(s, dim) for s, dim in zip(shard_shape, out_shape)],
                zarr_chunks,
            )
            logger.info(f"Aligned shard_shape to {shard_shape}")

        # check multiscale custom scale parameters before expensive data copying
        if multiscale:
            if not ms_workers:
                ms_workers = workers
            if not data_origin:
                raise ValueError('Data origin (raw/labels) is not specified')

            if custom_scale_factors:
                if any([int(s0_dim/ sc) > 32 for s0_dim, sc in zip(out_shape, custom_scale_factors[-1])]):
                    raise ValueError('Not enough custom scale levels to generate full multiscale pyramid')

        # Create output zarr3 array via TensorStore
        full_scale_arr_name = 's0'
        create_output_array(dest, out_shape, dataset.dtype, zarr_chunks,
                            shard_shape=shard_shape, codec=codec_dict, array_path=full_scale_arr_name)
        dest_spec = zarr3_spec(store_path=dest, array_path=full_scale_arr_name)
        logger.info(f"Created output Zarr array at {dest}/{full_scale_arr_name}")

        # Populate zarr metadata
        dataset.add_ome_metadata(dest, full_scale_arr_name)
        logger.info(f"Added OME-Zarr metadata")

        # Write data using new signature
        logger.info(f"Writing data to Zarr arrays...")
        client.cluster.scale(workers)
        if shard_shape is not None:
            check_shardslab_fits_in_ram(shard_shape, dataset.dtype, zarr_chunks, client)
        dataset.write_to_zarr(dest_spec, client, expand_dims=expand_dims)
        client.cluster.scale(0)
        logger.info(f"Completed writing all data to Zarr arrays")

        # create multiscale
        if multiscale:
            logger.info(f"create multiscale pyramid")
            z_root = zarr.open(zarr.storage.LocalStore(dest), mode='a')
            client.cluster.scale(ms_workers)
            dataset.create_multiscale(dest,
                                    z_root,
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
            shard_shape=configs.get('shard_shape'),
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
            custom_scale_factors=configs['custom_scale_factors'],
            codec=configs.get('codec', 'zstd'),
            codec_level=configs.get('codec_level'),
    )
    
if __name__ == "__main__":
    cli()
