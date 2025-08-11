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
from typing import Union


def init_dataset(src :str,
                 axes : list[str],
                 scale : list[float],
                 translation : list[float],
                 units : list[str]) -> Union[TiffStack, Tiff, N5Group, Mrc3D]:
    """Returns an instance of a dataset class (TiffStack, N5Group, Mrc3D, or Tiff), depending on the input file format.

    Args:
        src (str): source file/container location
        axes (list[str]): axis order (for ome-zarr metadata)
        scale (list[float]): voxel size (for ome-zarr metadata)
        translation (list[float]): offset (for ome-zarr metadata)
        units (list[str]): physical units (for ome-zarr metadata)

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
        return Tiff(*params)
    
    raise ValueError(f"Unsupported source type: {src}")

def to_zarr(src : str,
            dest: str,
            client : Client,
            workers : int = 20,
            zarr_chunks : list[int] = [3, 128, 128, 128],
            axes : list[str] = ['c','z', 'y', 'x'], 
            scale : list[float] = [1.0,]*4,
            translation : list[float] = [0.0,]*4,
            units: list[str] = ['nanometer',]*4):
    """Convert Tiff stack, 3D Tiff, N5, or MRC file to OME-Zarr.

    Args:
        src (str): input data location.
        dest (str): output zarr group location.
        client (Client): dask client instance.
        workers (int, optional): Number of dask workers. Defaults to 20.
        zarr_chunks (list[int], optional): _description_. Defaults to [128,]*3.
        axes (list[str], optional): axis order. Defaults to ['z', 'y', 'x'].
        scale (list[float], optional): voxel size (in physical units). Defaults to [1.0,]*3.
        translation (list[float], optional): offset (in physical units). Defaults to [0.0,]*3.
        units (list[str], optional): physical units. Defaults to ['nanometer']*3.
    """
    
    dataset = init_dataset(src, axes, scale, translation, units)

    # write in parallel to zarr using dask
    client.cluster.scale(workers)
    dataset.write_to_zarr(dest, client, zarr_chunks)
    client.cluster.scale(0)
    # populate zarr metadata
    dataset.add_ome_metadata(dest)


@click.command("zarrify")
@click.option(
    "--src",
    "-s",
    type=click.Path(exists=True),
    help="Input file/directory location",
)
@click.option("--dest", "-d", type=click.STRING, help="Output .zarr file path.")
@click.option(
    "--workers", "-w", default=100, type=click.INT, help="Number of dask workers"
)
@click.option(
    "--cluster",
    "-c",
    default=None,
    type=click.STRING,
    help="Which instance of dask client to use. Local client - 'local', cluster 'lsf'",
)
@click.option(
    "--zarr_chunks",
    "-zc",
    nargs=4,
    default=(3, 64, 128, 128),
    type=click.INT,
    help="Chunk size. For 3D: (z, y, x), for 4D RGB: (z, y, x, c). Examples: -zc 64 128 128 or -zc 10 64 128 3",
)
@click.option(
    "--axes",
    "-a",
    nargs=4,
    default=("c", "z", "y", "x"),
    type=str,
    help="Metadata axis names. Order matters. \n Example: -a z y x",
)
@click.option(
    "--translation",
    "-t",
    nargs=4,
    default=(0.0, 0.0, 0.0, 0.0),
    type=float,
    help="Metadata translation(offset) value. Order matters. \n Example: -t 1.0 2.0 3.0",
)
@click.option(
    "--scale",
    "-sc",
    nargs=4,
    default=(1.0, 1.0, 1.0, 1.0),
    type=float,
    help="Metadata scale value. Order matters. \n Example: --scale 1.0 2.0 3.0",
)
@click.option(
    "--units",
    "-u",
    nargs=4,
    default=("nanometer", "nanometer", "nanometer", "nanometer"),
    type=str,
    help="Metadata unit names. Order matters. \n Example: -t nanometer nanometer nanometer",
)
@click.option('--log_dir', default = None, type=click.STRING,
    help="The path of the parent directory for all LSF worker logs.  Omit if you want worker logs to be emailed to you.")
def cli(src, dest, workers, cluster, zarr_chunks, axes, translation, scale, units, log_dir):
    # create a dask client to submit tasks
    client = initialize_dask_client(cluster, log_dir)
    
    # convert src dataset(n5, tiff, mrc) to zarr ome dataset 
    to_zarr(src,
            dest,
            client,
            workers,
            zarr_chunks,
            axes, 
            scale,
            translation,
            units)

if __name__ == "__main__":
    cli()
