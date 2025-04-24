import zarr
from numcodecs import Zstd
import os
import click
import sys
from dask.distributed import Client
import time
from zarrify.formats.tiff_stack import TiffStack
from zarrify.formats.tiff import Tiff3D
from zarrify.formats.mrc import Mrc3D
from zarrify.formats.n5 import N53D
from zarrify.utils.dask_utils import initialize_dask_client


# @click.command("zarrify")
# @click.option(
#     "--src",
#     "-s",
#     type=click.Path(exists=True),
#     help="Input file/directory location",
# )
# @click.option("--dest", "-s", type=click.STRING, help="Output .zarr file path.")
# @click.option(
#     "--num_workers", "-w", default=100, type=click.INT, help="Number of dask workers"
# )
# @click.option(
#     "--cluster",
#     "-c",
#     default="",
#     type=click.STRING,
#     help="Which instance of dask client to use. Local client - 'local', cluster 'lsf'",
# )
# @click.option(
#     "--zarr_chunks",
#     "-zc",
#     nargs=3,
#     default=(64, 128, 128),
#     type=click.INT,
#     help="Chunk size for (z, y, x) axis order. z-axis is normal to the tiff stack plane. Default (64, 128, 128)",
# )
# @click.option(
#     "--axes",
#     "-a",
#     nargs=3,
#     default=("z", "y", "x"),
#     type=str,
#     help="Metadata axis names. Order matters. \n Example: -a z y x",
# )
# @click.option(
#     "--translation",
#     "-t",
#     nargs=3,
#     default=(0.0, 0.0, 0.0),
#     type=float,
#     help="Metadata translation(offset) value. Order matters. \n Example: -t 1.0 2.0 3.0",
# )
# @click.option(
#     "--scale",
#     "-s",
#     nargs=3,
#     default=(1.0, 1.0, 1.0),
#     type=float,
#     help="Metadata scale value. Order matters. \n Example: -s 1.0 2.0 3.0",
# )
# @click.option(
#     "--units",
#     "-u",
#     nargs=3,
#     default=("nanometer", "nanometer", "nanometer"),
#     type=str,
#     help="Metadata unit names. Order matters. \n Example: -t nanometer nanometer nanometer",
# )
# def cli(src, dest, num_workers, cluster, zarr_chunks, axes, translation, scale, units):

    # create a dask client to submit tasks
#client = initialize_dask_client(cluster)

def to_zarr(src : str,
            dest: str,
            client : Client,
            num_workers : int = 20,
            zarr_chunks : list[int] = [128]*3,
            axes : list[str] = ('z', 'y', 'x'), 
            scale : list[float] = [1.0]*3,
            translation : list[float] = [0.0]*3,
            units: list[str] = ['nanometer']*3):   
    if '.n5' in src:
        dataset = N53D(src, axes, scale, translation, units)
    if src.endswith(".mrc"):
        dataset = Mrc3D(src, axes, scale, translation, units)
    elif src.endswith(".tif") or src.endswith(".tiff"):
        dataset = Tiff3D(src, axes, scale, translation, units)
    if os.path.isdir(src):
        dataset = TiffStack(src, axes, scale, translation, units)

    z_store = zarr.NestedDirectoryStore(dest)
    z_root = zarr.open(store=z_store, mode="a")
    z_arr = z_root.require_dataset(
        name="s0",
        shape=dataset.shape,
        dtype=dataset.dtype,
        chunks=zarr_chunks,
        compressor=Zstd(level=6),
    )

    # write in parallel to zarr using dask
    client.cluster.scale(num_workers)
    dataset.write_to_zarr(z_arr, client)
    client.cluster.scale(0)
    # populate zarr metadata
    dataset.add_ome_metadata(z_root)


# if __name__ == "__main__":
#     cli()
