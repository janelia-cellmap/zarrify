import json
import logging
import os
import time
from itertools import chain
from operator import itemgetter
from typing import Tuple

import natsort
import numpy as np
import pint
import pydantic_zarr as pz
import zarr
from dask.array.core import normalize_chunks, slices_from_chunks
from dask.distributed import Client, wait
from toolz import partition_all

from zarrify.utils.ts_utils import n5_spec, zarr3_spec, open_ts, zstd_codec
from zarrify.utils.volume import Volume

class N5Group(Volume):
    
    def __init__(
        self,
        src_path: str,
        axes: list[str],
        scale: list[float],
        translation: list[float],
        units: list[str],
    ):
        """Construct all the necessary attributes for the proper conversion of tiff to OME-NGFF Zarr.

        Args:
            input_filepath (str): path to source tiff file.
        """
        super().__init__(src_path, axes, scale, translation, units)
        self.store_path, self.path = self.separate_store_path(src_path, '')
        self.n5_store = zarr.N5Store(self.store_path)
        
        self.n5_obj = zarr.open(store = self.n5_store, path=self.path, mode='r')

    def separate_store_path(self,
                            store : str,
                            path : str):
        """
        sometimes you can pass a total os path to node, leading to
        an empty('') node.path attribute.
        the correct way is to separate path to container(.n5, .zarr)
        from path to array within a container.

        Args:
            store (string): path to store
            path (string): path array/group (.n5 or .zarr)

        Returns:
            (string, string): returns regularized store and group/array path
        """
        new_store, path_prefix = os.path.split(store)
        if ".n5" in path_prefix:
            return store, path
        return self.separate_store_path(new_store, os.path.join(path_prefix, path))
    
    #creates attributes.json recursively within n5 group, if missing
    def reconstruct_json(self, n5src: str) -> None:
        """Recursively create missing attributes.json files inside an N5 tree.

        Some N5 writers omit attributes.json for intermediate groups. zarr
        and pydantic-zarr require it to be present before they can read the
        hierarchy.

        Parameters
        ----------
        n5src:
            Absolute path to the N5 store root or any sub-directory within it.
        """
        dir_list = os.listdir(n5src)
        if "attributes.json" not in dir_list:
            with open(os.path.join(n5src, "attributes.json"), "w") as jfile:
                jfile.write(json.dumps({"n5": "2.0.0"}, indent=4))
        for obj in dir_list:
            if os.path.isdir(os.path.join(n5src, obj)):
                self.reconstruct_json(os.path.join(n5src, obj))
                
    def apply_ome_template(self, zgroup: zarr.Group) -> dict:
        """Build an OME-NGFF v0.4 multiscales attribute dict from N5 group attributes.

        Parameters
        ----------
        zgroup:
            Zarr group with N5-style "axes", "units", and "scales" attributes.

        Returns
        -------
        dict
            A dict suitable for writing to zgroup.attrs["multiscales"].
        """
        z_attrs: dict = {"multiscales": [{}]}

        # normalize input units, i.e. 'meter' or 'm'-> 'meter'
        ureg = pint.UnitRegistry()
        units_list = [ureg.Unit(unit) for unit in zgroup.attrs['units']]            

        #populate .zattrs
        z_attrs['multiscales'][0]['axes'] = [{"name": axis, 
                                            "type": "space",
                                            "unit": unit} for (axis, unit) in zip(zgroup.attrs['axes'], 
                                                                                    units_list)]
        z_attrs['multiscales'][0]['version'] = '0.4'
        z_attrs['multiscales'][0]['name'] = zgroup.name
        z_attrs['multiscales'][0]['coordinateTransformations'] = [{"type": "scale",
                        "scale": [1.0, 1.0, 1.0]}, {"type" : "translation", "translation" : [1.0, 1.0, 1.0]}]
        
        return z_attrs

    def normalize_to_omengff(self, zgroup: zarr.Group) -> None:
        """Recursively convert N5 metadata to OME-NGFF multiscales attributes.

        Parameters
        ----------
        zgroup:
            Root zarr group of the output zarr store.
        """
        group_keys = zgroup.keys()
        
        for key in chain(group_keys, '/'):
            if isinstance(zgroup[key], zarr.hierarchy.Group):
                if key!='/':
                    self.normalize_to_omengff(zgroup[key])
                if 'scales' in zgroup[key].attrs.asdict():
                    zattrs = self.apply_ome_template(zgroup[key])
                    zarrays = zgroup[key].arrays(recurse=True)

                    unsorted_datasets = []
                    for arr in zarrays:
                        unsorted_datasets.append(self.ome_dataset_metadata(arr[1], zgroup[key]))

                    #1.apply natural sort to organize datasets metadata array for different resolution degrees (s0 -> s10)
                    #2.add datasets metadata to the omengff template
                    zattrs['multiscales'][0]['datasets'] = natsort.natsorted(unsorted_datasets, key=itemgetter(*['path']))
                    zgroup[key].attrs['multiscales'] = zattrs['multiscales']
                    
    @staticmethod
    def ome_dataset_metadata(n5arr: zarr.Array, group: zarr.Group) -> dict:
        """Build one OME-NGFF dataset metadata entry from an N5 array.

        Parameters
        ----------
        n5arr:
            Source N5 array with a "transform" attribute.
        group:
            Parent group used to compute the relative path.

        Returns
        -------
        dict
            A single entry suitable for the "datasets" list in multiscales.
        """
    
        arr_attrs_n5 = n5arr.attrs['transform']
        dataset_meta =  {
                        "path": os.path.relpath(n5arr.path, group.path),
                        "coordinateTransformations": [{
                            'type': 'scale',
                            'scale': arr_attrs_n5['scale']},{
                            'type': 'translation',
                            'translation' : arr_attrs_n5['translate']
                        }]}
        
        return dataset_meta
    
    # d=groupspec.to_dict(),
    def normalize_groupspec(self, d: dict, codec: dict) -> None:
        for k, v in d.items():
            if k == "compressor":
                d[k] = codec
            elif k == "dimension_separator":
                d[k] = "/"
            elif isinstance(v, dict):
                self.normalize_groupspec(v, codec)

    def copy_n5_tree(
        self,
        n5_root: zarr.Group,
        z_store: zarr.storage.LocalStore,
        codec: dict,
    ) -> zarr.Group:
        spec_n5 = pz.GroupSpec.from_zarr(n5_root)
        spec_n5_dict = spec_n5.model_dump()
        self.normalize_groupspec(spec_n5_dict, codec)
        spec_n5 = pz.GroupSpec(**spec_n5_dict)
        return spec_n5.to_zarr(z_store, path="")

    def write_to_zarr(
        self,
        dest: str,
        client: Client,
        zarr_chunks : list[int],
        comp : ABCMeta = Zstd(level=6),
    ):
        
        logging.basicConfig(level=logging.INFO, 
                         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        self.reconstruct_json(os.path.join(self.store_path, self.path))
        
        # input n5 arrays list to convert
        n5_root = zarr.open_group(self.n5_store, mode = 'r')
        zarr_arrays = n5_root.arrays(recurse=True)
        # copy input n5 tree structure to output zarr and add ome-metadata, when N5 metadata is present
        z_store = zarr.storage.LocalStore(dest)
        zg = self.copy_n5_tree(n5_root, z_store, comp)

        self.normalize_to_omengff(zg)
        
        for item in zarr_arrays:
            n5arr = item[1]
            dest_arr = zarr.open_array(os.path.join(dest, n5arr.path), mode='a')
            
            out_slices = slices_from_chunks(normalize_chunks(dest_arr.chunks, shape=dest_arr.shape))
            # break the slices up into batches, to make things easier for the dask scheduler
            out_slices_partitioned = tuple(partition_all(100000, out_slices))
            
            for idx, part in enumerate(out_slices_partitioned):
                logging.info(f'{idx + 1} / {len(out_slices_partitioned)}')
                start = time.time()
                fut = client.map(lambda v: self.save_chunk(n5arr, dest_arr, v, invert=False), part)
                logging.info(f'Submitted {len(part)} tasks to the scheduler in {time.time()- start}s')
                # wait for all the futures to complete
                result = wait(fut)
                logging.info(f'Completed {len(part)} tasks in {time.time() - start}s')


def save_chunk(
    src_spec: dict,
    dst_spec: dict,
    chunk_slice: Tuple[slice, ...],
    invert: bool = False,
) -> None:
    """Copy one chunk from an N5 array into a zarr3 TensorStore array.

    All-zero chunks are skipped to avoid unnecessary writes.

    Parameters
    ----------
    src_spec:
        TensorStore N5 driver spec for the source array.
    dst_spec:
        TensorStore zarr3 driver spec for the destination array.
    chunk_slice:
        Slice tuple identifying the chunk region to copy.
    invert:
        When True, apply bitwise inversion to the data before writing.
    """
    src = open_ts(src_spec)
    data = src[chunk_slice].read().result()
    # only store data if it is not all 0s
    if (data == 0).all():
        return
    if invert:
        data = np.invert(data)
    dest = open_ts(dst_spec)
    dest[chunk_slice].write(data).result()
