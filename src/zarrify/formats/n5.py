import zarr
import os
from zarrify.utils.volume import Volume
import json
from itertools import chain
import natsort
from operator import itemgetter
import pydantic_zarr as pz
import dask.array as da
from dask.array.core import slices_from_chunks, normalize_chunks
from toolz import partition_all
from dask.distributed import Client, wait
from typing import Tuple
from time import time
import numpy as np
import logging 
import pint
from abc import ABCMeta
from numcodecs import Zstd

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

        self.shape = self.n5_arr.shape
        self.dtype = self.n5_arr.dtype
        self.chunks = self.n5_arr.chunks
        self.compressor = self.n5_arr.compressor

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
    def reconstruct_json(self,
                         n5src : str):
        dir_list = os.listdir(n5src)
        if "attributes.json" not in dir_list:
            with open(os.path.join(n5src,"attributes.json"), "w") as jfile:
                dict = {"n5": "2.0.0"}
                jfile.write(json.dumps(dict, indent=4))
        for obj in dir_list:
            if os.path.isdir(os.path.join(n5src, obj)):
                self.reconstruct_json(os.path.join(n5src, obj))
                
    def apply_ome_template(self, zgroup : zarr.Group):
    
        z_attrs = {"multiscales": [{}]}

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

    def normalize_to_omengff(self,
                             zgroup : zarr.Group):
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
                    
    def ome_dataset_metadata(n5arr : zarr.Array,
                             group : zarr.Group):
    
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
    def normalize_groupspec(self, d, comp):
        for k,v in d.items():
            if k == "compressor":
                d[k] = comp.get_config()

            elif k == 'dimension_separator':
                d[k] = '/'
            elif isinstance(v,  dict):
                self.normalize_groupspec(v, comp)

    def copy_n5_tree(self, n5_root, z_store, comp):
        spec_n5 = pz.GroupSpec.from_zarr(n5_root)
        spec_n5_dict = spec_n5.dict()
        self.normalize_groupspec(spec_n5_dict, comp)
        spec_n5 = pz.GroupSpec(**spec_n5_dict)
        return spec_n5.to_zarr(z_store, path= '')
        
    
    #creates attributes.json, if missing 
    def reconstruct_json(self, n5src):
        dir_list = os.listdir(n5src)
        if "attributes.json" not in dir_list:
            with open(os.path.join(n5src,"attributes.json"), "w") as jfile:
                dict = {"n5": "2.0.0"}
                jfile.write(json.dumps(dict, indent=4))
        for obj in dir_list:
            if os.path.isdir(os.path.join(n5src, obj)):
                self.reconstruct_json(os.path.join(n5src, obj))

    
    
    def save_chunk(
        self,
        source: zarr.Array, 
        dest: zarr.Array, 
        out_slices: Tuple[slice, ...],
        invert: bool):
    
        in_slices = tuple(out_slice for out_slice in out_slices)
        source_data = source[in_slices]
        # only store source_data if it is not all 0s
        if not (source_data == 0).all():
            if invert == True:
                dest[out_slices] = np.invert(source_data)
            else:
                dest[out_slices] = source_data
        return 1
        
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
        zarr_arrays = [n5_root.arrays(recurse=True)]

        # copy input n5 tree structure to output zarr and add ome-metadata, when N5 metadata is present
        z_store = zarr.NestedDirectoryStore(dest)
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
