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
        else:
            # N5 arrays store chunks in subdirectories — don't recurse into them.
            try:
                with open(os.path.join(n5src, "attributes.json")) as f:
                    attrs = json.load(f)
                if 'dataType' in attrs and 'dimensions' in attrs:
                    return
            except (json.JSONDecodeError, OSError):
                pass
        for obj in dir_list:
            if os.path.isdir(os.path.join(n5src, obj)):
                self.reconstruct_json(os.path.join(n5src, obj))

    def _find_n5_arrays(self, root_path: str) -> list[str]:
        """Walk the N5 directory and return store-relative paths of all arrays.

        An N5 array is identified by an attributes.json that contains both
        "dataType" and "dimensions" keys.

        Parameters
        ----------
        root_path:
            Absolute path to the N5 store root or sub-group to search.

        Returns
        -------
        list[str]
            Paths relative to self.store_path, suitable for passing to
            n5_spec() and zarr3_spec().
        """
        array_paths = []
        for dirpath, _, filenames in os.walk(root_path):
            if 'attributes.json' not in filenames:
                continue
            with open(os.path.join(dirpath, 'attributes.json')) as f:
                try:
                    attrs = json.load(f)
                except json.JSONDecodeError:
                    continue
            # N5 arrays carry 'dataType' and 'dimensions' in their attributes
            if 'dataType' in attrs and 'dimensions' in attrs:
                array_paths.append(os.path.relpath(dirpath, self.store_path))
        return array_paths

    def _copy_n5_group_attrs(self, root_path: str, z_root: zarr.Group) -> None:
        """Copy N5 group attributes to the output zarr3 store.

        Only group-level attributes.json files are copied (array nodes, which
        carry "dataType"/"dimensions", are skipped). Group attributes are needed
        by normalize_to_omengff to build OME-NGFF multiscales metadata.

        Parameters
        ----------
        root_path:
            Absolute path to the N5 store root.
        z_root:
            Opened zarr3 group at the output store root.
        """
        for dirpath, _, filenames in os.walk(root_path):
            if 'attributes.json' not in filenames:
                continue
            with open(os.path.join(dirpath, 'attributes.json')) as f:
                try:
                    attrs = json.load(f)
                except json.JSONDecodeError:
                    continue
            if 'dataType' in attrs and 'dimensions' in attrs:
                continue
            rel = os.path.relpath(dirpath, root_path)
            target = z_root if rel == '.' else z_root.require_group(rel)
            for k, v in attrs.items():
                target.attrs[k] = v

    def _copy_n5_array_attrs(self, dest: str, array_paths: list[str]) -> None:
        """Copy N5 array attributes (e.g. transform) to the output zarr3 arrays.

        Called after TensorStore has created the output zarr3 arrays so that
        normalize_to_omengff can read coordinate transform metadata.

        Parameters
        ----------
        dest:
            Absolute path to the output zarr3 store root.
        array_paths:
            Store-relative paths of arrays whose attributes should be copied.
        """
        z_store = zarr.storage.LocalStore(dest)
        for rel_path in array_paths:
            attrs_file = os.path.join(self.store_path, rel_path, 'attributes.json')
            if not os.path.exists(attrs_file):
                continue
            with open(attrs_file) as f:
                try:
                    attrs = json.load(f)
                except json.JSONDecodeError:
                    continue
            z_arr = zarr.open_array(store=z_store, path=rel_path, mode='a')
            for k, v in attrs.items():
                z_arr.attrs[k] = v

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
        units_list = [str(ureg.Unit(unit)) for unit in zgroup.attrs['units']]

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
            if isinstance(zgroup[key], zarr.Group):
                if key!='/':
                    self.normalize_to_omengff(zgroup[key])
                if 'scales' in zgroup[key].attrs.asdict():
                    zattrs = self.apply_ome_template(zgroup[key])
                    unsorted_datasets = []
                    for arr in self._iter_arrays(zgroup[key]):
                        unsorted_datasets.append(self.ome_dataset_metadata(arr[1], zgroup[key]))

                    #1.apply natural sort to organize datasets metadata array for different resolution degrees (s0 -> s10)
                    #2.add datasets metadata to the omengff template
                    zattrs['multiscales'][0]['datasets'] = natsort.natsorted(unsorted_datasets, key=itemgetter(*['path']))
                    zgroup[key].attrs['multiscales'] = zattrs['multiscales']

    @staticmethod
    def _iter_arrays(group: zarr.Group):
        """Yield (name, Array) pairs recursively from *group* (zarr v3 compat)."""
        for name, node in group.members():
            if isinstance(node, zarr.Array):
                yield name, node
            elif isinstance(node, zarr.Group):
                yield from N5Group._iter_arrays(node)

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

    def write_to_zarr(
        self,
        dest: str,
        client: Client,
        chunk_shape: list[int],
        shard_shape: list[int] | None = None,
        codec: dict = zstd_codec(level=6),
    ) -> None:
        """Copy all N5 arrays into a zarr3 store via TensorStore.

        The N5 hierarchy is traversed with os.walk to discover arrays; group
        and array attributes are copied directly from attributes.json files,
        preserving all N5 metadata without requiring zarr's N5 driver. Data
        is copied chunk-by-chunk using the TensorStore N5 driver for reads
        and the zarr3 driver for writes. All-zero chunks are skipped.

        Parameters
        ----------
        dest:
            Absolute path to the output zarr store root directory.
        client:
            Dask distributed client used to parallelise chunk copies.
        chunk_shape:
            Inner chunk shape for the output zarr3 arrays.
        shard_shape:
            Outer shard shape. When provided, enables sharding.
        codec:
            Compression codec dict. Defaults to zstd_codec(level=6).
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        n5_root_path = os.path.join(self.store_path, self.path) if self.path else self.store_path
        self.reconstruct_json(n5_root_path)

        # input n5 arrays list to convert
        n5_array_paths = self._find_n5_arrays(n5_root_path)

        # copy input n5 tree structure to output zarr and add ome-metadata, when N5 metadata is present
        z_store = zarr.storage.LocalStore(dest)
        z_root = zarr.open_group(store=z_store, mode='a')
        self._copy_n5_group_attrs(n5_root_path, z_root)

        for rel_path in n5_array_paths:
            src_spec = n5_spec(self.store_path, rel_path)
            src_arr = open_ts(src_spec)
            shape = src_arr.shape
            dtype = np.dtype(src_arr.dtype.numpy_dtype)

            # trim chunk/shard shapes to array ndim; N5 trees can hold mixed-dimensionality arrays
            arr_chunk_shape = [min(c, s) for c, s in zip(list(chunk_shape)[-len(shape):], shape)]
            arr_shard_shape = (
                [min(s, dim) for s, dim in zip(list(shard_shape)[-len(shape):], shape)]
                if shard_shape is not None else None
            )

            dst_spec = zarr3_spec(
                store_path=dest,
                array_path=rel_path,
                shape=shape,
                dtype=dtype,
                chunk_shape=arr_chunk_shape,
                shard_shape=arr_shard_shape,
                codec=codec,
                create=True,
            )

            dest_arr = open_ts(dst_spec)
            dest_chunks = dest_arr.chunk_layout.write_chunk.shape

            out_slices = slices_from_chunks(
                normalize_chunks(dest_chunks, shape=shape)
            )
            # break the slices up into batches, to make things easier for the dask scheduler
            out_slices_partitioned = tuple(partition_all(100000, out_slices))

            for idx, part in enumerate(out_slices_partitioned):
                logging.info(f"{idx + 1} / {len(out_slices_partitioned)}")
                start = time.time()
                fut = client.map(
                    lambda v: save_chunk(src_spec, dst_spec, v, invert=False), part
                )
                logging.info(
                    f"Submitted {len(part)} tasks to the scheduler in {time.time() - start:.2f}s"
                )
                wait(fut)
                logging.info(
                    f"Completed {len(part)} tasks in {time.time() - start:.2f}s"
                )

        # copy array-level N5 attributes (e.g. transform) then build OME metadata
        self._copy_n5_array_attrs(dest, n5_array_paths)
        z_root = zarr.open_group(store=z_store, mode='a')
        self.normalize_to_omengff(z_root)


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
