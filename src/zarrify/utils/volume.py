import zarr
from abc import ABCMeta


class Volume:

    def __init__(
        self,
        src_path: str,
        axes: list[str],
        scale: list[float],
        translation: list[float],
        units: list[str],
    ):
        self.src_path = src_path
        self.metadata = {
            "axes": axes,
            "translation": translation,
            "scale": scale,
            "units": units,
        }
        
    def get_output_array(self, dest: str, chunks: list[int], comp: ABCMeta) -> zarr.Array:
        z_store = zarr.NestedDirectoryStore(dest)
        z_root = zarr.open(store=z_store, mode="a")
        
        return z_root.require_dataset(
            name="s0",
            shape=self.shape,
            dtype=self.dtype,
            chunks=chunks,
            compressor=comp,
        )
        
    def reshape_to_arr_shape(self, param_arr, ref_arr):
        from itertools import cycle, islice
        return list(islice(cycle(param_arr), len(ref_arr)))

    def add_ome_metadata(self, dest: str):
        """Add selected tiff metadata to zarr attributes file (.zattrs).

        Args:
            dest (str): path to the output zarr
        """
        print(self.metadata['axes'])
        root = zarr.open(dest, mode = 'a')
        # json template for a multiscale structure
        z_attrs: dict = {"multiscales": [{}]}
        z_attrs["multiscales"][0]["axes"] = [
            {"name": axis, "type": "space", "unit": unit}
            for axis, unit in zip(list(self.metadata["axes"]), self.metadata["units"])
        ]
        z_attrs["multiscales"][0]["coordinateTransformations"] = [
            {"scale": [1.0]*len(self.metadata['axes']), "type": "scale"}
        ]
        z_attrs["multiscales"][0]["datasets"] = [
            {
                "coordinateTransformations": [
                    {"scale": self.metadata["scale"], "type": "scale"},
                    {
                        "translation": self.metadata["translation"],
                        "type": "translation",
                    },
                ],
                "path": list(root.array_keys())[0],
            }
        ]

        z_attrs["multiscales"][0]["name"] = "/" if root.path == "" else root.path
        z_attrs["multiscales"][0]["version"] = "0.4"

        # add multiscale template to .attrs
        root.attrs["multiscales"] = z_attrs["multiscales"]
