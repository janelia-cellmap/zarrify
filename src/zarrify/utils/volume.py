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
        
    def reshape_to_arr_shape(self, param_arr, ref_arr):
        from itertools import cycle, islice
        return list(islice(cycle(param_arr), len(ref_arr)))

    def add_ome_metadata(self, dest: str, full_scale_group_name: str = 's0'):
        """Add selected tiff metadata to zarr attributes file (.zattrs).

        Args:
            dest (str): path to the output zarr
        """
        print(f"Adding OME-Zarr metadata to {dest}")
        print(f"Metadata axes: {self.metadata['axes']}")
        print(f"Metadata units: {self.metadata['units']}")
        print(f"Metadata scale: {self.metadata['scale']}")
        print(f"Metadata translation: {self.metadata['translation']}", flush=True)

        def get_axis(axis : str, unit : str) -> dict:
            if unit:
                return {"name": axis, "type": "space", "unit": unit}
            else:
                return {"name": axis, "type": "space"}

        root = zarr.open(dest, mode = 'a')
        # json template for a multiscale structure
        z_attrs: dict = {"multiscales": [{}]}
        z_attrs["multiscales"][0]["axes"] = [
            get_axis(axis, unit)
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
                "path": full_scale_group_name,
            }
        ]

        z_attrs["multiscales"][0]["name"] = "/" if root.path == "" else root.path
        z_attrs["multiscales"][0]["version"] = "0.4"

        # add multiscale template to .attrs
        root.attrs["multiscales"] = z_attrs["multiscales"]
