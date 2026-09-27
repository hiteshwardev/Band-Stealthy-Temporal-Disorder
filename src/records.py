"""Result files that are identical from one run to the next.

`numpy.savez_compressed` stamps every archive member with the current time, so two runs
that compute identical arrays still write different files. `save_npz` writes the same
format with a fixed timestamp; `save_json` writes JSON with a fixed layout.
"""
import io
import json
import zipfile

import numpy as np

_EPOCH = (1980, 1, 1, 0, 0, 0)


def save_npz(path, **arrays):
    """Compressed .npz archive readable by numpy.load, with fixed member timestamps."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(value), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, buffer.getvalue())


def save_json(path, data):
    """JSON with one-space indentation; numpy scalars are written as floats."""
    with open(path, "w") as handle:
        json.dump(data, handle, indent=1, default=float)
        handle.write("\n")
