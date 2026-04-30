from pathlib import Path
from typing import Any
import copy

import numpy as np
import scipy.io as sciIO
from scipy import sparse


SSM_DIR = Path(__file__).resolve().parent
ROM_DIR = SSM_DIR / "ROM8_Expe"
TRAJ_INFO_DIR = SSM_DIR.parent / "trajinfom"
EIGHT_TRAJ_DIR = TRAJ_INFO_DIR / "eight"
SPIRAL_TRAJ_DIR = TRAJ_INFO_DIR / "spiral"


def loadmat(path: str | Path) -> dict[str, Any]:
    return sciIO.loadmat(str(path), squeeze_me=True, struct_as_record=False)


def get_field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return to_dense(obj[name])
    return to_dense(getattr(obj, name))


def to_dense(value: Any) -> Any:
    if sparse.issparse(value):
        return value.toarray()
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        return to_dense(value.item())
    return value


def as_column(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1, 1)


def as_vector(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def matlab_struct_array(value: Any) -> list[Any]:
    arr = np.asarray(value, dtype=object)
    if arr.shape == ():
        return [arr.item()]
    return [item for item in arr.reshape(-1)]


def clone_struct_array(value: Any) -> list[Any]:
    return copy.deepcopy(matlab_struct_array(value))


def subset_mapping_rows(w_map: Any, rows_1based: Any) -> list[Any]:
    rows = np.asarray(rows_1based, dtype=int).reshape(-1) - 1
    subset = clone_struct_array(w_map)
    for item in subset:
        coeffs = np.asarray(to_dense(get_field(item, "coeffs")))
        item.coeffs = coeffs[rows, :]
    return subset


def load_rom_data(rom_dir: str | Path = ROM_DIR, traj_dir: str | Path | None = None) -> dict[str, Any]:
    rom_dir = Path(rom_dir)
    traj_dir = rom_dir if traj_dir is None else Path(traj_dir)
    return {
        "auData": loadmat(rom_dir / "auData.mat")["auData"],
        "W_0": loadmat(rom_dir / "W_0.mat")["W_0"],
        "Node_List": loadmat(rom_dir / "Node_List.mat")["Node_List"],
        "zd_proj": np.asarray(loadmat(traj_dir / "zd_proj.mat")["zd_proj"], dtype=float),
        "t_list": np.asarray(loadmat(traj_dir / "t_list.mat")["t_list"], dtype=float).reshape(-1),
        "outdof_endeff": np.asarray(loadmat(rom_dir / "outdof_endeff.mat")["outdof_endeff"]),
        "outdof_half": np.asarray(loadmat(rom_dir / "outdof_half.mat")["outdof_half"]),
    }
