import os
import sys
from pathlib import Path


def configure_sdk_path(base_dir: Path | None = None, bin_dir_name: str = "bin") -> Path:
    """Add the local SDK binary directory to Python and Windows DLL paths."""
    base_dir = Path(__file__).resolve().parent if base_dir is None else Path(base_dir)
    bin_dir = (base_dir / bin_dir_name).resolve()

    if not bin_dir.exists():
        raise FileNotFoundError(f"SDK binary directory does not exist: {bin_dir}")

    bin_dir_str = str(bin_dir)
    if bin_dir_str not in sys.path:
        sys.path.insert(0, bin_dir_str)

    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir_str not in path_parts:
        os.environ["PATH"] = bin_dir_str + os.pathsep + os.environ.get("PATH", "")

    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(bin_dir_str)

    return bin_dir


def load_lumo_sdk(base_dir: Path | None = None):
    """Configure paths and import the LuMo SDK extension."""
    configure_sdk_path(base_dir)
    import LuMoSDKPy311 as lumo_sdk

    return lumo_sdk
