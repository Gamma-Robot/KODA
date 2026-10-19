"""
Compatibility shim so `import sam3` resolves to the upstream package copied under
`sam3/sam3` in this repository layout.
"""

from importlib import import_module
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_INNER = _HERE / "sam3"

# Ensure Python searches the nested SAM3 directory for submodules such as
# `sam3.model`, `sam3.utils`, etc.
if _INNER.is_dir():
    __path__ = [str(_HERE), str(_INNER)]

_inner_pkg = import_module("sam3.sam3")
build_sam3_image_model = import_module("sam3.sam3.model_builder").build_sam3_image_model
build_efficientsam3_image_model = import_module("sam3.sam3.model_builder").build_efficientsam3_image_model

# Re-export everything the upstream package declared plus the builder helper.
__all__ = list(getattr(_inner_pkg, "__all__", []))
if "build_sam3_image_model" not in __all__:
    __all__.append("build_sam3_image_model")
if "build_efficientsam3_image_model" not in __all__:
    __all__.append("build_efficientsam3_image_model")

