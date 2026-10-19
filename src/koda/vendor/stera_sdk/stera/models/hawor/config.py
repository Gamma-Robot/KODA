"""Configuration for the external ThunderVVV/HaWoR model."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class HaWoRConfig:
    """Paths are absolute or relative to ``hawor_dir``.

    The checkpoint, config, detector and licensed MANO_RIGHT model must already
    be installed using HaWoR's setup instructions. The backend does not download
    weights. ``device`` is a logical PyTorch device inside the caller's existing
    CUDA_VISIBLE_DEVICES allocation. ``work_dir`` optionally selects the parent
    directory for temporary lossless video frames, which are always cleaned up.
    """

    hawor_dir: Optional[str] = None
    checkpoint: Optional[str] = None
    model_config: Optional[str] = None
    hand_detector_path: Optional[str] = None
    yolo_conf: float = 0.2
    device: str = "cuda"
    save_mano_vertices: bool = True
    work_dir: Optional[str] = None
