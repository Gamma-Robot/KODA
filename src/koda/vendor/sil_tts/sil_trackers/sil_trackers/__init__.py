"""Pretrained-tracker semantics and inference contracts."""

from sil_trackers.adapter import TrackerAdapter
from sil_trackers.batched_tracker import (
    BatchedTracker,
    DeviceTrackerProvider,
    PreparedDeviceTracker,
    RobotStateView,
    SerialReferenceTracker,
)
from sil_trackers.contracts import (
    DevicePDCommand,
    MOTION_GUIDANCE_ARRAY_FIELDS,
    MotionGuidance,
    NamedTensors,
    PDCommand,
    RobotState,
    TensorSpec,
    TrackerStateBatch,
)
from sil_trackers.registry import (
    TrackerInstanceConfig,
    available,
    get_family,
    instance_config_from_mapping,
    load_instance_config,
    register,
)

__all__ = [
    "DevicePDCommand",
    "DeviceTrackerProvider",
    "MOTION_GUIDANCE_ARRAY_FIELDS",
    "MotionGuidance",
    "NamedTensors",
    "PDCommand",
    "RobotState",
    "RobotStateView",
    "BatchedTracker",
    "PreparedDeviceTracker",
    "SerialReferenceTracker",
    "TensorSpec",
    "TrackerAdapter",
    "TrackerInstanceConfig",
    "TrackerStateBatch",
    "available",
    "get_family",
    "instance_config_from_mapping",
    "load_instance_config",
    "register",
]

from sil_trackers import families as families  # noqa: E402,F401  (registers built-ins)
