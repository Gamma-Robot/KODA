"""Batched TorchScript policy session with named tensor I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sil_trackers.contracts import NamedTensors, TensorSpec
from sil_trackers.registry import TrackerInstanceConfig


class BatchedTorchScriptPolicy:
    """One CUDA (or CPU) TorchScript module consuming/returning named tensors."""

    def __init__(
        self,
        model_path: str | Path,
        config: TrackerInstanceConfig,
        *,
        device_id: int = 0,
    ) -> None:
        if device_id < 0:
            raise ValueError("device_id must be non-negative")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "TorchScript policies require torch; install sil-trackers[scalebfm]"
            ) from exc

        self._torch = torch
        self._input_specs = {spec.name: spec for spec in config.policy_inputs}
        self._output_specs = {spec.name: spec for spec in config.policy_outputs}
        self._input_names = tuple(spec.name for spec in config.policy_inputs)
        self._output_names = tuple(spec.name for spec in config.policy_outputs)
        self._device_id = int(device_id)
        if torch.cuda.is_available():
            self._device = torch.device(f"cuda:{self._device_id}")
        else:
            self._device = torch.device("cpu")
        self._module = torch.jit.load(str(model_path), map_location=self._device)
        self._module.eval()

    @property
    def input_names(self) -> tuple[str, ...]:
        return self._input_names

    @property
    def output_names(self) -> tuple[str, ...]:
        return self._output_names

    @property
    def device_id(self) -> int:
        return self._device_id

    def run(self, inputs: NamedTensors) -> dict[str, np.ndarray]:
        """Run one forward pass and copy named outputs back to host memory."""

        batch = self._validate_inputs(inputs)
        tensors = [
            self._as_torch(name, inputs[name]) for name in self._input_names
        ]
        with self._torch.inference_mode():
            raw = self._module(*tensors)
        arrays = self._unpack_outputs(raw)
        result: dict[str, np.ndarray] = {}
        for name in self._output_names:
            array = np.asarray(arrays[name])
            self._validate_output_array(name, array, batch)
            result[name] = np.ascontiguousarray(array)
        return result

    def _as_torch(self, name: str, array: np.ndarray) -> Any:
        spec = self._input_specs[name]
        tensor = self._torch.from_numpy(np.ascontiguousarray(array))
        expected = np.dtype(spec.dtype)
        if expected == np.float32:
            tensor = tensor.to(dtype=self._torch.float32)
        elif expected == np.float64:
            tensor = tensor.to(dtype=self._torch.float64)
        elif expected == np.int64:
            tensor = tensor.to(dtype=self._torch.int64)
        elif expected == np.int32:
            tensor = tensor.to(dtype=self._torch.int32)
        return tensor.to(self._device)

    def _unpack_outputs(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, (tuple, list)):
            values = tuple(raw)
            if (
                len(values) == 2
                and self._output_names == ("action",)
            ):
                return {"action": values[1].detach().cpu().numpy()}
            if len(values) != len(self._output_names):
                raise ValueError(
                    "TorchScript output arity mismatch: "
                    f"got {len(values)} tensors, expected {len(self._output_names)}"
                )
            return {
                name: value.detach().cpu().numpy()
                for name, value in zip(self._output_names, values)
            }
        if len(self._output_names) != 1:
            raise ValueError("TorchScript returned one tensor for multiple outputs")
        return {self._output_names[0]: raw.detach().cpu().numpy()}

    def _validate_inputs(self, inputs: NamedTensors) -> int:
        if set(inputs) != set(self._input_names):
            raise ValueError(
                "policy input names mismatch: "
                f"got={sorted(inputs)} expected={sorted(self._input_names)}"
            )
        batch: int | None = None
        for name in self._input_names:
            array = inputs[name]
            spec = self._input_specs[name]
            self._validate_array(name, array, spec)
            if batch is None:
                batch = int(array.shape[0])
            elif int(array.shape[0]) != batch:
                raise ValueError(
                    f"policy inputs disagree on batch size: {name} has "
                    f"{array.shape[0]}, expected {batch}"
                )
        assert batch is not None
        return batch

    def _validate_output_array(
        self, name: str, array: np.ndarray, batch: int
    ) -> None:
        spec = self._output_specs[name]
        self._validate_array(name, array, spec)
        if int(array.shape[0]) != batch:
            raise ValueError(
                f"policy output {name!r} batch {array.shape[0]} != input batch {batch}"
            )

    @staticmethod
    def _validate_array(name: str, array: np.ndarray, spec: TensorSpec) -> None:
        if not isinstance(array, np.ndarray):
            raise TypeError(f"tensor {name!r} must be a numpy array")
        expected_dtype = np.dtype(spec.dtype)
        if array.dtype != expected_dtype:
            raise TypeError(
                f"tensor {name!r} dtype {array.dtype} != expected {expected_dtype}"
            )
        expected_shape = (array.shape[0],) + spec.non_batch_shape
        if array.shape != expected_shape:
            raise ValueError(
                f"tensor {name!r} shape {array.shape} != expected {expected_shape}"
            )
        if array.shape[0] < 1:
            raise ValueError(f"tensor {name!r} batch must be non-empty")
