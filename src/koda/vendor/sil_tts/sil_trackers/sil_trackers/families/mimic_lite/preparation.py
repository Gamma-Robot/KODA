"""Offline batch transform for the Mimic-Lite 32x8192-huge policy."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from typing import Any

from sil_trackers.preparation import register_transform


_SOURCE_SHA256 = "1a61b8d1aad10ce962de1a3e7fa4f399fb6271d20c68a36afadc4cf45aaa7d1f"

_NODE_SIGNATURES = (
    ("node_sub_1", "Sub", ("policy", "div"), ("sub_1",)),
    ("node_div_2", "Div", ("sub_1", "sqrt"), ("div_2",)),
    ("node_sub_3", "Sub", ("command", "div_3"), ("sub_3",)),
    ("node_div_5", "Div", ("sub_3", "sqrt_1"), ("div_5",)),
    ("node_cat", "Concat", ("div_2", "div_5"), ("cat",)),
    ("node_MatMul_7", "MatMul", ("cat", "val_6"), ("val_7",)),
    (
        "node_linear",
        "Add",
        ("val_7", "module.2.module.1.module.0.bias"),
        ("linear",),
    ),
    (
        "node_layer_norm",
        "LayerNormalization",
        (
            "linear",
            "module.2.module.1.module.1.weight",
            "module.2.module.1.module.1.bias",
        ),
        ("layer_norm",),
    ),
    ("n0", "Mish", ("layer_norm",), ("mish",)),
    ("node_MatMul_9", "MatMul", ("mish", "val_10"), ("val_11",)),
    (
        "node_linear_1",
        "Add",
        ("val_11", "module.2.module.1.module.3.bias"),
        ("linear_1",),
    ),
    (
        "node_layer_norm_1",
        "LayerNormalization",
        (
            "linear_1",
            "module.2.module.1.module.4.weight",
            "module.2.module.1.module.4.bias",
        ),
        ("layer_norm_1",),
    ),
    ("n0_2", "Mish", ("layer_norm_1",), ("mish_1",)),
    ("node_MatMul_11", "MatMul", ("mish_1", "val_14"), ("val_15",)),
    (
        "node_linear_2",
        "Add",
        ("val_15", "module.2.module.1.module.6.bias"),
        ("linear_2",),
    ),
    (
        "node_layer_norm_2",
        "LayerNormalization",
        (
            "linear_2",
            "module.2.module.1.module.7.weight",
            "module.2.module.1.module.7.bias",
        ),
        ("layer_norm_2",),
    ),
    ("n0_3", "Mish", ("layer_norm_2",), ("mish_2",)),
    ("node_MatMul_13", "MatMul", ("mish_2", "val_18"), ("val_19",)),
    (
        "node_linear_3",
        "Add",
        ("val_19", "module.2.module.2.module.actor_mean.bias"),
        ("action",),
    ),
)

_INITIALIZER_SIGNATURES = {
    "module.2.module.1.module.0.bias": (1, (1024,)),
    "module.2.module.1.module.1.weight": (1, (1024,)),
    "module.2.module.1.module.1.bias": (1, (1024,)),
    "module.2.module.1.module.3.bias": (1, (1024,)),
    "module.2.module.1.module.4.weight": (1, (1024,)),
    "module.2.module.1.module.4.bias": (1, (1024,)),
    "module.2.module.1.module.6.bias": (1, (1024,)),
    "module.2.module.1.module.7.weight": (1, (1024,)),
    "module.2.module.1.module.7.bias": (1, (1024,)),
    "module.2.module.2.module.actor_mean.bias": (1, (29,)),
    "div": (1, (535,)),
    "sqrt": (1, (535,)),
    "div_3": (1, (304,)),
    "sqrt_1": (1, (304,)),
    "val_6": (1, (839, 1024)),
    "val_10": (1, (1024, 1024)),
    "val_14": (1, (1024, 1024)),
    "val_18": (1, (1024, 29)),
}


def _shape(value_info: Any) -> list[int | str]:
    return [
        dim.dim_value if dim.HasField("dim_value") else dim.dim_param
        for dim in value_info.type.tensor_type.shape.dim
    ]


def _set_shape(value_info: Any, shape: Sequence[int | str]) -> None:
    import onnx

    tensor = value_info.type.tensor_type
    tensor.elem_type = onnx.TensorProto.FLOAT
    del tensor.shape.dim[:]
    for item in shape:
        dim = tensor.shape.dim.add()
        if isinstance(item, int):
            dim.dim_value = item
        else:
            dim.dim_param = item


def _validate_source(model: Any) -> None:
    import onnx

    if (
        model.ir_version != 10
        or model.producer_name != "pytorch"
        or model.producer_version != "2.10.0+cu128"
        or model.graph.name != "main_graph"
        or [(item.domain, item.version) for item in model.opset_import]
        != [("", 20)]
    ):
        raise ValueError("Mimic-Lite 32x8192-huge source identity changed")

    inputs = {value.name: value for value in model.graph.input}
    outputs = {value.name: value for value in model.graph.output}
    expected_inputs = {"policy": [535], "command": [304]}
    if (
        set(inputs) != set(expected_inputs)
        or any(
            value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
            or _shape(value) != expected_inputs[name]
            for name, value in inputs.items()
        )
        or set(outputs) != {"action"}
        or outputs["action"].type.tensor_type.elem_type
        != onnx.TensorProto.FLOAT
        or _shape(outputs["action"]) != [29]
    ):
        raise ValueError("Mimic-Lite 32x8192-huge policy ABI changed")

    actual_nodes = tuple(
        (node.name, node.op_type, tuple(node.input), tuple(node.output))
        for node in model.graph.node
    )
    actual_initializers = {
        value.name: (value.data_type, tuple(value.dims))
        for value in model.graph.initializer
    }
    if (
        actual_nodes != _NODE_SIGNATURES
        or actual_initializers != _INITIALIZER_SIGNATURES
        or len(model.graph.value_info) != 36
    ):
        raise ValueError("Mimic-Lite 32x8192-huge graph structure changed")

    digest = sha256(model.SerializeToString()).hexdigest()
    if digest != _SOURCE_SHA256:
        raise ValueError(
            "Mimic-Lite 32x8192-huge source identity changed: "
            f"expected sha256={_SOURCE_SHA256}, actual={digest}"
        )


@register_transform("mimic_lite_32x8192_huge_batch", 1)
def mimic_lite_32x8192_huge_batch(model: Any) -> None:
    """Promote the pinned rank-one Mimic-Lite graph to symbolic batch."""
    import onnx

    _validate_source(model)
    inputs = {value.name: value for value in model.graph.input}
    outputs = {value.name: value for value in model.graph.output}
    _set_shape(inputs["policy"], ["batch", 535])
    _set_shape(inputs["command"], ["batch", 304])
    _set_shape(outputs["action"], ["batch", 29])
    del model.graph.value_info[:]
    onnx.checker.check_model(model)


__all__ = ["mimic_lite_32x8192_huge_batch"]
