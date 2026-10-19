"""Import and fuse the official low-latency SONIC deployment graphs."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import yaml

from sil_tts.artifacts import default_repo_root, load_lock, resolve, sha256_file, verify

ENCODER_ID = "sonic-low-latency-g1-encoder"
DECODER_ID = "sonic-low-latency-g1-decoder"
OBSERVATION_CONFIG_ID = "sonic-low-latency-observation-config"
POLICY_ID = "sonic-low-latency-g1-policy.batched"


def _verify_external(path: Path, artifact_id: str, entries) -> None:
    entry = resolve(artifact_id, lock=entries)
    if not path.is_file():
        raise FileNotFoundError(f"artifact {artifact_id!r} missing at {path}")
    if path.stat().st_size != entry.bytes:
        raise ValueError(f"artifact {artifact_id!r} size mismatch")
    if sha256_file(path) != entry.sha256:
        raise ValueError(f"artifact {artifact_id!r} sha256 mismatch")


def _validate_observation_config(path: Path) -> None:
    payload = yaml.safe_load(path.read_text())
    try:
        encoder = payload["encoder"]
        modes = encoder["encoder_modes"]
        g1 = next(mode for mode in modes if mode["name"] == "g1")
    except (KeyError, StopIteration, TypeError) as exc:
        raise ValueError("invalid low-latency SONIC observation config") from exc
    if encoder.get("dimension") != 64:
        raise ValueError("low-latency SONIC encoder dimension must be 64")
    required = list(g1.get("required_observations", []))
    expected = [
        "encoder_mode_4",
        "motion_joint_positions_10frame_step1",
        "motion_joint_velocities_10frame_step1",
        "motion_anchor_orientation_10frame_step1",
    ]
    if required != expected:
        raise ValueError(f"low-latency SONIC G1 observations must be {expected}")


def _set_batch(value_info) -> None:
    dims = value_info.type.tensor_type.shape.dim
    if not dims:
        raise ValueError(f"ONNX value {value_info.name!r} has no batch dimension")
    dims[0].ClearField("dim_value")
    dims[0].dim_param = "batch"


def _promote_encoder_reshape_batch(model) -> None:
    import numpy as np
    from onnx import numpy_helper

    expected = {
        "/Reshape_1": [1, 1, 10, 58],
        "/Reshape_2": [1, 1, 6],
        "/Reshape_3": [1, 1, 10, 6],
        "/Reshape_4": [1, 1, 240],
        "/Reshape_5": [1, 1, 9],
        "/Reshape_6": [1, 1, 12],
        "/Reshape_7": [1, 1, 4, 72],
        "/Reshape_8": [1, 1, 4, 6],
        "/Reshape_9": [1, 1, 4, 6],
        "/g1/Reshape": [1, 640],
        "/g1/Reshape_1": [1, 2, 32],
        "/quantizer/Reshape": [1, 2, 1, 32],
        "/quantizer/Reshape_1": [1, 2, 32],
    }
    nodes = {node.name: node for node in model.graph.node}
    constants = {
        node.output[0]: node
        for node in model.graph.node
        if node.op_type == "Constant" and len(node.output) == 1
    }
    for node_name, shape in expected.items():
        node = nodes.get(node_name)
        if node is None or node.op_type != "Reshape" or len(node.input) != 2:
            raise ValueError(f"low-latency SONIC encoder is missing {node_name!r}")
        constant = constants.get(node.input[1])
        if constant is None:
            raise ValueError(f"low-latency SONIC {node_name!r} shape is not constant")
        attribute = next((item for item in constant.attribute if item.name == "value"), None)
        if attribute is None:
            raise ValueError(f"low-latency SONIC {node_name!r} shape has no value")
        actual = numpy_helper.to_array(attribute.t)
        if actual.tolist() != shape:
            raise ValueError(
                f"low-latency SONIC {node_name!r} shape changed: {actual.tolist()}"
            )
        replacement = np.asarray([-1, *shape[1:]], dtype=actual.dtype)
        attribute.t.CopyFrom(numpy_helper.from_array(replacement, attribute.t.name))


def fuse_low_latency_policy(encoder_path: Path, decoder_path: Path):
    import onnx
    from onnx import compose, helper

    encoder = onnx.load(str(encoder_path), load_external_data=True)
    decoder = onnx.load(str(decoder_path), load_external_data=True)
    onnx.checker.check_model(encoder)
    onnx.checker.check_model(decoder)
    if [(value.name, tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)) for value in encoder.graph.input] != [("obs_dict", (1, 1247))]:
        raise ValueError("unexpected low-latency SONIC encoder input ABI")
    if [value.name for value in encoder.graph.output] != ["encoded_tokens"]:
        raise ValueError("unexpected low-latency SONIC encoder output ABI")
    if [(value.name, tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)) for value in decoder.graph.input] != [("obs_dict", (1, 994))]:
        raise ValueError("unexpected low-latency SONIC decoder input ABI")
    if [value.name for value in decoder.graph.output] != ["action"]:
        raise ValueError("unexpected low-latency SONIC decoder output ABI")
    _promote_encoder_reshape_batch(encoder)
    # The exported multi-mode graph selects G1/teleop/SMPL tokens with a
    # fixed [1, 3] mask. This tracker only supports official G1 mode, so take
    # the G1 quantizer output directly. It is identical for batch=1 and keeps
    # every candidate lane independent for dynamic batches.
    g1_token = next(
        (node.output[0] for node in encoder.graph.node if node.name == "/quantizer/Cast"),
        None,
    )
    if g1_token is None:
        raise ValueError("low-latency SONIC encoder is missing G1 token output")
    original_output = encoder.graph.output[0]
    original_output.name = g1_token
    original_output.type.tensor_type.shape.dim[0].dim_value = 1
    original_output.type.tensor_type.shape.dim[1].dim_value = 2
    original_output.type.tensor_type.shape.dim.add().dim_value = 32
    encoder = onnx.utils.Extractor(encoder).extract_model(
        [encoder.graph.input[0].name], [g1_token]
    )
    # The exporter left fixed batch=1 annotations on intermediates. ORT may
    # use those stale hints during graph optimization, so discard them after
    # promoting the actual reshape operations.
    del encoder.graph.value_info[:]

    encoder = compose.add_prefix(encoder, "encoder/")
    decoder = compose.add_prefix(decoder, "decoder/")
    decoder_input = decoder.graph.input[0].name
    for node in decoder.graph.node:
        for index, name in enumerate(node.input):
            if name == decoder_input:
                node.input[index] = "decoder_obs"
    del decoder.graph.input[:]

    encoder_input = encoder.graph.input[0]
    previous_encoder_input = encoder_input.name
    encoder_input.name = "encoder_obs"
    for node in encoder.graph.node:
        for index, name in enumerate(node.input):
            if name == previous_encoder_input:
                node.input[index] = "encoder_obs"
    _set_batch(encoder_input)
    token_output = encoder.graph.output[0]
    _set_batch(token_output)
    token_name = token_output.name
    del encoder.graph.output[:]

    proprioception = helper.make_tensor_value_info(
        "proprioception", onnx.TensorProto.FLOAT, ["batch", 930]
    )
    token_shape = helper.make_tensor(
        "encoder/g1_token_shape", onnx.TensorProto.INT64, [2], [-1, 64]
    )
    flatten = helper.make_node(
        "Reshape", [token_name, token_shape.name], ["motion_token"],
        name="fused/flatten_g1_token"
    )
    concat = helper.make_node(
        "Concat", ["motion_token", "proprioception"], ["decoder_obs"],
        axis=1, name="fused/concat_token_proprioception"
    )
    action = decoder.graph.output[0]
    previous_action = action.name
    action.name = "action"
    for node in decoder.graph.node:
        for index, name in enumerate(node.output):
            if name == previous_action:
                node.output[index] = "action"
    _set_batch(action)
    motion_token = helper.make_tensor_value_info(
        "motion_token", onnx.TensorProto.FLOAT, ["batch", 64]
    )

    graph = helper.make_graph(
        [*encoder.graph.node, flatten, concat, *decoder.graph.node],
        "sonic_low_latency_g1_fused",
        [encoder_input, proprioception],
        [action, motion_token],
        initializer=[*encoder.graph.initializer, token_shape, *decoder.graph.initializer],
        value_info=[*encoder.graph.value_info, *decoder.graph.value_info],
    )
    model = helper.make_model(
        graph,
        producer_name="sil-tts",
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=max(encoder.ir_version, decoder.ir_version),
    )
    onnx.checker.check_model(model)
    return model


def prepare_sonic_low_latency(
    encoder: str | Path,
    decoder: str | Path,
    observation_config: str | Path,
    *,
    root: str | Path | None = None,
) -> Path:
    import onnx

    repo = default_repo_root() if root is None else Path(root).resolve()
    entries = load_lock(repo / "artifacts/lock.v1.json")
    sources = (
        (Path(encoder).resolve(), ENCODER_ID),
        (Path(decoder).resolve(), DECODER_ID),
        (Path(observation_config).resolve(), OBSERVATION_CONFIG_ID),
    )
    for path, artifact_id in sources:
        _verify_external(path, artifact_id, entries)
    _validate_observation_config(sources[2][0])

    for source, artifact_id in sources:
        target_entry = resolve(artifact_id, lock=entries)
        target = (repo / str(target_entry.path)).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
            candidate = Path(temporary) / target.name
            shutil.copy2(source, candidate)
            os.replace(candidate, target)
        verify(target_entry, root=repo)

    policy_entry = resolve(POLICY_ID, lock=entries)
    output = (repo / str(policy_entry.path)).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model = fuse_low_latency_policy(sources[0][0], sources[1][0])
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        candidate = Path(temporary) / output.name
        onnx.save(model, str(candidate))
        if candidate.stat().st_size != policy_entry.bytes:
            raise ValueError("derived low-latency SONIC policy size mismatch")
        if sha256_file(candidate) != policy_entry.sha256:
            raise ValueError("derived low-latency SONIC policy sha256 mismatch")
        os.replace(candidate, output)
    return verify(policy_entry, root=repo)


__all__ = ["fuse_low_latency_policy", "prepare_sonic_low_latency"]
