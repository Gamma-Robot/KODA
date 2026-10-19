#!/usr/bin/env python3
"""EfficientSAM3 per-frame batched masks.

Writes packed ``masks/frames.npy`` plus ``mask_metadata.json``. TinyViT runs
once per image batch, then each text prompt gets its own grounding pass.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_ROOT = PROJECT_ROOT / "resources"
ASSET_ROOT = Path(__import__("os").environ.get("KODA_ASSETS", str(PROJECT_ROOT / "assets"))).expanduser().resolve()
DEFAULT_EFF_ROOT = RESOURCE_ROOT / "efficientsam3/sam3"
DEFAULT_EFF_CKPT = ASSET_ROOT / "models/efficientsam3/efficientsam3_tinyvit.pt"
DEFAULT_PROMPT = "human arms and hands"


def _ensure_eff_path(eff_root: Path) -> None:
    root = str(Path(eff_root).resolve())
    # Prefer EfficientSAM3 package over facebook sam3_repo.
    sys.path = [p for p in sys.path if "sam3_repo" not in p]
    if root not in sys.path:
        sys.path.insert(0, root)
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)


def load_efficientsam3_image_model(
    *,
    eff_root: Path = DEFAULT_EFF_ROOT,
    eff_ckpt: Path = DEFAULT_EFF_CKPT,
    device: str = "cuda",
    conf_threshold: float = 0.35,
    text_encoder_type: str = "MobileCLIP-S0",
    text_encoder_context_length: int = 16,
    backbone_type: str = "tinyvit",
    model_name: str = "11m",
):
    _ensure_eff_path(eff_root)
    from sam3.model_builder import build_efficientsam3_image_model

    bpe = Path(eff_root) / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    model = build_efficientsam3_image_model(
        checkpoint_path=str(eff_ckpt),
        backbone_type=backbone_type,
        model_name=model_name,
        text_encoder_type=text_encoder_type,
        text_encoder_context_length=int(text_encoder_context_length),
        load_from_HF=False,
        bpe_path=str(bpe) if bpe.is_file() else None,
        device=device,
    )
    model = model.float().eval()
    model._hdmi_conf = float(conf_threshold)  # type: ignore[attr-defined]
    return model


def _make_transform(resolution: int = 1008):
    return v2.Compose(
        [
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(resolution, resolution)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _clone_vision_backbone(backbone_out: dict) -> dict:
    cloned: dict = {}
    for key, value in backbone_out.items():
        if torch.is_tensor(value):
            cloned[key] = value
        elif isinstance(value, list):
            cloned[key] = list(value)
        elif isinstance(value, tuple):
            cloned[key] = tuple(value)
        elif isinstance(value, dict):
            cloned[key] = dict(value)
        else:
            cloned[key] = value
    return cloned


@torch.inference_mode()
def _forward_batch(model, images_b: torch.Tensor, prompt: str, FindStage, conf: float):
    results = _forward_prompts(model, images_b, [prompt], FindStage, [float(conf)])
    return results[prompt]


@torch.inference_mode()
def _forward_prompts(model, images_b: torch.Tensor, prompts: list[str], FindStage, confs: list[float]):
    """TinyViT once, then text + grounding per prompt."""
    b = images_b.shape[0]
    device = images_b.device
    vision = model.backbone.forward_image(images_b)
    out_by_prompt: dict[str, list] = {}
    for prompt, conf in zip(prompts, confs):
        backbone_out = _clone_vision_backbone(vision)
        text_outputs = model.backbone.forward_text([prompt], device=device)
        backbone_out.update(text_outputs)
        find_input = FindStage(
            img_ids=torch.arange(b, device=device, dtype=torch.long),
            text_ids=torch.zeros(b, device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        geometric_prompt = model._get_dummy_prompt(num_prompts=b)
        out = model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )
        logits = out["pred_logits"].sigmoid()
        if logits.ndim == 3:
            logits = logits.squeeze(-1)
        presence = out["presence_logit_dec"].sigmoid()
        if presence.ndim == 1:
            presence = presence.unsqueeze(1)
        probs = logits * presence
        masks = out["pred_masks"]
        unions = []
        for i in range(b):
            keep = probs[i] > float(conf)
            if not bool(keep.any()):
                unions.append(None)
                continue
            m = masks[i, keep].sigmoid()
            u = (m.max(dim=0).values > 0.5).to(torch.uint8).cpu().numpy()
            unions.append(u)
        out_by_prompt[prompt] = unions
    return out_by_prompt


def run_episode_efficientsam3(
    model,
    *,
    video: Path,
    episode_dir: Path | None = None,
    start_frame: int,
    max_frames: int,
    prompts: list[str] | None = None,
    jobs: list[tuple[str, Path]] | None = None,
    packed_rgb: Path | None = None,
    close_px: int = 5,
    dilation_px: int = 4,
    temporal_radius: int = 1,
    conf_threshold: float = 0.35,
    batch_size: int = 16,
    resolution: int = 1008,
    write_diag: bool = False,
    gpu_lock=None,
    eff_ckpt: Path | None = None,
) -> dict:
    """Run EfficientSAM3 batched image PCS; write packed ``masks/frames.npy``."""
    from gpu_mask_postprocess import postprocess_masks_gpu
    from sam3.model.data_misc import FindStage

    try:
        from frame_volume import as_rgb, create_volume, is_packed_stream, open_volume
    except ImportError:
        from .frame_volume import as_rgb, create_volume, is_packed_stream, open_volume

    if jobs is None:
        prompt_list = list(prompts or [DEFAULT_PROMPT])
        if episode_dir is None:
            raise ValueError("episode_dir is required when jobs is omitted")
        jobs = [(prompt, Path(episode_dir) / ("" if len(prompt_list) == 1 else prompt)) for prompt in prompt_list]
        if len(prompt_list) == 1:
            jobs = [(prompt_list[0], Path(episode_dir))]
    jobs = [(str(prompt), Path(directory)) for prompt, directory in jobs]
    prompt_list = [prompt for prompt, _ in jobs]
    conf = float(getattr(model, "_hdmi_conf", conf_threshold))
    confs = [conf] * len(prompt_list)
    tfm = _make_transform(resolution)

    packed_vol = None
    cap = None
    fps = 30.0
    if packed_rgb is not None and is_packed_stream(packed_rgb):
        packed_vol = open_volume(packed_rgb, color_space="rgb")
        height, width = (int(v) for v in packed_vol.frame_shape[:2])
        total = len(packed_vol)
        meta_fps = None
        try:
            import json as _json
            from frame_volume import meta_path as _meta_path
            payload = _json.loads(_meta_path(Path(packed_rgb)).read_text(encoding="utf-8"))
            meta_fps = payload.get("fps")
        except Exception:
            meta_fps = None
        fps = float(meta_fps or 30.0)
    else:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"open failed: {video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if start_frame > 0 and packed_vol is not None and start_frame >= total:
        raise RuntimeError(f"start_frame {start_frame} >= packed RGB length {total}")
    n = total - start_frame if max_frames <= 0 else min(max_frames, max(0, total - start_frame))

    raw_by_prompt: dict[str, list[np.ndarray]] = {prompt: [] for prompt, _ in jobs}
    buf_imgs: list = []
    t0 = time.time()

    def _lock():
        return gpu_lock() if gpu_lock is not None else nullcontext()

    def _append_unions(prompt: str, unions: list) -> None:
        bucket = raw_by_prompt[prompt]
        for u in unions:
            if u is None:
                bucket.append(np.zeros((height, width), dtype=np.uint8))
            else:
                if u.shape != (height, width):
                    u = cv2.resize(u, (width, height), interpolation=cv2.INTER_NEAREST)
                bucket.append(u.astype(np.uint8))

    def _flush():
        if not buf_imgs:
            return
        tensors = [tfm(v2.functional.to_image(im).to("cuda")) for im in buf_imgs]
        batch = torch.stack(tensors, dim=0)
        with _lock():
            unions_by_prompt = _forward_prompts(model, batch, prompt_list, FindStage, confs)
            torch.cuda.synchronize()
        for prompt in prompt_list:
            _append_unions(prompt, unions_by_prompt[prompt])
        buf_imgs.clear()

    for local in range(n):
        if packed_vol is not None:
            rgb = as_rgb(packed_vol.frame_at(start_frame + local, local))
            buf_imgs.append(Image.fromarray(rgb))
        else:
            assert cap is not None
            ok, bgr = cap.read()
            if not ok:
                break
            buf_imgs.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        if len(buf_imgs) >= batch_size:
            _flush()
    _flush()
    if cap is not None:
        cap.release()
    n = min(len(raw_by_prompt[prompt]) for prompt in prompt_list) if prompt_list else 0
    fwd_s = time.time() - t0

    torch.cuda.empty_cache()
    t1 = time.time()
    processed_by_prompt: dict[str, list[np.ndarray]] = {}
    with _lock():
        for prompt in prompt_list:
            processed_by_prompt[prompt] = postprocess_masks_gpu(
                [u * 255 for u in raw_by_prompt[prompt][:n]],
                close_px=close_px,
                dilation_px=dilation_px,
                temporal_radius=temporal_radius,
                device="cuda",
            )
        torch.cuda.synchronize()
    post_s = time.time() - t1

    records = []
    last_meta: dict | None = None
    for prompt, directory in jobs:
        directory.mkdir(parents=True, exist_ok=True)
        masks_dir = directory / "masks"
        processed = processed_by_prompt[prompt]
        vol = create_volume(masks_dir, n, (height, width), np.uint8, fill=0)
        for i, mask in enumerate(processed[:n]):
            vol[i] = np.asarray(mask, dtype=np.uint8)
        vol.flush()
        meta = {
            "mode": "efficientsam3_perframe_batch",
            "source_video": str(Path(video).resolve()),
            "packed_rgb": str(Path(packed_rgb).resolve()) if packed_rgb is not None else None,
            "start_frame": start_frame,
            "num_frames": n,
            "fps": fps,
            "width": width,
            "height": height,
            "prompts": [prompt],
            "conf_threshold": conf,
            "close_px": close_px,
            "dilation_px": dilation_px,
            "temporal_radius": temporal_radius,
            "batch_size": batch_size,
            "resolution": resolution,
            "eff_ckpt": str(eff_ckpt) if eff_ckpt else None,
            "forward_s": round(fwd_s, 2),
            "post_s": round(post_s, 2),
            "masks_dir": str(masks_dir),
            "layout": "packed_memmap",
            "postprocess": "cpu_threads_morph+gpu_temporal",
        }
        (directory / "mask_metadata.json").write_text(json.dumps(meta, indent=2))
        records.append(meta)
        last_meta = meta
        if write_diag:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            bin_path = directory / "sam3_mask.mp4"
            vw = cv2.VideoWriter(
                str(bin_path), fourcc, fps, (width // 2, height // 2), isColor=False
            )
            for m in processed[:n]:
                vw.write(
                    cv2.resize(m, (width // 2, height // 2), interpolation=cv2.INTER_NEAREST)
                )
            vw.release()
            meta["sam3_mask_mp4"] = str(bin_path)
            (directory / "mask_metadata.json").write_text(json.dumps(meta, indent=2))
    print(
        f"EfficientSAM3 done: frames={n} prompts={len(jobs)} fwd={fwd_s:.1f}s "
        f"post={post_s:.1f}s batch={batch_size}",
        flush=True,
    )
    if last_meta is None:
        raise RuntimeError("EfficientSAM3 produced no prompt outputs")
    last_meta = dict(last_meta)
    last_meta["records"] = records
    last_meta["prompts"] = prompt_list
    return last_meta
