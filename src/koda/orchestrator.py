from __future__ import annotations
import json, os, shutil, subprocess, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import yaml
from .paths import assets_root, cache_root, repo_root

@dataclass
class PipelineContext:
    input_path: Path
    output_root: Path
    config_path: Path
    config: dict[str, Any]
    frames: int | None = None
    gpu: str = "0"
    session_id: str | None = None

def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict): raise ValueError(f"config must be a mapping: {path}")
    return payload

def _session_id(path: Path) -> str:
    if path.is_file():
        name = path.parent.name
        return name if name.startswith("session_") else path.stem
    return path.name

def _find_mcap(path: Path) -> Path | None:
    if path.is_file() and path.suffix.lower() == ".mcap": return path
    if not path.is_dir(): return None
    candidates = sorted(path.glob("*.mcap")) + sorted(path.glob("**/*.mcap"))
    unique = list(dict.fromkeys(p.resolve() for p in candidates))
    if len(unique) == 1: return unique[0]
    if len(unique) > 1: raise ValueError(f"input contains multiple MCAP files; pass one file: {unique}")
    return None

def _pythonpath(root: Path) -> str:
    entries = [
        root / "src",
        root / "src/koda/vendor/stera_sdk",
        root / "src/koda/vendor/sil_tts/sil_tts",
        root / "src/koda/vendor/sil_tts/sil_trackers",
        root / "src/koda/vendor/sil_tts/sil_planners",
        root / "src/koda/vendor/hydrax",
    ]
    return os.pathsep.join(str(p) for p in entries if p.exists())

def _env(ctx: PipelineContext) -> dict[str, str]:
    env = os.environ.copy()
    env.update({"KODA_HOME": str(repo_root()), "KODA_PYTHON": sys.executable, "PYTHONUNBUFFERED": "1"})
    env["PYTHONPATH"] = _pythonpath(repo_root()) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("KODA_CACHE", str(cache_root()))
    env.setdefault("CUDA_VISIBLE_DEVICES", ctx.gpu)
    return env

def _run(ctx: PipelineContext, name: str, command: Sequence[str], *, cwd: Path | None = None, env_extra: dict[str, str] | None = None) -> None:
    log_dir = ctx.output_root / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    env = _env(ctx); env.update(env_extra or {})
    manifest_path = ctx.output_root / "run_manifest.json"
    if manifest_path.is_file():
        try:
            run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            run_manifest = {"stages": []}
    else:
        run_manifest = {"stages": []}
    run_manifest.setdefault("stages", []).append({
        "stage": name,
        "command": list(map(str, command)),
        "cwd": str(cwd or repo_root()),
        "log": str(log_path),
        "status": "running",
    })
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(list(map(str, command)), cwd=str(cwd or repo_root()), env=env, stdout=log, stderr=subprocess.STDOUT)
    run_manifest["stages"][-1]["returncode"] = proc.returncode
    run_manifest["stages"][-1]["status"] = "passed" if proc.returncode == 0 else "failed"
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"stage {name} failed with exit code {proc.returncode}; see {log_path}")

def _episode_dir(ctx: PipelineContext) -> tuple[Path, Path]:
    sid = ctx.session_id or _session_id(ctx.input_path)
    converted_root = ctx.output_root / "converted"
    return converted_root / sid, converted_root

def convert(ctx: PipelineContext) -> Path:
    destination, converted_root = _episode_dir(ctx)
    if (destination / "annotation.hdf5").is_file() and (destination / "rgb.mp4").is_file(): return destination
    mcap = _find_mcap(ctx.input_path)
    if mcap is None:
        if (ctx.input_path / "annotation.hdf5").is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists(): shutil.rmtree(destination)
            shutil.copytree(ctx.input_path, destination)
            return destination
        raise FileNotFoundError(f"no MCAP or converted episode found under {ctx.input_path}")
    hawor_dir = Path(os.environ.get("KODA_HAWOR_DIR", str(assets_root() / "models/hawor")))
    if not hawor_dir.exists():
        raise RuntimeError("HaWoR is required for conversion; set KODA_HAWOR_DIR to the official HaWoR checkout and provide its checkpoints")
    command = [sys.executable, str(repo_root() / "process_session_noviz.py"), "--mcap", str(mcap), "--out", str(destination), "--hand-model", "hawor", "--hawor-dir", str(hawor_dir), "--no-viz", "--no-skeleton"]
    if ctx.config.get("conversion", {}).get("no_ground_align", False): command.append("--no-ground-align")
    if ctx.config.get("conversion", {}).get("no_trim_hands", False): command.append("--no-trim-hands")
    _run(ctx, "conversion", command, env_extra={"KODA_HAWOR_DIR": str(hawor_dir)})
    return destination

def track(ctx: PipelineContext, converted: Path) -> Path:
    tracking_root = ctx.output_root / "tracking"; tracking_root.mkdir(parents=True, exist_ok=True)
    sid = converted.name; selected = ctx.output_root / "selected_sessions.txt"; selected.write_text(sid + "\n")
    command = ["bash", str(repo_root() / "pipeline_smooth_motion.txt")]
    cfg = ctx.config.get("tracking", {})
    extra = {"INPUT_ROOT": str(converted.parent), "OUTPUT_ROOT": str(tracking_root), "WORK_ROOT": str(ctx.output_root / "work/tracking"), "SESSION_LIST": str(selected), "GPUS": ctx.gpu, "WORKERS": "1", "KODA_PYTHON": sys.executable, "HAND_MODE": str(ctx.config.get("hand_mode", "dex3")), "ALLOW_SMOOTHING_REVIEW": "1" if ctx.frames else "0"}
    if ctx.frames: extra["MAX_FRAMES"] = str(ctx.frames)
    if cfg.get("allow_smoothing_review"): extra["ALLOW_SMOOTHING_REVIEW"] = "1"
    _run(ctx, "tracking", command, env_extra=extra)
    result = tracking_root / sid
    if not (result / "kinematics.npz").is_file(): raise RuntimeError(f"tracking completed without {result / 'kinematics.npz'}")
    return result

def view_change(ctx: PipelineContext, converted: Path, tracking: Path) -> Path:
    view_root = ctx.output_root / "view_change"; view_root.mkdir(parents=True, exist_ok=True)
    command = ["bash", str(repo_root() / "run_batch_singleframe_view_change_30to50.sh")]
    extra = {"SOURCE_ROOT": str(converted.parent), "KINEMATICS_ROOT": str(tracking.parent), "BATCH_OUTPUT_ROOT": str(view_root), "GPU": ctx.gpu, "HAND_MODE": str(ctx.config.get("hand_mode", "dex3")), "SESSION_ID": converted.name, "MAX_EPISODES": "1", "KODA_PYTHON": sys.executable, "MAX_SOURCE_FRAMES": str(ctx.frames or ""), "DIFFUERASER_MAX_IMG_SIZE": "512" if ctx.frames else "768"}
    _run(ctx, "view_change", command, env_extra=extra)
    result = view_root / converted.name
    if not result.exists(): raise RuntimeError(f"view-change completed without {result}")
    return result

def export_groot(ctx: PipelineContext, view_root: Path) -> Path:
    groot_root = ctx.output_root / "groot"; clips = ctx.output_root / "clips"
    encoder = assets_root() / "models/sonic/low_latency/model_encoder.onnx"
    command_encode = [sys.executable, "-m", "sil_tts.run_sil", "encode-kinematics-tokens", "--source-root", str(view_root.parent), "--output-root", str(clips), "--encoder", str(encoder), "--gpus", ctx.gpu]
    _run(ctx, "encode_tokens", command_encode)
    command_export = [sys.executable, "-m", "sil_tts.run_sil", "export-groot", "--source-root", str(view_root.parent), "--clips-root", str(clips), "--output", str(groot_root), "--task", str(ctx.config.get("task", "koda_demo"))]
    _run(ctx, "export_groot", command_export)
    if not (groot_root / "meta/info.json").is_file(): raise RuntimeError(f"GR00T export missing {groot_root / 'meta/info.json'}")
    return groot_root

def finetune(ctx: PipelineContext, dataset: Path) -> Path:
    output = ctx.output_root / "finetune"
    command_text = os.environ.get("KODA_FINETUNE_COMMAND") or ctx.config.get("finetune", {}).get("command")
    if not command_text: raise RuntimeError("GR00T finetune needs KODA_FINETUNE_COMMAND or finetune.command in config; the vendored trainer requires IsaacLab/IsaacSim runtime")
    env_extra = {"KODA_DATASET": str(dataset), "KODA_OUTPUT": str(output)}
    _run(ctx, "finetune", ["bash", "-lc", command_text], env_extra=env_extra)
    return output

def run_pipeline(ctx: PipelineContext) -> dict[str, str]:
    ctx.output_root.mkdir(parents=True, exist_ok=True)
    converted = convert(ctx); result = {"converted": str(converted)}
    tracking = track(ctx, converted); result["tracking"] = str(tracking)
    view = view_change(ctx, converted, tracking); result["view_change"] = str(view)
    groot = export_groot(ctx, view); result["groot"] = str(groot)
    if ctx.config.get("stages", {}).get("finetune", False): result["finetune"] = str(finetune(ctx, groot))
    (ctx.output_root / "pipeline_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
