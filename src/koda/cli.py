from __future__ import annotations
import argparse, json, os
from pathlib import Path
from . import __version__
from .assets import download, verify
from .orchestrator import PipelineContext, convert, export_groot, finetune, load_config, run_pipeline, track, view_change
from .validation import validate

def _parser():
    parser = argparse.ArgumentParser(prog="koda", description="KODA humanoid data pipeline")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    assets = sub.add_parser("assets"); asset_sub = assets.add_subparsers(dest="assets_command", required=True)
    down = asset_sub.add_parser("download"); down.add_argument("--profile", default="smoke")
    ver = asset_sub.add_parser("verify"); ver.add_argument("--profile", default="smoke")
    val = sub.add_parser("validate"); val.add_argument("--profile", default="smoke")
    def stage(name):
        p = sub.add_parser(name); p.add_argument("--input", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--config", type=Path, default=Path("configs/default.yaml")); p.add_argument("--frames", type=int); p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0")); return p
    for name in ("convert", "track", "view-change", "export-groot", "finetune"): stage(name)
    run = sub.add_parser("run"); run.add_argument("--input", type=Path, required=True); run.add_argument("--output", type=Path, required=True); run.add_argument("--config", type=Path, default=Path("configs/default.yaml")); run.add_argument("--frames", type=int); run.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    return parser

def _context(args):
    config_path = args.config.resolve(); config = load_config(config_path)
    return PipelineContext(input_path=args.input.resolve(), output_root=args.output.resolve(), config_path=config_path, config=config, frames=args.frames, gpu=args.gpu)

def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "assets":
        if args.assets_command == "download":
            for path in download(args.profile): print(path)
        else: print(json.dumps({"missing": verify(profile=args.profile)}, indent=2))
        return 0
    if args.command == "validate":
        result = validate(args.profile); print(json.dumps(result, indent=2)); return 0 if result["passed"] else 2
    ctx = _context(args)
    if args.command == "run":
        print(json.dumps(run_pipeline(ctx), indent=2)); return 0
    if args.command == "convert": print(convert(ctx)); return 0
    converted = ctx.input_path
    if args.command == "track": print(track(ctx, converted)); return 0
    if args.command == "view-change": print(view_change(ctx, converted, ctx.output_root / "tracking" / converted.name)); return 0
    if args.command == "export-groot": print(export_groot(ctx, converted)); return 0
    if args.command == "finetune": print(finetune(ctx, converted)); return 0
    raise AssertionError(args.command)

if __name__ == "__main__": raise SystemExit(main())
