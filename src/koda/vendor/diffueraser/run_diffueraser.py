import torch
import os 
import time
import argparse
import json
import gc
from diffueraser.diffueraser import DiffuEraser
from propainter.inference import Propainter, get_device

def main():

    ## input params
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, default="examples/example3/video.mp4", help='Path to the input video')
    parser.add_argument('--input_mask', type=str, default="examples/example3/mask.mp4" , help='Path to the input mask')
    parser.add_argument('--video_length', type=float, default=10.0, help='Maximum input video duration in seconds')
    parser.add_argument('--mask_dilation_iter', type=int, default=8, help='Adjust it to change the degree of mask expansion')
    parser.add_argument('--max_img_size', type=int, default=960, help='The maximum length of output width and height')
    parser.add_argument('--save_path', type=str, default="results" , help='Path to the output')
    parser.add_argument('--ref_stride', type=int, default=10, help='Propainter params')
    parser.add_argument('--neighbor_length', type=int, default=10, help='Propainter params')
    parser.add_argument('--subvideo_length', type=int, default=50, help='Propainter params')
    parser.add_argument('--seed', type=int, default=42, help='Deterministic DiffuEraser noise seed')
    parser.add_argument('--ckpt', type=str, default='Normal CFG 8-Step',
                        choices=['2-Step', '4-Step', '8-Step', '16-Step',
                                 'Normal CFG 4-Step', 'Normal CFG 8-Step',
                                 'Normal CFG 16-Step', 'LCM-Like LoRA'],
                        help='PCM checkpoint and sampling contract')
    parser.add_argument('--keep_priori', action='store_true',
                        help='Keep the ProPainter prior beside the diffusion result for audit')
    parser.add_argument('--base_model_path', type=str, default="weights/stable-diffusion-v1-5" , help='Path to sd1.5 base model')
    parser.add_argument('--vae_path', type=str, default="weights/sd-vae-ft-mse" , help='Path to vae')
    parser.add_argument('--diffueraser_path', type=str, default="weights/diffuEraser" , help='Path to DiffuEraser')
    parser.add_argument('--propainter_model_dir', type=str, default="weights/propainter" , help='Path to priori model')
    parser.add_argument('--show-progress', action='store_true')
    parser.add_argument('--ff-chunk-size', type=int, default=0, help='Spatial feed-forward batch chunk size (0 disables); leaves temporal layers unchanged')
    parser.add_argument('--sequential-models', action='store_true', help='Unload ProPainter before loading diffusion to reduce peak VRAM')
    parser.add_argument('--prior-resize-ratio', type=float, default=0.6, help='ProPainter prior size relative to input (0 < ratio <= 1)')
    args = parser.parse_args()
    if not 0 < args.prior_resize_ratio <= 1:
        parser.error('--prior-resize-ratio must be in (0, 1]')
                  
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    priori_path = os.path.join(args.save_path, "priori.mp4")                        
    output_path = os.path.join(args.save_path, "diffueraser_result.mp4") 
    
    torch.manual_seed(args.seed)

    ## model initialization
    device = get_device()
    # PCM params. Normal CFG is the actual diffusion path; the old hardcoded
    # 2-Step/small-CFG setting made the result look like a ProPainter-only blur.
    video_inpainting_sd = None
    if not args.sequential_models:
        video_inpainting_sd = DiffuEraser(
            device, args.base_model_path, args.vae_path, args.diffueraser_path,
            ckpt=args.ckpt,
        )
    propainter = Propainter(args.propainter_model_dir, device=device)
    
    start_time = time.time()

    ## priori
    propainter.forward(args.input_video, args.input_mask, priori_path, video_length=args.video_length, 
                        ref_stride=args.ref_stride, neighbor_length=args.neighbor_length, subvideo_length = args.subvideo_length,
                        mask_dilation = args.mask_dilation_iter, resize_ratio=args.prior_resize_ratio)

    if args.sequential_models:
        del propainter
        gc.collect()
        torch.cuda.empty_cache()
        video_inpainting_sd = DiffuEraser(
            device, args.base_model_path, args.vae_path, args.diffueraser_path, ckpt=args.ckpt,
        )

    if args.ff_chunk_size > 0:
        from diffusers.models.transformers.transformer_2d import Transformer2DModel
        for module in video_inpainting_sd.unet_main.modules():
            if isinstance(module, Transformer2DModel):
                for block in module.transformer_blocks:
                    block.set_chunk_feed_forward(chunk_size=args.ff_chunk_size, dim=0)
    if args.show_progress:
        video_inpainting_sd.pipeline.set_progress_bar_config(disable=False)

    ## diffueraser
    guidance_scale = None
    video_inpainting_sd.forward(args.input_video, args.input_mask, priori_path, output_path,
                                max_img_size = args.max_img_size, video_length=args.video_length, mask_dilation_iter=args.mask_dilation_iter,
                                guidance_scale=guidance_scale, seed=args.seed,
                                keep_priori=args.keep_priori)
    run_manifest = {
        "backend": "DiffuEraser",
        "stages": ["ProPainter", "DiffuEraser diffusion"],
        "checkpoint": args.ckpt,
        "num_inference_steps": int(video_inpainting_sd.num_inference_steps),
        "guidance_scale": float(video_inpainting_sd.guidance_scale),
        "max_img_size": int(args.max_img_size),
        "seed": int(args.seed),
        "prior_resize_ratio": float(args.prior_resize_ratio),
        "sequential_models": bool(args.sequential_models),
        "ff_chunk_size": int(args.ff_chunk_size),
        "ff_chunk_scope": "spatial_transformer_only",
        "torch_global_seed": int(args.seed),
        "priori_path": os.path.abspath(priori_path),
        "priori_preserved": bool(args.keep_priori and os.path.isfile(priori_path)),
        "output_path": os.path.abspath(output_path),
    }
    with open(os.path.join(args.save_path, "diffueraser_run.json"), "w") as stream:
        json.dump(run_manifest, stream, indent=2)
        stream.write("\n")
    
    end_time = time.time()  
    inference_time = end_time - start_time  
    print(f"DiffuEraser inference time: {inference_time:.4f} s")

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
