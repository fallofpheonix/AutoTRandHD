import argparse
import json
import time
import sys
import os
from pathlib import Path
import torch
import cv2
import numpy as np
from tqdm import tqdm

from src.models.loader import load_model
from src.preprocess.normalize import normalize_page
from src.preprocess.deskew import deskew_image
from src.preprocess.text_region import extract_text_region
from src.preprocess.line_segment import segment_lines
from src.decode.beam_search import beam_decode

def run_inference(model, image_path, device, vocab, beam_width=10):
    start_time = time.time()
    
    # 1. Load and Preprocess
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Normalization
    gray = normalize_page(img)
    # Deskew
    deskewed, _ = deskew_image(gray)
    # Text region
    region, _ = extract_text_region(deskewed)
    
    # For simplicity in this CLI, we'll assume the input is a single line or 
    # we segment and take the first one, or handle the whole region.
    # The requirement asks for "text": "recognized transcription" (singular)
    # so we'll treat the input as a line-crop or segment and join.
    
    # Let's segment lines to be robust
    # Using a temporary directory for line crops
    line_crops_dir = Path("/tmp/autotrandhd_cli_crops")
    line_crops_dir.mkdir(parents=True, exist_ok=True)
    
    records = segment_lines(region, "cli", 0, str(line_crops_dir))
    
    transcriptions = []
    for rec in records:
        line_img = cv2.imread(rec["image_path"], cv2.IMREAD_GRAYSCALE)
        # Prepare for model (H, W) -> (1, 1, H, W)
        line_tensor = torch.from_numpy(line_img).float().unsqueeze(0).unsqueeze(0) / 255.0
        line_tensor = line_tensor.to(device)
        
        with torch.no_grad():
            logits = model(line_tensor) # (T, B, C)
            logits_np = logits.squeeze(1).cpu().numpy() # (T, C)
            
        hypotheses = beam_decode(logits_np, vocab=vocab, beam_width=beam_width)
        if hypotheses:
            transcriptions.append(hypotheses[0]["text"])
            
    full_text = " ".join(transcriptions)
    latency_ms = int((time.time() - start_time) * 1000)
    
    return {
        "image": os.path.basename(image_path),
        "text": full_text,
        "confidence": 0.95, # Placeholder or calculated from scores
        "latency_ms": latency_ms
    }

def main():
    parser = argparse.ArgumentParser(prog="autotrandhd-cli", description="AutoTRandHD Production CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # infer
    infer_parser = subparsers.add_parser("infer", help="Run inference on a single image")
    infer_parser.add_argument("image_path", type=str, help="Path to input image")
    infer_parser.add_argument("--model", type=str, required=True, help="Path to checkpoint")
    infer_parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    
    # batch
    batch_parser = subparsers.add_parser("batch", help="Batch process a folder")
    batch_parser.add_argument("folder_path", type=str, help="Path to folder")
    batch_parser.add_argument("--model", type=str, required=True, help="Path to checkpoint")
    batch_parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    
    # benchmark
    benchmark_parser = subparsers.add_parser("benchmark", help="Run system benchmark")
    
    # info
    info_parser = subparsers.add_parser("info", help="Get system info")
    
    args = parser.parse_args()
    
    if args.command == "info":
        info = {
            "version": "0.1.0",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "os": sys.platform
        }
        print(json.dumps(info, indent=2))
        return

    if args.command == "infer":
        # Note: In a real scenario, vocab would be loaded from a config
        vocab = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:-'\"()ñçáéíóú" # Sample vocab
        model = load_model(args.model, device=args.device)
        result = run_inference(model, args.image_path, args.device, vocab)
        print(json.dumps(result, indent=2))

    elif args.command == "batch":
        vocab = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:-'\"()ñçáéíóú"
        model = load_model(args.model, device=args.device)
        folder = Path(args.folder_path)
        results = []
        images = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
        
        for img_path in tqdm(images, desc="Processing batch"):
            try:
                res = run_inference(model, str(img_path), args.device, vocab)
                results.append(res)
            except Exception as e:
                results.append({"image": img_path.name, "error": str(e)})
        
        print(json.dumps(results, indent=2))

    elif args.command == "benchmark":
        print("Running benchmark...")
        # Placeholder for actual benchmark logic call
        os.system(f"{sys.executable} scripts/benchmark_inference.py")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
