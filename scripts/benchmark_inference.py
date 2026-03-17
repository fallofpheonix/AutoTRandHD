import argparse
import json
import os
import time

import numpy as np
import psutil
import torch

from autotrandhd.config import load_settings
from autotrandhd.core.recognition.loader import load_model

def run_benchmark(checkpoint, num_images=100, batch_size=16, device="cpu"):
    print(f"Benchmarking AutoTRandHD (Device: {device}, Batch Size: {batch_size})")
    
    # Load Model
    start_load = time.time()
    if not os.path.exists(checkpoint):
        print(f"Warning: Checkpoint {checkpoint} not found. Using randomly initialized model.")
        from autotrandhd.core.recognition.crnn import CRNN
        model = CRNN(num_classes=100).to(device)
    else:
        model = load_model(checkpoint, device=device)
    load_time = time.time() - start_load
    
    model.eval()
    
    # Metrics
    latencies = []
    mem_usages = []
    
    process = psutil.Process(os.getpid())
    
    dummy_input = torch.rand(batch_size, 1, 64, 256).to(device)
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy_input)
            
    total_start = time.time()
    for _ in range(num_images // batch_size):
        batch_input = torch.rand(batch_size, 1, 64, 256).to(device)
        
        start_step = time.time()
        with torch.no_grad():
            _ = model(batch_input)
        latencies.append((time.time() - start_step) / batch_size)
        mem_usages.append(process.memory_info().rss / 1024 / 1024)
        
    total_time = time.time() - total_start
    
    report = {
        "throughput_images_sec": round(num_images / total_time, 2),
        "avg_latency_ms": round(np.mean(latencies) * 1000, 2),
        "peak_memory_mb": round(np.max(mem_usages), 2),
        "load_time_sec": round(load_time, 2),
        "device": device
    }
    
    print("-" * 30)
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    settings = load_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(settings.model_path))
    parser.add_argument("--images", type=int, default=100)
    parser.add_argument("--batch", type=int, default=settings.benchmark_batch_size)
    parser.add_argument("--device", type=str, default=settings.device)
    args = parser.parse_args()
    
    run_benchmark(args.checkpoint, args.images, args.batch, args.device)
