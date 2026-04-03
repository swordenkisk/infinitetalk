#!/usr/bin/env python3
"""Benchmark script for InfiniteTalk"""
import torch
import time
import argparse
from pathlib import Path
import json

from infinitetalk.pipelines.inference_pipeline import InfiniteTalkPipeline
from infinitetalk.core.sparse_frame_generator import SparseFrameGenerator

def benchmark_inference(
    model_path: str,
    num_frames: int = 120,
    resolution: str = "720p",
    num_runs: int = 5,
    warmup: int = 2
):
    """Benchmark inference speed"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Parse resolution
    res_map = {"480p": (854, 480), "720p": (1280, 720)}
    w, h = res_map.get(resolution, (1280, 720))
    
    # Create dummy inputs
    dummy_image = torch.randn(1, 3, h, w).to(device)
    dummy_audio = torch.randn(1, num_frames, 512).to(device)
    
    results = {
        'resolution': resolution,
        'num_frames': num_frames,
        'device': str(device),
        'runs': []
    }
    
    print(f"\nBenchmarking {resolution} with {num_frames} frames...")
    print("-" * 50)
    
    for i in range(warmup + num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        # Simulate inference (replace with actual pipeline)
        # output = pipeline(image=dummy_image, audio=dummy_audio, num_frames=num_frames)
        torch.randn(1000, 1000, device=device)  # Dummy work
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        
        if i >= warmup:
            results['runs'].append(elapsed)
            print(f"Run {i - warmup + 1}: {elapsed:.3f}s")
            
    # Statistics
    times = results['runs']
    results['mean'] = sum(times) / len(times)
    results['min'] = min(times)
    results['max'] = max(times)
    results['fps'] = num_frames / results['mean']
    
    print("-" * 50)
    print(f"Mean time: {results['mean']:.3f}s")
    print(f"Min time: {results['min']:.3f}s")
    print(f"Max time: {results['max']:.3f}s")
    print(f"Effective FPS: {results['fps']:.2f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark InfiniteTalk")
    parser.add_argument("--resolution", choices=["480p", "720p"], default="720p")
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    
    args = parser.parse_args()
    
    results = benchmark_inference(
        model_path="",
        num_frames=args.num_frames,
        resolution=args.resolution,
        num_runs=args.num_runs
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
