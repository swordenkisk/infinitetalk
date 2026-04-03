"""Command Line Interface"""
import argparse
import sys
from pathlib import Path

def generate_command(args):
    """Generate video command"""
    print(f"Generating video from {args.image} and {args.audio}")
    print(f"Output: {args.output}")
    print(f"Resolution: {args.resolution}")
    print(f"Frames: {args.num_frames}")
    print(f"Steps: {args.num_steps}")
    
    # TODO: Implement actual generation
    print("\nNote: This is a placeholder. Actual generation requires model weights.")
    return 0

def serve_command(args):
    """Start API server"""
    import uvicorn
    from infinitetalk.api.server import create_app
    
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)
    return 0

def ui_command(args):
    """Start Gradio UI"""
    from infinitetalk.api.gradio_app import create_gradio_app
    
    demo = create_gradio_app()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0

def benchmark_command(args):
    """Run benchmarks"""
    import subprocess
    
    cmd = [
        sys.executable,
        "scripts/benchmark.py",
        "--resolution", args.resolution,
        "--num-frames", str(args.num_frames),
        "--num-runs", str(args.num_runs)
    ]
    
    subprocess.run(cmd)
    return 0

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="InfiniteTalk - Sparse-Frame Video Dubbing"
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate video')
    gen_parser.add_argument('--image', '-i', required=True, help='Reference image path')
    gen_parser.add_argument('--audio', '-a', required=True, help='Audio file path')
    gen_parser.add_argument('--output', '-o', required=True, help='Output video path')
    gen_parser.add_argument('--config', '-c', help='Config file path')
    gen_parser.add_argument('--resolution', '-r', default='720p', choices=['480p', '720p'])
    gen_parser.add_argument('--num-frames', '-n', type=int, default=120)
    gen_parser.add_argument('--num-steps', type=int, default=4)
    gen_parser.add_argument('--guidance-scale', type=float, default=1.5)
    gen_parser.set_defaults(func=generate_command)
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument('--host', default='0.0.0.0', help='Server host')
    serve_parser.add_argument('--port', '-p', type=int, default=8000, help='Server port')
    serve_parser.set_defaults(func=serve_command)
    
    # UI command
    ui_parser = subparsers.add_parser('ui', help='Start web UI')
    ui_parser.add_argument('--host', default='0.0.0.0', help='Server host')
    ui_parser.add_argument('--port', '-p', type=int, default=7860, help='Server port')
    ui_parser.add_argument('--share', action='store_true', help='Create public link')
    ui_parser.set_defaults(func=ui_command)
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    bench_parser.add_argument('--resolution', '-r', default='720p', choices=['480p', '720p'])
    bench_parser.add_argument('--num-frames', '-n', type=int, default=120)
    bench_parser.add_argument('--num-runs', type=int, default=5)
    bench_parser.set_defaults(func=benchmark_command)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
        
    return args.func(args)

if __name__ == '__main__':
    sys.exit(main())
