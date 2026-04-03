"""Video I/O utilities"""
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import torch

def load_video(video_path: str, max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], dict]:
    """
    Load video as list of frames
    
    Returns:
        frames: List of numpy arrays (H, W, C)
        info: Dict with fps, total_frames, etc.
    """
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
        count += 1
        if max_frames and count >= max_frames:
            break
            
    cap.release()
    
    info = {
        'fps': fps,
        'total_frames': total_frames,
        'width': width,
        'height': height,
        'loaded_frames': len(frames)
    }
    
    return frames, info

def save_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: float = 30.0,
    codec: str = 'mp4v'
):
    """Save frames as video"""
    if not frames:
        raise ValueError("No frames to save")
        
    height, width = frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # RGB to BGR
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        
    out.release()
    
def extract_frames(
    video_path: str,
    output_dir: str,
    interval: int = 1
) -> List[str]:
    """Extract frames from video to images"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % interval == 0:
            path = os.path.join(output_dir, f"frame_{count:06d}.png")
            cv2.imwrite(path, frame)
            frame_paths.append(path)
            
        count += 1
        
    cap.release()
    return frame_paths

def frames_to_tensor(frames: List[np.ndarray]) -> torch.Tensor:
    """Convert list of frames to tensor [T, C, H, W]"""
    frames = [torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frames]
    return torch.stack(frames)

def tensor_to_frames(tensor: torch.Tensor) -> List[np.ndarray]:
    """Convert tensor [T, C, H, W] to list of frames"""
    tensor = (tensor.clamp(0, 1) * 255).byte()
    return [t.permute(1, 2, 0).cpu().numpy() for t in tensor]
