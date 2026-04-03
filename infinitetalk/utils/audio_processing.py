"""Audio processing utilities"""
import torch
import torchaudio
import numpy as np
from typing import Optional, Tuple

def load_audio(
    audio_path: str,
    target_sr: int = 16000,
    max_length: Optional[int] = None
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file
    
    Returns:
        waveform: [channels, samples]
        sample_rate
    """
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr
        
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        
    # Trim if needed
    if max_length and waveform.shape[1] > max_length:
        waveform = waveform[:, :max_length]
        
    return waveform, sr

def extract_audio_features(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    feature_type: str = "mel"
) -> torch.Tensor:
    """
    Extract audio features
    
    Args:
        waveform: [1, samples]
        feature_type: "mel", "mfcc", or "raw"
        
    Returns:
        features: [time, feature_dim]
    """
    if feature_type == "mel":
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        features = transform(waveform)
        # Convert to log scale
        features = torch.log(features + 1e-6)
        
    elif feature_type == "mfcc":
        transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,
            melkwargs={'n_fft': n_fft, 'hop_length': hop_length}
        )
        features = transform(waveform)
        
    else:  # raw
        # Just normalize
        features = waveform.unsqueeze(0)
        
    return features.squeeze(0).transpose(0, 1)  # [time, dim]

def align_audio_to_frames(
    audio_features: torch.Tensor,
    num_frames: int,
    fps: float = 30.0
) -> torch.Tensor:
    """
    Align audio features to video frames
    
    Args:
        audio_features: [time, dim]
        num_frames: Number of video frames
        fps: Video frame rate
        
    Returns:
        aligned: [num_frames, dim]
    """
    # Simple linear interpolation
    audio_len = audio_features.shape[0]
    
    # Create interpolation indices
    indices = torch.linspace(0, audio_len - 1, num_frames)
    indices = indices.long()
    
    return audio_features[indices]
