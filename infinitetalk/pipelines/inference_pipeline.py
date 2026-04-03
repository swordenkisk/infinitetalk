"""Main Inference Pipeline - 4-step video generation"""
import torch
from typing import Optional, Union, List
from diffusers import DiffusionPipeline
import numpy as np

class InfiniteTalkPipeline(DiffusionPipeline):
    """
    Main pipeline for InfiniteTalk sparse-frame video generation
    """
    
    def __init__(self, unet, vae, audio_encoder, scheduler):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            audio_encoder=audio_encoder,
            scheduler=scheduler
        )
        
    @torch.no_grad()
    def __call__(
        self,
        image: Union[torch.Tensor, np.ndarray],
        audio: Union[torch.Tensor, np.ndarray],
        num_frames: int = 120,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.5,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil"
    ):
        """
        Generate video from image and audio
        
        Args:
            image: Reference image [B, C, H, W]
            audio: Audio features [B, T, D]
            num_frames: Number of frames to generate
            num_inference_steps: Number of denoising steps (default 4)
            guidance_scale: CFG scale
            generator: Random generator
            output_type: "pil" or "latent" or "np"
        """
        device = self._execution_device
        
        # Encode reference image
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(device=device, dtype=self.unet.dtype)
            
        ref_latent = self.vae.encode(image).latent_dist.sample(generator)
        ref_latent = ref_latent * self.vae.config.scaling_factor
        
        # Encode audio
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).to(device=device)
            
        audio_features = self.audio_encoder(audio)
        
        # Prepare latents
        shape = (
            image.shape[0],
            self.unet.config.in_channels // 2,  # Account for reference concat
            num_frames,
            ref_latent.shape[-2] // 8,
            ref_latent.shape[-1] // 8
        )
        latents = torch.randn(shape, generator=generator, device=device, dtype=self.unet.dtype)
        
        # Set timesteps (4-step)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            # Expand latents for batch
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Concatenate reference
            ref_expanded = ref_latent.unsqueeze(2).expand(-1, -1, num_frames, -1, -1)
            latent_model_input = torch.cat([latent_model_input, ref_expanded], dim=1)
            
            # Predict noise
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=audio_features
            ).sample
            
            # CFG
            if guidance_scale > 1:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
            # Step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        # Decode
        if output_type == "latent":
            return latents
            
        # VAE decode in chunks
        frames = []
        chunk_size = 8
        for i in range(0, num_frames, chunk_size):
            chunk = latents[:, :, i:i+chunk_size].transpose(1, 2)
            b, t, c, h, w = chunk.shape
            chunk = chunk.reshape(b * t, c, h, w)
            decoded = self.vae.decode(chunk / self.vae.config.scaling_factor).sample
            decoded = decoded.reshape(b, t, 3, decoded.shape[-2], decoded.shape[-1])
            frames.append(decoded)
            
        frames = torch.cat(frames, dim=1)
        
        if output_type == "pil":
            from diffusers.utils import export_to_video
            return export_to_video(frames)
            
        return frames.cpu().numpy()
