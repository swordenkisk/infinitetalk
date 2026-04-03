import torch
import torch.nn as nn
from diffusers.models.unets.unet_3d_condition import UNet3DConditionModel
from peft import PeftModel

class HybridUNet4Step(nn.Module):
    """
    UNet optimized for 4-step inference using Consistency Models
    with FusioniX and LightX2V LoRA fusion
    """
    def __init__(self, base_model_path: str):
        super().__init__()
        self.unet = UNet3DConditionModel.from_pretrained(base_model_path, subfolder="unet")
        self.num_inference_steps = 4
        
        # Timestep schedule for 4-step (distilled)
        self.distilled_timesteps = torch.tensor([999, 749, 499, 249])
        
    def load_dual_lora(self, fusionix_path: str, lightx2v_path: str):
        """Load both LoRAs with specific weights"""
        # FusioniX for temporal consistency (high weight)
        self.unet = PeftModel.from_pretrained(self.unet, fusionix_path, adapter_name="fusionix")
        
        # LightX2V for 4-step speed (medium weight)
        self.unet.load_adapter(lightx2v_path, adapter_name="lightx2v")
        
        # Set mixing weights
        self.unet.set_adapters(["fusionix", "lightx2v"], [0.8, 0.6])
        
    def forward(self, sample, timestep, encoder_hidden_states, 
                reference_latent=None, return_dict=True):
        """
        Modified forward with sparse reference conditioning
        """
        if reference_latent is not None:
            # Concatenate reference along channel dim for identity preservation
            sample = torch.cat([sample, reference_latent], dim=1)
            
        # Standard UNet forward with LoRA adapters active
        return self.unet(sample, timestep, encoder_hidden_states, return_dict=return_dict)
    
    def get_consistency_target(self, x_t, t, x_anchor):
        """
        LightX2V consistency function: predict x_0 directly or jump to anchor
        """
        if t in self.distilled_timesteps[-2:]:  # Final steps
            return x_anchor
        return None
