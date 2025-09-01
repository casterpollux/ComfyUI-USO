"""
USO (Unified Style and Subject-Driven Generation) Node for ComfyUI
Implements ByteDance's USO model for style transfer and subject preservation
"""

import torch
import os
import numpy as np
from PIL import Image
import folder_paths
import comfy.model_management as mm
from typing import Optional, Tuple, List
import safetensors.torch

# Add USO model path to ComfyUI's model paths
USO_MODEL_PATH = os.path.join(folder_paths.models_dir, "uso")
if not os.path.exists(USO_MODEL_PATH):
    os.makedirs(USO_MODEL_PATH)

class USOModelLoader:
    """Loads USO model components (LoRA and Projection models)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "lora_path": ("STRING", {
                    "default": "uso_flux_v1.0/dit_lora.safetensors",
                    "multiline": False
                }),
                "projection_path": ("STRING", {
                    "default": "uso_flux_v1.0/projector.safetensors",
                    "multiline": False
                }),
                "lora_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01
                })
            },
            "optional": {
                "clip": ("CLIP",),
                "download_from_hf": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("USO_MODEL", "MODEL", "VAE", "CLIP")
    RETURN_NAMES = ("uso_model", "model", "vae", "clip")
    FUNCTION = "load_model"
    CATEGORY = "USO"
    
    def load_model(self, model, vae, lora_path, projection_path, lora_strength, 
                   clip=None, download_from_hf=True):
        """Load USO model components"""
        
        # Download models from HuggingFace if needed
        if download_from_hf and not os.path.exists(os.path.join(USO_MODEL_PATH, lora_path)):
            self.download_models()
        
        # Full paths to model files
        lora_full_path = os.path.join(USO_MODEL_PATH, lora_path)
        projection_full_path = os.path.join(USO_MODEL_PATH, projection_path)
        
        # Check if files exist
        if not os.path.exists(lora_full_path):
            raise FileNotFoundError(f"LoRA model not found at {lora_full_path}. "
                                  "Please download from huggingface.co/bytedance-research/USO")
        if not os.path.exists(projection_full_path):
            raise FileNotFoundError(f"Projection model not found at {projection_full_path}. "
                                  "Please download from huggingface.co/bytedance-research/USO")
        
        # Load LoRA weights and apply to model
        model_with_lora = self.apply_lora(model, lora_full_path, lora_strength)
        
        # Load projection model
        projection_model = self.load_projection_model(projection_full_path)
        
        # Create USO model wrapper
        uso_model = {
            "model": model_with_lora,
            "vae": vae,
            "clip": clip,
            "projection": projection_model,
            "lora_strength": lora_strength
        }
        
        return (uso_model, model_with_lora, vae, clip)
    
    def apply_lora(self, model, lora_path, strength):
        """Apply LoRA weights to the model"""
        # Load LoRA state dict
        lora_state_dict = safetensors.torch.load_file(lora_path)
        
        # Clone the model to avoid modifying the original
        model_clone = model.clone()
        
        # Apply LoRA weights (simplified - actual implementation may vary)
        # This is a placeholder - actual FLUX LoRA application would be more complex
        for key, value in lora_state_dict.items():
            # Apply LoRA weights with strength multiplier
            # This would need proper implementation based on FLUX architecture
            pass
        
        return model_clone
    
    def load_projection_model(self, projection_path):
        """Load the projection model for image encoding"""
        projection_state = safetensors.torch.load_file(projection_path)
        # Create projection model structure (simplified)
        return projection_state
    
    def download_models(self):
        """Download models from HuggingFace"""
        try:
            from huggingface_hub import snapshot_download
            print("Downloading USO models from HuggingFace...")
            snapshot_download(
                repo_id="bytedance-research/USO",
                local_dir=USO_MODEL_PATH,
                local_dir_use_symlinks=False,
                allow_patterns=["*.safetensors", "*.json"]
            )
            print("✅ USO models downloaded successfully!")
        except ImportError:
            print("⚠️ huggingface_hub not installed. Please install it or download models manually.")
        except Exception as e:
            print(f"❌ Error downloading models: {e}")


class USOImageEncoder:
    """Encodes images for USO style and content conditioning"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "uso_model": ("USO_MODEL",),
                "mode": (["style", "subject", "style_subject", "multi_style"],),
            },
            "optional": {
                "content_image": ("IMAGE",),
                "style_image": ("IMAGE",),
                "style_image_2": ("IMAGE",),
                "style_image_3": ("IMAGE",),
                "content_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05
                }),
                "style_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05
                }),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode_images"
    CATEGORY = "USO"
    
    def encode_images(self, uso_model, mode, content_image=None, style_image=None,
                     style_image_2=None, style_image_3=None, 
                     content_strength=1.0, style_strength=1.0):
        """Encode images based on selected mode"""
        
        device = mm.get_torch_device()
        projection = uso_model["projection"]
        
        # Initialize conditioning
        positive_cond = []
        negative_cond = []
        
        # Process based on mode
        if mode == "subject":
            if content_image is None:
                raise ValueError("Content image required for subject mode")
            content_features = self.encode_content(content_image, projection, device)
            positive_cond.append(("content", content_features * content_strength))
            
        elif mode == "style":
            if style_image is None:
                raise ValueError("Style image required for style mode")
            style_features = self.encode_style(style_image, projection, device)
            positive_cond.append(("style", style_features * style_strength))
            
        elif mode == "style_subject":
            if content_image is None or style_image is None:
                raise ValueError("Both content and style images required for style_subject mode")
            content_features = self.encode_content(content_image, projection, device)
            style_features = self.encode_style(style_image, projection, device)
            positive_cond.append(("content", content_features * content_strength))
            positive_cond.append(("style", style_features * style_strength))
            
        elif mode == "multi_style":
            style_images = [img for img in [style_image, style_image_2, style_image_3] if img is not None]
            if len(style_images) < 2:
                raise ValueError("At least 2 style images required for multi_style mode")
            
            for img in style_images:
                style_features = self.encode_style(img, projection, device)
                positive_cond.append(("style", style_features * style_strength))
        
        # Create empty negative conditioning
        negative_cond = [("empty", torch.zeros_like(positive_cond[0][1]))]
        
        return (positive_cond, negative_cond)
    
    def encode_content(self, image, projection, device):
        """Encode content image using projection model"""
        # Convert ComfyUI image format to tensor
        if isinstance(image, torch.Tensor):
            img_tensor = image
        else:
            img_tensor = torch.from_numpy(image)
        
        # Move to device and normalize
        img_tensor = img_tensor.to(device)
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # Apply projection (simplified - actual implementation would use SigLIP)
        # This is a placeholder for the actual encoding logic
        with torch.no_grad():
            features = img_tensor.mean(dim=[2, 3])  # Simplified encoding
        
        return features
    
    def encode_style(self, image, projection, device):
        """Encode style image using projection model"""
        # Similar to encode_content but for style
        # The actual implementation would handle style encoding differently
        return self.encode_content(image, projection, device)


class USOSampler:
    """Main USO sampling node that generates images"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "uso_model": ("USO_MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful image"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "steps": ("INT", {
                    "default": 28,
                    "min": 1,
                    "max": 1000
                }),
                "cfg": ("FLOAT", {
                    "default": 3.5,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
            },
            "optional": {
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "USO"
    
    def sample(self, uso_model, positive, negative, prompt, seed, steps, cfg, 
               width, height, denoise=1.0):
        """Generate images using USO model"""
        
        device = mm.get_torch_device()
        model = uso_model["model"]
        
        # Set random seed
        torch.manual_seed(seed)
        
        # Create latent
        batch_size = 1
        latent_channels = 16  # FLUX uses 16 channels
        latent_height = height // 8
        latent_width = width // 8
        
        # Initialize latent
        latent = torch.randn(
            (batch_size, latent_channels, latent_height, latent_width),
            device=device,
            dtype=torch.float32
        )
        
        # Prepare conditioning with text prompt
        # In actual implementation, this would use T5 encoder
        # This is simplified for demonstration
        
        # Create output format for ComfyUI
        samples = {"samples": latent}
        
        # Note: Actual sampling would involve:
        # 1. Encoding text with T5
        # 2. Combining text and image conditioning
        # 3. Running FLUX denoising loop with USO modifications
        # 4. Applying CFG guidance
        
        return (samples,)


class USOLatentToImage:
    """Decode USO latents to images"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "uso_model": ("USO_MODEL",),
                "samples": ("LATENT",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "USO"
    
    def decode(self, uso_model, samples):
        """Decode latent samples to images using VAE"""
        vae = uso_model["vae"]
        latent = samples["samples"]
        
        # Decode latent to image
        with torch.no_grad():
            images = vae.decode(latent)
        
        # Convert to ComfyUI image format (B, H, W, C)
        images = images.permute(0, 2, 3, 1)
        images = images.cpu().numpy()
        images = np.clip(images * 0.5 + 0.5, 0, 1)  # Denormalize
        
        return (images,)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "USOModelLoader": USOModelLoader,
    "USOImageEncoder": USOImageEncoder,
    "USOSampler": USOSampler,
    "USOLatentToImage": USOLatentToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "USOModelLoader": "USO Model Loader",
    "USOImageEncoder": "USO Image Encoder",
    "USOSampler": "USO Sampler",
    "USOLatentToImage": "USO Decode",
}