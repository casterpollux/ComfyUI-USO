"""
USO Inference Pipeline for ComfyUI
Adapted from ByteDance USO implementation
"""

import torch
import torch.nn as nn
from typing import List, Optional, Union, Tuple
import numpy as np
from PIL import Image
from transformers import T5Tokenizer, T5EncoderModel, CLIPTextModel, CLIPTokenizer
from diffusers import FluxPipeline, AutoencoderKL
import safetensors.torch
from dataclasses import dataclass
import os

@dataclass
class USOConfig:
    """Configuration for USO model"""
    num_inference_steps: int = 25  # Updated to match HF demo
    guidance_scale: float = 4.0    # Updated to match HF demo
    height: int = 1024
    width: int = 1024
    max_sequence_length: int = 512
    content_reference_size: int = 512  # Added from HF demo
    torch_dtype: torch.dtype = torch.bfloat16
    
    # USO specific
    content_encoder_path: str = "google/siglip-so400m-patch14-384"
    style_encoder_path: str = "google/siglip-so400m-patch14-384"
    projection_dim: int = 4096
    
    # Conditioning strengths
    default_content_strength: float = 1.0
    default_style_strength: float = 1.0


def preprocess_reference_image(image: Image.Image, size: int = 512) -> Image.Image:
    """Preprocess reference images to specified size"""
    # Center crop to square
    width, height = image.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    image = image.crop((left, top, right, bottom))
    
    # Resize to target size
    image = image.resize((size, size), Image.LANCZOS)
    return image


class ProjectionModel(nn.Module):
    """Projection model for USO image conditioning"""
    
    def __init__(self, input_dim: int = 1152, output_dim: int = 4096):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projection(x)
    
    @classmethod
    def from_pretrained(cls, path: str, device: torch.device):
        """Load projection model from checkpoint"""
        state_dict = safetensors.torch.load_file(path)
        
        # Infer dimensions from state dict
        input_dim = state_dict['projection.0.weight'].shape[1]
        output_dim = state_dict['projection.2.weight'].shape[0]
        
        model = cls(input_dim, output_dim)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        return model


class ImageEncoder:
    """Encode images using SigLIP for USO conditioning"""
    
    def __init__(self, model_path: str, device: torch.device):
        from transformers import AutoProcessor, AutoModel
        
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.model.eval()
    
    @torch.no_grad()
    def encode(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """Encode a single image to features"""
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL Image
            if image.dim() == 4:
                image = image[0]
            image = image.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        
        # Get pooled features
        features = outputs.pooler_output
        return features


class USOPipeline:
    """Main USO inference pipeline"""
    
    def __init__(
        self,
        flux_model: FluxPipeline,
        projection_model: ProjectionModel,
        content_encoder: Optional[ImageEncoder] = None,
        style_encoder: Optional[ImageEncoder] = None,
        device: torch.device = torch.device("cuda"),
        config: Optional[USOConfig] = None
    ):
        self.flux = flux_model
        self.projection = projection_model
        self.content_encoder = content_encoder
        self.style_encoder = style_encoder
        self.device = device
        self.config = config or USOConfig()
    
    def encode_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompt using T5 and CLIP"""
        # Get text embeddings from FLUX pipeline
        prompt_embeds, pooled_prompt_embeds = self.flux.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=self.config.max_sequence_length,
        )
        
        return prompt_embeds, pooled_prompt_embeds
    
    def prepare_image_conditioning(
        self,
        content_images: Optional[List[Image.Image]] = None,
        style_images: Optional[List[Image.Image]] = None,
        content_strength: float = 1.0,
        style_strength: float = 1.0
    ) -> torch.Tensor:
        """Prepare image conditioning from content and style images"""
        
        conditions = []
        
        # Encode content images
        if content_images and self.content_encoder:
            for img in content_images:
                features = self.content_encoder.encode(img)
                projected = self.projection(features)
                conditions.append(projected * content_strength)
        
        # Encode style images
        if style_images and self.style_encoder:
            for img in style_images:
                features = self.style_encoder.encode(img)
                projected = self.projection(features)
                conditions.append(projected * style_strength)
        
        if conditions:
            # Combine all conditions
            return torch.stack(conditions).mean(dim=0, keepdim=True)
        else:
            # Return zero conditioning if no images provided
            return torch.zeros(1, self.config.projection_dim, device=self.device)
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        content_images: Optional[List[Image.Image]] = None,
        style_images: Optional[List[Image.Image]] = None,
        negative_prompt: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        content_strength: float = 1.0,
        style_strength: float = 1.0,
        generator: Optional[torch.Generator] = None,
        **kwargs
    ) -> Image.Image:
        """Generate image with USO conditioning"""
        
        # Use config defaults if not specified
        height = height or self.config.height
        width = width or self.config.width
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        guidance_scale = guidance_scale or self.config.guidance_scale
        
        # Encode text prompt
        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt)
        
        # Prepare image conditioning
        image_conditioning = self.prepare_image_conditioning(
            content_images=content_images,
            style_images=style_images,
            content_strength=content_strength,
            style_strength=style_strength
        )
        
        # Inject image conditioning into prompt embeddings
        # This is a simplified version - actual implementation may differ
        if image_conditioning is not None:
            # Add image conditioning to pooled embeddings
            pooled_prompt_embeds = pooled_prompt_embeds + image_conditioning
        
        # Generate with modified FLUX pipeline
        image = self.flux(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **kwargs
        ).images[0]
        
        return image


def create_uso_pipeline(
    flux_model_path: str = "black-forest-labs/FLUX.1-dev",
    lora_path: Optional[str] = None,
    projection_path: Optional[str] = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    use_safetensors: bool = True
) -> USOPipeline:
    """Create a complete USO pipeline"""
    
    device = torch.device(device)
    
    # Load FLUX pipeline
    flux = FluxPipeline.from_pretrained(
        flux_model_path,
        torch_dtype=torch_dtype,
        use_safetensors=use_safetensors
    ).to(device)
    
    # Load and apply LoRA if provided
    if lora_path and os.path.exists(lora_path):
        flux.load_lora_weights(lora_path)
    
    # Load projection model
    projection = None
    if projection_path and os.path.exists(projection_path):
        projection = ProjectionModel.from_pretrained(projection_path, device)
    
    # Create image encoders
    config = USOConfig(torch_dtype=torch_dtype)
    content_encoder = ImageEncoder(config.content_encoder_path, device)
    style_encoder = ImageEncoder(config.style_encoder_path, device)
    
    # Create pipeline
    pipeline = USOPipeline(
        flux_model=flux,
        projection_model=projection,
        content_encoder=content_encoder,
        style_encoder=style_encoder,
        device=device,
        config=config
    )
    
    return pipeline


def convert_comfy_image_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI image tensor to PIL Image"""
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    
    # ComfyUI images are in [H, W, C] format with values in [0, 1]
    image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(image_np)


def convert_pil_to_comfy_image(pil_image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI image tensor"""
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np)
    
    # Add batch dimension if needed
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

