# USO ComfyUI Node Requirements
# These are the dependencies needed for the USO custom node

# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
Pillow>=9.4.0
safetensors>=0.4.0

# Model downloading
huggingface_hub>=0.20.0

# Image processing and encoding
transformers>=4.36.0  # For T5 text encoder
diffusers>=0.25.0  # For FLUX components
accelerate>=0.25.0  # For model optimization
sentencepiece>=0.1.99  # For T5 tokenizer

# Optional but recommended
opencv-python>=4.8.0  # For advanced image processing
scipy>=1.10.0  # For numerical operations
einops>=0.7.0  # For tensor operations

# CLIP/SigLIP for image encoding
open_clip_torch>=2.20.0  # For SigLIP vision encoder
ftfy>=6.1.0  # For text cleaning
regex>=2023.10.3  # For text processing

# Memory optimization (optional)
xformers>=0.0.22  # For memory-efficient attention (if CUDA available)
bitsandbytes>=0.41.0  # For 8-bit optimization (optional)