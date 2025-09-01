# ComfyUI USO Custom Node

A ComfyUI custom node implementation of ByteDance's **USO (Unified Style and Subject-Driven Generation)** model, which enables advanced style transfer and subject preservation using FLUX.

## 🌟 Features

- **Subject-Driven Generation**: Preserve identity and characteristics from reference images
- **Style-Driven Generation**: Apply artistic styles from reference images
- **Combined Style & Subject**: Mix both style and subject conditioning
- **Multi-Style Blending**: Combine multiple style references
- **FLUX Integration**: Built on top of FLUX.1 for high-quality generation

## 📋 Requirements

- ComfyUI (latest version recommended)
- Python 3.10 - 3.12
- CUDA-capable GPU with 16GB+ VRAM recommended
- FLUX.1-dev model

## 🚀 Installation

### Method 1: Install via ComfyUI Manager (Recommended)

1. Install [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) if you haven't already
2. Open ComfyUI Manager in the UI
3. Search for "USO"
4. Click Install and restart ComfyUI

### Method 2: Manual Installation

1. Navigate to your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes/
```

2. Clone this repository:

```bash
git clone https://github.com/casterpollux/ComfyUI-USO
cd ComfyUI-USO
```

3. Install dependencies:

**For ComfyUI Portable:**

```bash
..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

**For regular Python environment:**

```bash
pip install -r requirements.txt
```

4. Download USO model files:

**Option A: Automatic download (recommended)**

The USO node will automatically download the required model files from the [ByteDance USO repository](https://github.com/bytedance/USO) on first use. No manual intervention needed!

**Option B: Manual download (if automatic fails)**

```bash
# Download from HuggingFace
huggingface-cli download bytedance-research/USO --local-dir models/uso --local-dir-use-symlinks False
```

**Note**: You do **NOT** need to separately install or clone the [ByteDance USO repository](https://github.com/bytedance/USO). The ComfyUI node handles all model downloads automatically.

5. Restart ComfyUI

## 📁 Model Files

The USO node requires the following model files:

- `uso_flux_v1.0/dit_lora.safetensors` - LoRA weights for FLUX
- `uso_flux_v1.0/projector.safetensors` - Projection model for image encoding

These will be automatically downloaded to `ComfyUI/models/uso/` on first use, or you can manually place them there.

## 🎨 Usage

### Basic Workflow

1. **Load USO Model**
   - Add "USO Model Loader" node
   - Connect your FLUX model and VAE
   - Configure LoRA strength (default: 1.0)
2. **Encode Images**
   - Add "USO Image Encoder" node
   - Select mode:
       - `subject`: Identity preservation only
       - `style`: Style transfer only
       - `style_subject`: Both style and subject
       - `multi_style`: Blend multiple styles
   - Connect your reference images
3. **Generate**
   - Add "USO Sampler" node
   - Connect model, conditioning, and set your prompt
   - Configure generation parameters
4. **Decode**
   - Add "USO Decode" node to convert latents to images

### Example Workflows

#### Subject-Driven Generation

```
[Load Image] → [USO Image Encoder (mode: subject)] → [USO Sampler] → [USO Decode] → [Save Image]
```

#### Style Transfer

```
[Load Style Image] → [USO Image Encoder (mode: style)] → [USO Sampler] → [USO Decode] → [Save Image]
```

#### Combined Style & Subject

```
[Load Content Image] ─┐
                      ├→ [USO Image Encoder (mode: style_subject)] → [USO Sampler] → [USO Decode]
[Load Style Image] ───┘
```

## ⚙️ Node Parameters

### USO Model Loader

- **model**: FLUX model to use as base
- **vae**: VAE for encoding/decoding
- **lora_path**: Path to USO LoRA weights
- **projection_path**: Path to projection model
- **lora_strength**: Strength of LoRA application (0-2)

### USO Image Encoder

- **mode**: Generation mode selection
- **content_image**: Reference for subject/identity
- **style_image**: Reference for style
- **content_strength**: How strongly to apply content (0-2)
- **style_strength**: How strongly to apply style (0-2)

### USO Sampler

- **prompt**: Text description for generation
- **seed**: Random seed (-1 for random, 0+ for fixed)
- **steps**: Number of denoising steps (1-50, default: 25)
- **guidance**: Classifier-free guidance scale (1.0-5.0, default: 4.0)
- **width/height**: Output dimensions (512-1536, default: 1024)
- **content_reference_size**: Reference image processing size (default: 512)

## 🔧 Troubleshooting

### Out of Memory (OOM)

- Reduce image dimensions (try 768x768 or 512x512)
- Lower batch size to 1
- Enable CPU offloading in ComfyUI settings
- Use xformers for memory-efficient attention

### Models Not Found

- Ensure models are in `ComfyUI/models/uso/` directory
- Check file paths in the loader node
- Try automatic download by setting `auto_download` to true

### Poor Quality Results

- Ensure you're using FLUX.1-dev (not schnell)
- Adjust content/style strengths
- Try different guidance values (3.0-7.0)
- Experiment with step counts (20-50)

## 🎯 Tips for Best Results

1. **Image Preparation**
   - Use high-quality reference images
   - Crop subjects tightly for better identity preservation
   - Choose style images with clear, distinctive styles
2. **Prompt Engineering**
   - Be descriptive but not overly specific
   - Let the image conditioning guide details
   - Use negative prompts to avoid unwanted elements
3. **Parameter Tuning**
   - Start with default values
   - Adjust strengths incrementally (0.1 steps)
   - Higher guidance for more prompt adherence
   - Lower guidance for more creative freedom

## 📚 Technical Details

USO uses a disentangled learning approach to separate content and style:

- **Content Encoding**: Preserves identity and structure
- **Style Encoding**: Captures artistic style and aesthetics
- **Projection Model**: Maps image features to FLUX latent space
- **LoRA Adaptation**: Modifies FLUX attention for conditioning

## 🤝 Credits

- **USO Model**: [ByteDance Research](https://github.com/bytedance/USO) - Original USO implementation
- **Paper**: [arXiv:2508.18966](https://arxiv.org/abs/2508.18966) - "USO: Unified Style and Subject-Driven Generation via Disentangled and Reward Learning"
- **FLUX Base**: [Black Forest Labs](https://github.com/black-forest-labs/flux)
- **ComfyUI Integration**: [casterpollux](https://github.com/casterpollux) - This ComfyUI node implementation

## 📄 License

This ComfyUI node is released under Apache 2.0 License, matching the original USO model license.

## 🐛 Issues & Support

- Report issues on [GitHub Issues](https://github.com/casterpollux/ComfyUI-USO/issues)
- Join ComfyUI Discord for community support
- Check the [USO project page](https://bytedance.github.io/USO/) for model updates

## 🚧 Roadmap

- [ ]  Support for FLUX.1-schnell variant
- [ ]  Batch processing optimization
- [ ]  Advanced masking options
- [ ]  ControlNet integration
- [ ]  Training/fine-tuning support
- [ ]  Web UI for easier configuration

---

## 🔗 Related Projects

- **Official USO**: [ByteDance USO repository](https://github.com/bytedance/USO) - Original implementation with Gradio interface
- **USO Demo**: [bytedance.github.io/USO/](https://bytedance.github.io/USO/) - Online demo and project page

**Note**: This is an unofficial ComfyUI implementation that automatically downloads models from the official ByteDance USO repository. No separate installation of the original USO repo is required.
