"""
ComfyUI Custom Node for USO (Unified Style and Subject-Driven Generation)
Author: ComfyUI Community
License: Apache 2.0
"""

from .uso_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# ComfyUI will automatically load this when the custom node is installed
print("✨ USO Custom Node loaded successfully!")