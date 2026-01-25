"""
Embedding utilities matching the exact approach from embed_images.py
"""
from PIL import Image
from vllm import LLM
from typing import Union


class EmbeddingModel:
    """Singleton embedding model using factory pattern from embed_images.py"""
    _instance = None
    
    def __init__(self):
        self.llm = None
    
    @classmethod
    def create(cls):
        """Factory method to create and initialize the embedding model."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.llm = LLM(
                model="Qwen/Qwen3-VL-Embedding-2B",
                runner="pooling",
                quantization="mxfp4",
                max_model_len=600,
                limit_mm_per_prompt={"image": 1},
                gpu_memory_utilization=0.80,
                enforce_eager=False
            )
            print("✅ Qwen3-VL-Embedding model loaded with vLLM")
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        """Get existing instance or create new one."""
        if cls._instance is None:
            return cls.create()
        return cls._instance


def load_model():
    """Load the Qwen3-VL-Embedding model - uses factory pattern."""
    return EmbeddingModel.create()


def build_request(image=None, text=""):
    """Build request dict in official Qwen3-VL format - matching embed_images.py exactly."""
    default_instruction = "Represent the user's input for multi-modal ecommerce products recommendation. focus more on product's detailing"
    vision_token = "<|vision_start|><|image_pad|><|vision_end|>"
    
    if image and text:
        # Multimodal: image + text
        prompt = f"<|im_start|>system\n{default_instruction}<|im_end|>\n<|im_start|>user\n{vision_token}{text}<|im_end|>\n<|im_start|>assistant\n"
        return {
            "prompt": prompt,
            "multi_modal_data": {"image": image}
        }
    elif image:
        # Image only
        prompt = f"<|im_start|>system\n{default_instruction}<|im_end|>\n<|im_start|>user\n{vision_token}<|im_end|>\n<|im_start|>assistant\n"
        return {
            "prompt": prompt,
            "multi_modal_data": {"image": image}
        }
    else:
        # Text only
        prompt = f"<|im_start|>system\n{default_instruction}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        return {
            "prompt": prompt,
            "multi_modal_data": None
        }


def embed_text(text: str) -> list[float]:
    """
    Generate embedding for text query - exactly like embed_images.py
    
    Args:
        text: Search query text
        
    Returns:
        Embedding vector as list of floats
    """
    model = EmbeddingModel.get_instance()
    
    # Build request in exact format from embed_images.py
    request = build_request(image=None, text=text)
    
    # Generate embedding
    outputs = model.llm.embed([request], use_tqdm=False)
    embedding = outputs[0].outputs.embedding
    
    return embedding


def embed_image(image: Union[Image.Image, str]) -> list[float]:
    """
    Generate embedding for image - exactly like embed_images.py
    
    Args:
        image: PIL Image object or path to image
        
    Returns:
        Embedding vector as list of floats
    """
    model = EmbeddingModel.get_instance()
    
    # Load image if path provided
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
        # Resize if needed
        MAX_SIZE = 672
        if max(image.size) > MAX_SIZE:
            image.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)
    
    # Build request in exact format from embed_images.py
    request = build_request(image=image, text="")
    
    # Generate embedding
    outputs = model.llm.embed([request], use_tqdm=False)
    embedding = outputs[0].outputs.embedding
    
    return embedding


def embed_multimodal(image: Union[Image.Image, str], text: str) -> list[float]:
    """
    Generate embedding for combined text + image query.
    
    Args:
        image: PIL Image object or path to image
        text: Search query text to combine with image
        
    Returns:
        Embedding vector as list of floats
    """
    model = EmbeddingModel.get_instance()
    
    # Load image if path provided
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
        # Resize if needed
        MAX_SIZE = 672
        if max(image.size) > MAX_SIZE:
            image.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)
    
    # Build request in exact format from embed_images.py
    request = build_request(image=image, text=text)
    
    # Generate embedding
    outputs = model.llm.embed([request], use_tqdm=False)
    embedding = outputs[0].outputs.embedding
    
    return embedding
