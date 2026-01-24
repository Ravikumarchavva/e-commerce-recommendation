"""
Embedding utilities matching the exact approach from embed_images.py
"""
from PIL import Image
from vllm import LLM
from typing import Union

# Global LLM instance (loaded once at startup)
llm = None

def load_model():
    """Load the Qwen3-VL-Embedding model with vLLM once at startup."""
    global llm
    
    if llm is not None:
        return
    
    llm = LLM(
        model="Qwen/Qwen3-VL-Embedding-2B",
        runner="pooling",
        max_model_len=600,
        quantization="mxfp4",
        gpu_memory_utilization=0.8
    )
    
    print(f"✅ Qwen3-VL-Embedding model loaded with vLLM")


def build_text_prompt(text: str):
    """Build prompt for text embedding - matching embed_images.py pattern."""
    return [
        {
            "role": "system",
            "content": "You are a multimodal embedding model. Encode the search query into a semantic vector."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text}
            ]
        }
    ]


def build_image_prompt(image: Image.Image):
    """Build prompt for image embedding - should match embed_images.py pattern."""
    return [
        {
            "role": "system",
            "content": "You are a multimodal embedding model. Encode the product image and title into a single semantic vector."
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Product title: "}
            ]
        }
    ]


def build_multimodal_prompt(image: Image.Image, text: str):
    """Build prompt for combined text + image embedding."""
    return [
        {
            "role": "system",
            "content": "You are a multimodal embedding model. Encode the product image and text query into a single semantic vector."
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text}
            ]
        }
    ]


def embed_text(text: str) -> list[float]:
    """
    Generate embedding for text query - exactly like embed_images.py
    
    Args:
        text: Search query text
        
    Returns:
        Embedding vector as list of floats
    """
    if llm is None:
        load_model()
    
    # Build conversation
    conversation = build_text_prompt(text)
    
    # Apply chat template (exactly like embed_images.py)
    prompt = llm.llm_engine.tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate embedding
    outputs = llm.embed([prompt], use_tqdm=False)
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
    if llm is None:
        load_model()
    
    # Load image if path provided
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    # Build conversation (exactly like embed_images.py)
    conversation = build_image_prompt(image)
    
    # Apply chat template (exactly like embed_images.py)
    prompt = llm.llm_engine.tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate embedding
    outputs = llm.embed([prompt], use_tqdm=False)
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
    if llm is None:
        load_model()
    
    # Load image if path provided
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    # Build multimodal conversation
    conversation = build_multimodal_prompt(image, text)
    
    # Apply chat template
    prompt = llm.llm_engine.tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate embedding
    outputs = llm.embed([prompt], use_tqdm=False)
    embedding = outputs[0].outputs.embedding
    
    return embedding
