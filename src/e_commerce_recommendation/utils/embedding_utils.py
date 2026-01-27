from PIL import Image
from vllm import LLM
import asyncpg
from functools import lru_cache
from typing import List, Optional

def load_model() -> LLM:
    llm = LLM(
            model="Qwen/Qwen3-VL-Embedding-2B",
            runner="pooling",
            quantization="mxfp4",
            max_model_len=600,
            limit_mm_per_prompt={"image": 1},
            gpu_memory_utilization=0.80,
            enforce_eager=False
        )
    return llm

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

def load_image_sync(image: Image.Image):
    """Load and preprocess image on CPU thread."""
    try:
        img = image.convert("RGB")
        # Pre-resize to speed up processing
        max_size = 1024
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return img
    except Exception:
        return None

def embed_text(llm: LLM, text: str) -> list[float]:
    """
    Generate embedding for text query - exactly like embed_images.py
    
    Args:
        text: Search query text
        
    Returns:
        Embedding vector as list of floats
    """
    # Build request in exact format from embed_images.py
    request = build_request(image=None, text=text)
    
    # Generate embedding
    outputs = llm.embed([request], use_tqdm=False)
    embedding = outputs[0].outputs.embedding
    
    return embedding

def embed_image(llm: LLM, image: Image.Image) -> list[float]:
    """
    Generate embedding for image
    
    Args:
        image: PIL Image object or path to image
        
    Returns:
        Embedding vector as list of floats
    """
    image = load_image_sync(image)    
    # Build request in exact format from embed_images.py
    request = build_request(image=image, text="")
    
    # Generate embedding
    outputs = llm.embed([request], use_tqdm=False)
    embedding = outputs[0].outputs.embedding
    
    return embedding

def embed_multimodal(llm: LLM, image: Image.Image, text: str) -> list[float]:
    """
    Generate embedding for image and text
    """
    image = load_image_sync(image)    
    # Build request in exact format from embed_images.py
    request = build_request(image=image, text=text)
    
    # Generate embedding
    outputs = llm.embed([request], use_tqdm=False)
    embedding = outputs[0].outputs.embedding
    
    return embedding

async def get_similar_categories(pool: asyncpg.Pool, embedding: list[float], top_k: int = 10) -> list[int]:
    """
    Get similar categories using vector similarity search
    """
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id
            FROM category_embeddings
            ORDER BY embedding <=> $1
            LIMIT $2
            """,
            str(embedding),
            top_k
        )
    
    return [row["id"] for row in rows]

async def get_similar_products(pool: asyncpg.Pool, embedding: list[float], categories: list[int], top_k: int = 10, offset: int = 0) -> list[int]:
    """
    Get similar products using vector similarity search
    """
    
    if categories:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id
                FROM products
                WHERE labels && $1::smallint[]
                ORDER BY embedding <=> $2
                LIMIT $3 OFFSET $4
                """,
                categories,
                str(embedding),
                top_k,
                offset
            )
        return [row["id"] for row in rows]
    else:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id
                FROM products
                ORDER BY embedding <=> $1
                LIMIT $2 OFFSET $3
                """,
                str(embedding),
                top_k,
                offset
            )
        return [row["id"] for row in rows]
