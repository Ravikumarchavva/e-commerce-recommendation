import os
import asyncio
import asyncpg
from PIL import Image
from vllm import LLM
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from e_commerce_recommendation.configs.settings import Settings
import cv2

# ---------------------------
# CONFIG
# ---------------------------
settings = Settings()
DATABASE_URL = settings.DATABASE_URL

GPU_BATCH_SIZE = 128          # Conservative for vision encoder memory
DB_FETCH_SIZE = 1024           # Smaller DB chunks for streaming
PREFETCH_QUEUE_SIZE = 128*4      # Deep queue - GPU never starves
WRITE_BUFFER_SIZE = 128*32      # Bulk write every N embeddings
IMAGE_WORKERS = 128            # More parallel image loading
NUM_CONSUMERS = 4

# ---------------------------
# SQL
# ---------------------------
COUNT_SQL = "SELECT COUNT(*) FROM products WHERE embedding IS NULL"

FETCH_SQL = """
SELECT id, image_path, title
FROM products
WHERE embedding IS NULL
ORDER BY id
LIMIT $1
"""

BULK_UPDATE_SQL = """
UPDATE products AS p SET
    embedding = v.embedding::vector
FROM (SELECT unnest($1::bigint[]) as id, 
             unnest($2::vector[]) as embedding) AS v
WHERE p.id = v.id
"""

# ---------------------------
# PGVECTOR CONVERTER
# ---------------------------
def to_pgvector(vec):
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

# ---------------------------
# IMAGE-ONLY PROMPT (OFFICIAL QWEN FORMAT)
# ---------------------------
def build_prompt_string(image):
    """Build raw prompt string in official Qwen3-VL format."""
    default_instruction = "Represent the user's input."
    image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    prompt = f"<|im_start|>system\n{default_instruction}<|im_end|>\n<|im_start|>user\n{image_placeholder}<|im_end|>\n<|im_start|>assistant\n"
    return prompt

# ---------------------------
# IMAGE LOADER (CPU-bound, runs in thread pool)
# ---------------------------
def load_image_sync(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if max(h, w) > 512:
        scale = 512 / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    return Image.fromarray(img)

# ---------------------------
# PRODUCER: Fetch from DB + Load Images (async)
# ---------------------------
async def producer(pool, input_queue: asyncio.Queue, total_count: int, executor: ThreadPoolExecutor):
    """Streams products from DB, loads images on-the-fly, keeps GPU fed."""
    loop = asyncio.get_event_loop()
    batch_buffer = []
    products_queued = 0
    
    while products_queued < total_count:
        # Keep fetching WHERE embedding IS NULL (no OFFSET - rows disappear as we update)
        async with pool.acquire() as conn:
            rows = await conn.fetch(FETCH_SQL, DB_FETCH_SIZE)
        
        if not rows:
            break
        
        # Stream-load images one by one
        for row in rows:
            # Load image async
            img = await loop.run_in_executor(executor, load_image_sync, row["image_path"])
            
            if img is not None:
                batch_buffer.append((row["id"], img, row["title"]))
                products_queued += 1
                
                # Push batch as soon as we have GPU_BATCH_SIZE items
                if len(batch_buffer) >= GPU_BATCH_SIZE:
                    await input_queue.put(batch_buffer[:GPU_BATCH_SIZE])
                    batch_buffer = batch_buffer[GPU_BATCH_SIZE:]
    
    # Push remaining items
    if batch_buffer:
        await input_queue.put(batch_buffer)
    # Signal completion
    await input_queue.put(None)

# ---------------------------
# CONSUMER: GPU Embedding (sync, runs in thread)
# ---------------------------
def embed_batch_sync(batch, llm: LLM):
    """Run GPU embedding synchronously using official Qwen3-VL format."""
    ids, images, titles = zip(*batch)
    
    # Build prompts and multi-modal data in official format
    prompt_str = build_prompt_string(None)  # Same prompt for all images
    
    inputs = []
    for img in images:
        inputs.append({
            "prompt": prompt_str,
            "multi_modal_data": {"image": img}
        })
    
    outputs = llm.embed(inputs, use_tqdm=False)
    embeddings = [o.outputs.embedding for o in outputs]
    
    return list(zip(ids, embeddings))

# ---------------------------
# CONSUMER: GPU Worker (async wrapper)
# ---------------------------
async def consumer(input_queue: asyncio.Queue, output_queue: asyncio.Queue, llm: LLM, executor: ThreadPoolExecutor):
    """Consumes preprocessed batches, runs GPU embedding, pushes results."""
    loop = asyncio.get_event_loop()
    
    while True:
        batch = await input_queue.get()
        
        if batch is None:  # Shutdown signal
            await output_queue.put(None)
            break
        
        # Run GPU embedding in thread (vLLM releases GIL)
        results = await loop.run_in_executor(executor, embed_batch_sync, batch, llm)
        await output_queue.put(results)

# ---------------------------
# WRITER: Bulk Write to DB
# ---------------------------
async def writer(pool, output_queue: asyncio.Queue, pbar: tqdm):
    """Buffers results and writes in bulk to Postgres."""
    buffer_ids = []
    buffer_embeddings = []
    
    while True:
        results = await output_queue.get()
        
        if results is None:  # Shutdown signal
            # Flush remaining buffer
            if buffer_ids:
                async with pool.acquire() as conn:
                    await conn.execute(
                        BULK_UPDATE_SQL,
                        buffer_ids,
                        [to_pgvector(emb) for emb in buffer_embeddings]
                    )
                pbar.update(len(buffer_ids))
            break
        
        # Add to buffer
        for pid, embedding in results:
            buffer_ids.append(pid)
            buffer_embeddings.append(embedding)
        
        # Bulk write when buffer is full
        if len(buffer_ids) >= WRITE_BUFFER_SIZE:
            async with pool.acquire() as conn:
                await conn.execute(
                    BULK_UPDATE_SQL,
                    buffer_ids,
                    [to_pgvector(emb) for emb in buffer_embeddings]
                )
            pbar.update(len(buffer_ids))
            buffer_ids.clear()
            buffer_embeddings.clear()

# ---------------------------
# ORCHESTRATOR
# ---------------------------
async def embed_all_products(pool, llm: LLM):
    # Get total count
    async with pool.acquire() as conn:
        total = await conn.fetchval(COUNT_SQL)
    
    if total == 0:
        print("✓ Nothing left to embed.")
        return
    
    print(f"\n🚀 Embedding {total:,} products with optimized pipeline\n")
    print(f"   GPU Batch: {GPU_BATCH_SIZE} | Prefetch: {PREFETCH_QUEUE_SIZE} | Image Workers: {IMAGE_WORKERS}\n")
    
    # Create queues
    input_queue = asyncio.Queue(maxsize=PREFETCH_QUEUE_SIZE)   # Prefetch limit
    output_queue = asyncio.Queue(maxsize=4)  # Write buffer
    
    # Thread pool for CPU-bound tasks
    executor = ThreadPoolExecutor(max_workers=IMAGE_WORKERS + 1)  # +1 for GPU
    
    # Progress bar
    pbar = tqdm(total=total, unit="products", smoothing=0.1)
    
    # Launch pipeline
    prod_task = asyncio.create_task(producer(pool, input_queue, total, executor))
    cons_tasks = [asyncio.create_task(consumer(input_queue, output_queue, llm, executor)) for _ in range(NUM_CONSUMERS)]
    write_task = asyncio.create_task(writer(pool, output_queue, pbar))
    
    # Wait for completion - this will raise if any task fails
    try:
        await asyncio.gather(prod_task, *cons_tasks, write_task)
        pbar.close()
        executor.shutdown(wait=True)
        print("\n✓ All products embedded successfully. GPU should show steady utilization.")
    except Exception as e:
        print(f"\n✗ Embedding failed with error: {e}")
        pbar.close()
        executor.shutdown(wait=True)
        raise

# ---------------------------
# BOOTSTRAP
# ---------------------------
async def main():
    pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=2,
        max_size=5,
        command_timeout=60
    )
    
    llm = LLM(
        model="Qwen/Qwen3-VL-Embedding-2B",
        runner="pooling",
        max_model_len=1024,        # Match official example
        quantization="mxfp4",
        gpu_memory_utilization=0.90,
        limit_mm_per_prompt={"image": 1},  # Official format
        enforce_eager=False
    )
    
    print(f"\n🖼️  IMAGE-ONLY EMBEDDING MODE")
    print(f"   Batch Size: {GPU_BATCH_SIZE} | Workers: {IMAGE_WORKERS}")
    print(f"   Max Token Length: 2048 (image-only)\n")
    
    await embed_all_products(pool, llm)
    await pool.close()

# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    asyncio.run(main())
