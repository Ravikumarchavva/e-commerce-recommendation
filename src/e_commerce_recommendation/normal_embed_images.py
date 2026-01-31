import asyncio
import asyncpg
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from vllm import LLM
import cv2
from typing import Optional

from e_commerce_recommendation.configs.settings import Settings

# ---------------------------
# CONFIG
# ---------------------------
settings = Settings()
DATABASE_URL = settings.DATABASE_URL

GPU_BATCH_SIZE = 20_000
DB_FETCH = 200_000
IMAGE_WORKERS = 128
QUEUE_MAXSIZE = 40_000   # backpressure

# ---------------------------
# SQL
# ---------------------------
COUNT_SQL = "SELECT COUNT(*) FROM products WHERE embedding IS NULL"

FETCH_SQL = """
SELECT id, image_path, title
FROM products
WHERE embedding IS NULL
FOR UPDATE SKIP LOCKED
LIMIT $1
"""

UPDATE_SQL = """
UPDATE products
SET embedding = $2::vector
WHERE id = $1
"""

# ---------------------------
# PROMPT
# ---------------------------
def build_request(image, text):
    instruction = (
        "Represent the user's input for multi-modal ecommerce products "
        "recommendation. Focus on product detailing."
    )
    vision = "<|vision_start|><|image_pad|><|vision_end|>"

    prompt = (
        "<|im_start|>system\n"
        f"{instruction}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{vision}{text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    return {
        "prompt": prompt,
        "multi_modal_data": {"image": image}
    }

# ---------------------------
# IMAGE LOADER (THREADPOOL)
# ---------------------------
def load_image(image_path: str) -> Optional[Image.Image]:
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        if max(h, w) > 512:
            scale = 512 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        return Image.fromarray(img)

    except Exception:
        return None

# ---------------------------
# EMBED WORKER (SINGLE CALLER)
# ---------------------------
async def embed_worker(llm, pool, queue: asyncio.Queue, pbar: tqdm):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break

        ids, images, titles = item

        prompts = [
            build_request(img, title)
            for img, title in zip(images, titles)
        ]

        # 🔴 SINGLE THREAD SAFE CALL
        outputs = llm.embed(prompts, use_tqdm=False)

        vectors = [o.outputs.embedding for o in outputs]

        async with pool.acquire() as conn:
            await conn.executemany(
                UPDATE_SQL,
                [(pid, str(vec)) for pid, vec in zip(ids, vectors)]
            )

        pbar.update(len(ids))
        queue.task_done()

# ---------------------------
# MAIN PIPELINE
# ---------------------------
async def embed_all_products(pool, llm):
    async with pool.acquire() as conn:
        total = await conn.fetchval(COUNT_SQL)

    if total == 0:
        print("Nothing to embed.")
        return

    print(f"\nEmbedding {total:,} products\n")
    pbar = tqdm(total=total, unit="img")

    queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    image_executor = ThreadPoolExecutor(max_workers=IMAGE_WORKERS)

    # Start EXACTLY ONE embed worker
    worker_task = asyncio.create_task(embed_worker(llm, pool, queue, pbar))

    while True:
        async with pool.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch(FETCH_SQL, DB_FETCH)

        if not rows:
            break

        loop = asyncio.get_running_loop()
        images = await asyncio.gather(*[
            loop.run_in_executor(image_executor, load_image, r["image_path"])
            for r in rows
        ])

        filtered = [
            (r["id"], img, r["title"])
            for r, img in zip(rows, images)
            if img is not None
        ]

        for i in range(0, len(filtered), GPU_BATCH_SIZE):
            batch = filtered[i:i + GPU_BATCH_SIZE]

            ids = [x[0] for x in batch]
            imgs = [x[1] for x in batch]
            titles = [x[2] for x in batch]

            await queue.put((ids, imgs, titles))

    # Shutdown
    await queue.put(None)
    await queue.join()
    await worker_task

    image_executor.shutdown()
    pbar.close()

    print("\nAll products embedded successfully.")

# ---------------------------
# BOOTSTRAP
# ---------------------------
async def main():
    pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=5,
        max_size=20
    )

    llm = LLM(
        model="Qwen/Qwen3-VL-Embedding-2B",
        runner="pooling",
        max_model_len=768,
        quantization="mxfp4",
        gpu_memory_utilization=0.90,
    )

    await embed_all_products(pool, llm)

# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    asyncio.run(main())
