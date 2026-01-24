import os
import asyncpg
from PIL import Image
from vllm import LLM
from tqdm import tqdm
from e_commerce_recommendation.configs.settings import Settings

# ---------------------------
# CONFIG
# ---------------------------
settings = Settings()
DATABASE_URL = settings.DATABASE_URL

BATCH_SIZE = 240     # GPU batch
DB_BATCH = 960     # DB fetch batch

# ---------------------------
# SQL
# ---------------------------
COUNT_SQL = "SELECT COUNT(*) FROM products WHERE embedding IS NULL"

FETCH_SQL = """
SELECT id, image_path, title
FROM products
WHERE embedding IS NULL
LIMIT $1
"""

# ---------------------------
# PGVECTOR CONVERTER
# ---------------------------
def to_pgvector(vec):
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

# ---------------------------
# MULTIMODAL PROMPT
# ---------------------------
def build_prompt(image, title):
    return [
        {
            "role": "system",
            "content": "You are a multimodal embedding model. Encode the product image and title into a single semantic vector."
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"Product title: {title}"}
            ]
        }
    ]

# ---------------------------
# BATCH EMBED
# ---------------------------
def embed_products_batch(images, titles, llm: LLM):
    prompts = []

    for img, title in zip(images, titles):
        conversation = build_prompt(img, title)
        prompt = llm.llm_engine.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)

    outputs = llm.embed(prompts, use_tqdm=False)
    return [o.outputs.embedding for o in outputs]

# ---------------------------
# MAIN LOOP
# ---------------------------
async def embed_all_products(pool, llm: LLM):

    async with pool.acquire() as conn:
        total = await conn.fetchval(COUNT_SQL)

    if total == 0:
        print("Nothing left to embed.")
        return

    print(f"\nEmbedding {total:,} products\n")
    pbar = tqdm(total=total, unit="img")

    while True:
        async with pool.acquire() as conn:
            rows = await conn.fetch(FETCH_SQL, DB_BATCH)

        if not rows:
            break

        ids, images, titles = [], [], []

        for r in rows:
            try:
                img = Image.open(r["image_path"]).convert("RGB")
                images.append(img)
                titles.append(r["title"])
                ids.append(r["id"])
            except:
                continue

        # GPU batches
        for i in range(0, len(images), BATCH_SIZE):
            batch_imgs = images[i:i+BATCH_SIZE]
            batch_titles = titles[i:i+BATCH_SIZE]
            batch_ids = ids[i:i+BATCH_SIZE]

            vectors = embed_products_batch(batch_imgs, batch_titles, llm)

            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.executemany(
                        "UPDATE products SET embedding=$2::vector WHERE id=$1",
                        [(pid, to_pgvector(vec)) for pid, vec in zip(batch_ids, vectors)]
                    )

            pbar.update(len(batch_ids))

    pbar.close()
    print("\nAll products embedded successfully.")

# ---------------------------
# BOOTSTRAP
# ---------------------------
async def main():
    pool = await asyncpg.create_pool(DATABASE_URL)

    llm = LLM(
        model="Qwen/Qwen3-VL-Embedding-2B",
        runner="pooling",
        max_model_len=600,       # optimal for images
        quantization="mxfp4",
        gpu_memory_utilization=0.90
    )

    await embed_all_products(pool, llm)

# ---------------------------
# RUN
# ---------------------------
import asyncio
asyncio.run(main())
