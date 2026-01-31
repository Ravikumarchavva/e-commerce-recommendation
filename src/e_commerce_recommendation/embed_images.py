import asyncpg
import asyncio
from vllm import LLM, EmbeddingRequestOutput
from PIL import Image
import cv2
from tqdm import tqdm
from asyncio import Queue
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional


class ProductEmbedder:
    def __init__(self):
        # ---------------- PERFORMANCE CONFIG ----------------
        self.GPU_BATCH_SIZE = 1000
        self.IMAGE_LOAD_BATCH_SIZE = 1000
        self.PREFETCH_QUEUE_SIZE = 2000
        self.IMAGE_WORKERS = 256
        self.WRITE_BUFFER_SIZE = 2000

        # ---------------- SQL ----------------
        self.COUNT_SQL = "SELECT COUNT(*) FROM products WHERE embedding IS NULL"

        self.FETCH_SQL = """
            SELECT id, image_path, title
            FROM products
            WHERE embedding IS NULL
            ORDER BY id
        """

        self.BULK_UPDATE_SQL = """
            UPDATE products AS p
            SET embedding = v.embedding::vector
            FROM (
                SELECT unnest($1::bigint[]) AS id,
                       unnest($2::vector[]) AS embedding
            ) AS v
            WHERE p.id = v.id
        """

    # ---------------------------------------------------------
    @classmethod
    async def create(cls, database_url: str):
        self = cls()

        self.pool = await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=5,
            command_timeout=300
        )

        self.llm = LLM(
            model="Qwen/Qwen3-VL-Embedding-2B",
            runner="pooling",
            quantization="mxfp4",
            max_model_len=768,
            limit_mm_per_prompt={"image": 1},
            gpu_memory_utilization=0.90,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            enforce_eager=False,
        )

        print("✓ vLLM initialized (single GPU, large batch mode)")
        return self

    # ---------------------------------------------------------
    @staticmethod
    def build_request(image: Image.Image, text: str = "") -> dict:
        instruction = (
            "Represent the user's input for multi-modal ecommerce "
            "product recommendation. Focus on product details."
        )
        vision = "<|vision_start|><|image_pad|><|vision_end|>"

        prompt = (
            f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\n{vision}{text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        return {
            "prompt": prompt,
            "multi_modal_data": {"image": image}
        }

    # ---------------------------------------------------------
    @staticmethod
    def load_image(path: str) -> Optional[Image.Image]:
        try:
            img = cv2.imread(path)
            if img is None:
                return None

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            if max(h, w) > 512:
                s = 512 / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)))

            return Image.fromarray(img)
        except Exception:
            return None

    # ---------------------------------------------------------
    def _embed_batch_sync(self, requests: List[dict]) -> List[list]:
        """Synchronous GPU embedding - runs in thread executor."""
        outputs: List[EmbeddingRequestOutput] = self.llm.embed(
            requests,
            use_tqdm=False
        )
        return [o.outputs.embedding for o in outputs]

    # ---------------------------------------------------------
    async def producer(
        self,
        input_queue: Queue,
        executor: ThreadPoolExecutor
    ):
        loop = asyncio.get_running_loop()

        async with self.pool.acquire() as conn:
            async with conn.transaction(readonly=True):
                batch = []

                async for row in conn.cursor(self.FETCH_SQL):
                    batch.append(row)

                    if len(batch) >= self.IMAGE_LOAD_BATCH_SIZE:
                        await self._process_batch(batch, input_queue, loop, executor)
                        batch.clear()

                if batch:
                    await self._process_batch(batch, input_queue, loop, executor)

        await input_queue.put(None)

    # ---------------------------------------------------------
    async def _process_batch(
        self,
        rows,
        input_queue: Queue,
        loop,
        executor
    ):
        tasks = [
            loop.run_in_executor(executor, self.load_image, r["image_path"])
            for r in rows
        ]
        images = await asyncio.gather(*tasks)

        for row, img in zip(rows, images):
            if img is None:
                continue
            req = self.build_request(img, row["title"] or "")
            await input_queue.put((row["id"], req))

    # ---------------------------------------------------------
    async def embed_and_update(
        self,
        input_queue: Queue,
        pbar: tqdm,
        executor: ThreadPoolExecutor
    ):
        """Consumer: collect batches, embed on GPU, write to DB."""
        loop = asyncio.get_running_loop()
        buffer_ids = []
        buffer_vecs = []

        while True:
            # Collect batch
            batch = []
            sentinel_received = False
            
            while len(batch) < self.GPU_BATCH_SIZE:
                item = await input_queue.get()
                if item is None:
                    sentinel_received = True
                    break  # Stop collecting, process what we have
                batch.append(item)

            # If no items collected and we got sentinel, we're done
            if not batch and sentinel_received:
                break

            # Process the batch (even if it's smaller than GPU_BATCH_SIZE)
            if batch:
                ids, requests = zip(*batch)

                # 🔥 Run GPU embedding in thread executor (non-blocking)
                vectors = await loop.run_in_executor(
                    executor,
                    self._embed_batch_sync,
                    list(requests)
                )

                buffer_ids.extend(ids)
                buffer_vecs.extend(vectors)

                # Write to DB when buffer is full
                if len(buffer_ids) >= self.WRITE_BUFFER_SIZE:
                    async with self.pool.acquire() as conn:
                        await conn.execute(
                            self.BULK_UPDATE_SQL,
                            buffer_ids,
                            [str(v) for v in buffer_vecs]
                        )
                    pbar.update(len(buffer_ids))
                    buffer_ids.clear()
                    buffer_vecs.clear()

            # If we got sentinel, we're done after processing this batch
            if sentinel_received:
                break

        # Final flush
        if buffer_ids:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    self.BULK_UPDATE_SQL,
                    buffer_ids,
                    [str(v) for v in buffer_vecs]
                )
            pbar.update(len(buffer_ids))

    # ---------------------------------------------------------
    async def embed_products(self):
        async with self.pool.acquire() as conn:
            total = await conn.fetchval(self.COUNT_SQL)

        print(f"Found {total} products to embed")

        queue = Queue(maxsize=self.PREFETCH_QUEUE_SIZE)
        # Executor for BOTH image loading AND GPU embedding
        executor = ThreadPoolExecutor(max_workers=self.IMAGE_WORKERS + 1)
        pbar = tqdm(total=total, unit="product", desc="Embedding")

        try:
            await asyncio.gather(
                self.producer(queue, executor),
                self.embed_and_update(queue, pbar, executor)
            )
            print("\n✓ Embedding completed successfully")
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            executor.shutdown(wait=True)
            pbar.close()


# ---------------------------------------------------------
async def main():
    from e_commerce_recommendation.configs.settings import Settings
    settings = Settings()

    embedder = await ProductEmbedder.create(settings.DATABASE_URL)
    try:
        await embedder.embed_products()
    finally:
        await embedder.pool.close()


if __name__ == "__main__":
    asyncio.run(main())