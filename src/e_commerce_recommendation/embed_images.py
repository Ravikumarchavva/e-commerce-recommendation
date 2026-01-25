import asyncpg
import asyncio
from vllm import LLM, EmbeddingRequestOutput
from PIL import Image
from tqdm import tqdm
from asyncio import Queue
from concurrent.futures import ThreadPoolExecutor

class ProductEmbedder:
    """Class to handle embedding of product images and titles using Qwen3-VL-Embedding-2B model."""
    def __init__(self):
        self.GPU_BATCH_SIZE = 128
        self.PREFETCH_QUEUE_SIZE = 256
        self.IMAGE_WORKERS = 16
        self.DB_FETCH_SIZE = 512
        self.COUNT_SQL = "SELECT COUNT(*) FROM products WHERE embedding IS NULL"

        self.FETCH_SQL = """
                            SELECT id, image_path, title
                            FROM products
                            WHERE embedding IS NULL
                            ORDER BY id
                            LIMIT $1
                        """

        self.BULK_UPDATE_SQL = """
                                UPDATE products AS p 
                                SET embedding = v.embedding::vector
                                FROM (SELECT unnest($1::bigint[]) as id, 
                                            unnest($2::vector[]) as embedding) AS v
                                WHERE p.id = v.id
                              """

    @classmethod
    async def create(cls, database_url):
        self = cls()
        self.pool= await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=5,
            command_timeout=60
        )
        self.llm = LLM(
            model="Qwen/Qwen3-VL-Embedding-2B",
            runner="pooling",
            quantization="mxfp4",
            max_model_len=600,
            limit_mm_per_prompt={"image": 1},
            gpu_memory_utilization=0.90,
            enforce_eager=False
        )
        return self

    @staticmethod
    def build_request(image, text=""):
        """Build request dict in official Qwen3-VL format."""
        default_instruction = "Represent the user's input for multi-modal ecommerce products recommendation. focus more on product's detailing"
        vision_token = "<|vision_start|><|image_pad|><|vision_end|>"
        if text:
            prompt = f"<|im_start|>system\n{default_instruction}<|im_end|>\n<|im_start|>user\n{vision_token}{text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"<|im_start|>system\n{default_instruction}<|im_end|>\n<|im_start|>user\n{vision_token}<|im_end|>\n<|im_start|>assistant\n"
        
        return {
            "prompt": prompt,
            "multi_modal_data": {"image": image}
        }
    
    @staticmethod
    def load_image(image_path):
        """Load and preprocess image from URL or file path."""
        try:
            image = Image.open(image_path).convert("RGB")
            MAX_SIZE = 672
            if max(image.size) > MAX_SIZE:
                image.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    async def producer(self, pool: asyncpg.Pool, input_queue: Queue, total: int, executor: ThreadPoolExecutor):
        """Fetch products from DB and enqueue for processing."""
        async with pool.acquire(timeout=60) as conn:
            if isinstance(conn, asyncpg.Connection):
                async with conn.transaction(readonly=True):
                    stmt = await conn.prepare(self.FETCH_SQL)
                    async for record in stmt.cursor(total, prefetch=self.PREFETCH_QUEUE_SIZE):
                        product_id, image_path, title = record
                        image = await asyncio.get_event_loop().run_in_executor(executor, self.load_image, image_path)
                        if not image:
                            continue
                        
                        request = self.build_request(image, text=title or "")
                        await input_queue.put((product_id, request))
            else:
                raise RuntimeError("Failed to acquire a valid database connection.")
        
        # Signal completion
        await input_queue.put(None)
            
    async def consumer(self, input_queue: Queue, output_queue: Queue, llm: LLM, pbar: tqdm):
        """Consume products from input queue, generate embeddings, and enqueue results."""
        loop = asyncio.get_event_loop()
        
        while True:
            # Collect batch
            batch = []
            for _ in range(self.GPU_BATCH_SIZE):
                try:
                    item = await asyncio.wait_for(input_queue.get(), timeout=1.0)
                    if item is None:
                        break
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
            
            # Check if we should stop
            if not batch:
                await output_queue.put(None)
                break
            
            # Unpack batch
            product_ids, requests = zip(*batch)
            
            # Embed with official format
            embeddings: list[EmbeddingRequestOutput] = await loop.run_in_executor(
                None,
                lambda: llm.embed(list(requests), use_tqdm=False)
            )
            embeddings = [o.outputs.embedding for o in embeddings]
            
            # Update progress
            pbar.update(len(batch))
            
            await output_queue.put((product_ids, embeddings))

    async def updater(self, pool: asyncpg.Pool, output_queue: Queue):
        """Update product embeddings in the database."""
        async with pool.acquire(timeout=60) as conn:
            if isinstance(conn, asyncpg.Connection):
                while True:
                    item = await output_queue.get()
                    if item is None:
                        break
                    product_ids, embeddings = item
                    # Convert embeddings to PostgreSQL vector format strings
                    embedding_strs = [str(emb) for emb in embeddings]
                    await conn.execute(self.BULK_UPDATE_SQL, product_ids, embedding_strs)
            else:
                raise RuntimeError("Failed to acquire a valid database connection.")

    async def embed_products(self):
        """Main method to embed products using producer-consumer pattern."""
        total_count = 0
        async with self.pool.acquire(timeout=60) as conn:
            if isinstance(conn, asyncpg.Connection):
                total_count = await conn.fetchval(self.COUNT_SQL)
            else:
                raise RuntimeError("Failed to acquire a valid database connection.")

        input_queue = asyncio.Queue(maxsize=self.PREFETCH_QUEUE_SIZE)
        output_queue = asyncio.Queue(maxsize=self.PREFETCH_QUEUE_SIZE)

        executor = ThreadPoolExecutor(max_workers=self.IMAGE_WORKERS + 1)

        p_bar = tqdm(total=total_count, desc="Embedding Products", unit="product", smoothing=0.1)

        producer_coro = self.producer(self.pool, input_queue, total_count, executor)
        consumer_coro = self.consumer(input_queue, output_queue, self.llm, p_bar)
        updater_coro = self.updater(self.pool, output_queue)

        try:
            await asyncio.gather(producer_coro, consumer_coro, updater_coro)
            executor.shutdown(wait=True)
            print("Embedding completed successfully.")
        except Exception as e:
            print(f"Error during embedding process: {e}")
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=True)
            p_bar.close()


async def main():
    from e_commerce_recommendation.configs.settings import Settings
    settings = Settings()
    DATABASE_URL = settings.SAMPLE_DATABASE_URL
    embedder = await ProductEmbedder.create(DATABASE_URL)
    await embedder.embed_products()


if __name__ == "__main__" :
    asyncio.run(main())