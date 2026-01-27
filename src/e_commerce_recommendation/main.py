import asyncpg
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm import LLM
import torch
import gc
import contextlib

from e_commerce_recommendation.configs.settings import Settings
from e_commerce_recommendation.utils.sql_query_helper import filter_helper
from e_commerce_recommendation.utils.utils import format_product
from e_commerce_recommendation.utils.embedding_utils import (
    load_model, embed_text, get_similar_categories, get_similar_products
)

settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await asyncpg.create_pool(
        settings.SAMPLE_DATABASE_URL,
        min_size=5,
        max_size=20,
        command_timeout=60
    )
    app.state.llm: LLM = load_model()
    yield
    destroy_model_parallel()
    del app.state.llm
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    await app.state.pool.close()

app = FastAPI(
    title="E-Commerce Recommendation System",
    description="E-Commerce Recommendation System",
    version="1.0.0",
    lifespan=lifespan
)
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=settings.IMAGE_ROOT), name="images")


@app.get("/")
def read_root():
    # Get the directory of the current file
    return FileResponse(str(settings.BASE_DIR) + "/src/e_commerce_recommendation/index.html")


@app.get("/categories")
async def get_categories(country: str | None = Query(default=None)):
    async with app.state.pool.acquire() as conn:
        if country:
            query = """
                SELECT c.id, c.name
                FROM categories c
                WHERE c.id IN (
                    SELECT DISTINCT unnest(labels)
                    FROM products
                    WHERE country = $1
                )
                ORDER BY c.name
            """
            rows = await conn.fetch(query, country.lower())
        else:
            query = """
                SELECT id, name
                FROM categories
                ORDER BY name
            """
            rows = await conn.fetch(query)
    return {
        "total": len(rows),
        "categories": [{"id": r["id"], "name": r["name"]} for r in rows]
    }

@app.get("/browse")
async def browse(
    category: int | None = Query(default=None, description="Filter by Category ID"),
    country: str | None = Query(default=None, description="Filter by Country"),
    min_price: float | None = Query(default=None, description="Filter by Minimum Price"),
    max_price: float | None = Query(default=None, description="Filter by Maximum Price"),
    best_seller: bool | None = Query(default=None, description="Filter by Best Seller"),
    page: int = Query(default=1, ge=1, description="Page Number"),
    limit: int = Query(default=50, ge=1, le=100, description="Results Per Page"),
):
    query, params = filter_helper(
        categories=[category],
        country=country,
        min_price=min_price,
        max_price=max_price,
        best_seller=best_seller,
        page=page,
        limit=limit
    )

    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    return {
        "total": len(rows),
        "products": [format_product(row, settings) for row in rows]
    }

@app.get("/similar_txt")
async def similar_txt(
    q: str = Query(..., description="Search query"),
    limit: int = Query(default=10, ge=1, le=100),
    page: int = Query(default=1, ge=1),
    category: int | None = Query(default=None),
    country: str | None = Query(default=None),
    min_price: float | None = Query(default=None),
    max_price: float | None = Query(default=None),
    best_seller: bool | None = Query(default=None),
):
    if not q.strip():
        return {"total": 0, "products": []}
        
    # 1. Generate embedding for the text query
    embedding = embed_text(app.state.llm, q)
    
    # 2. Get similar categories using vector search
    # We use the category from query if provided, else search for similar categories
    if category:
        category_ids = [category]
    else:
        category_ids = await get_similar_categories(app.state.pool, embedding, top_k=5)
    
    # 3. Get similar products within those categories
    offset = (page - 1) * limit
    product_ids = await get_similar_products(
        app.state.pool, 
        embedding, 
        categories=category_ids, 
        top_k=limit, 
        offset=offset
    )
    
    if not product_ids:
        return {"total": 0, "products": []}

    # 4. Fetch full product details maintaining similarity order
    # Here we also apply other filters (country, price, etc.)
    where_clauses = ["id = ANY($1)"]
    params = [product_ids]
    
    if country:
        params.append(country.lower())
        where_clauses.append(f"country = ${len(params)}")
    if min_price is not None:
        params.append(min_price)
        where_clauses.append(f"price >= ${len(params)}")
    if max_price is not None:
        params.append(max_price)
        where_clauses.append(f"price <= ${len(params)}")
    if best_seller is not None:
        params.append(best_seller)
        where_clauses.append(f"is_best_seller = ${len(params)}")
        
    query = f"""
        SELECT id, asin, country, title, image_path, product_url,
               stars, reviews, price, is_best_seller, bought_in_last_month, labels
        FROM products
        WHERE {" AND ".join(where_clauses)}
        ORDER BY array_position($1, id)
    """
    
    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    
    return {
        "total": len(rows),
        "products": [format_product(row, settings) for row in rows]
    }


if __name__ == "__main__":
    import uvicorn
    settings = Settings()
    uvicorn.run(
        app, 
        host=settings.HOST, 
        port=settings.PORT, 
        ssl_certfile=settings.SSL_CERT_FILE, 
        ssl_keyfile=settings.SSL_KEY_FILE
    )