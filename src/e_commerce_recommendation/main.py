import asyncpg
from fastapi import FastAPI, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.exceptions import HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm import LLM
import torch
import gc
import contextlib
from PIL import Image
import io

from e_commerce_recommendation.configs.settings import Settings
from e_commerce_recommendation.utils.sql_query_helper import filter_helper
from e_commerce_recommendation.utils.utils import format_product
from e_commerce_recommendation.utils.embedding_utils import (
    load_model, embed_text, embed_image, embed_multimodal, get_similar_categories
)

settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await asyncpg.create_pool(
        settings.DATABASE_URL,
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

class ImageSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > settings.MAX_IMAGE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image size exceeds limit of {settings.MAX_IMAGE_SIZE / 1024 / 1024}MB"
                )
        response = await call_next(request)
        return response

app.add_middleware(ImageSizeMiddleware)
    
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
        
    embedding = embed_text(app.state.llm, q)
    category_ids = [category] if category else await get_similar_categories(app.state.pool, embedding, top_k=10)
    
    query, params = filter_helper(
        categories=category_ids,
        country=country,
        min_price=min_price,
        max_price=max_price,
        best_seller=best_seller,
        page=page,
        limit=limit,
        embedding=embedding
    )
    
    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    
    return {
        "total": len(rows),
        "products": [format_product(row, settings) for row in rows]
    }

async def get_image_from_file(file: UploadFile) -> Image.Image:
    import io
    content = await file.read()
    if len(content) > settings.MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Image size exceeds limit of {settings.MAX_IMAGE_SIZE / 1024 / 1024}MB"
        )
    
    # Validate format
    image_format = Image.open(io.BytesIO(content)).format
    if image_format not in settings.ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format. Allowed: {', '.join(settings.ALLOWED_FORMATS)}"
        )
    
    return Image.open(io.BytesIO(content))

@app.post("/similar_img")
async def similar_img(
    image: UploadFile = File(...),
    limit: int = Form(default=10),
    page: int = Form(default=1),
    category: int | None = Form(default=None),
    country: str | None = Form(default=None),
    min_price: float | None = Form(default=None),
    max_price: float | None = Form(default=None),
    best_seller: bool | None = Form(default=None),
):
    # 1. Generate embedding for the image
    img = await get_image_from_file(image)
    embedding = embed_image(app.state.llm, img)
    
    # 2. Get similar categories
    category_ids = [category] if category else await get_similar_categories(app.state.pool, embedding, top_k=10)
    
    # 3. filtering
    query, params = filter_helper(
        categories=category_ids,
        country=country,
        min_price=min_price,
        max_price=max_price,
        best_seller=best_seller,
        page=page,
        limit=limit,
        embedding=embedding
    )
    
    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    
    return {
        "total": len(rows),
        "products": [format_product(row, settings) for row in rows]
    }

@app.post("/similar_multimodal")
async def similar_multimodal(
    q: str = Form(...),
    image: UploadFile = File(...),
    limit: int = Form(default=10),
    page: int = Form(default=1),
    category: int | None = Form(default=None),
    country: str | None = Form(default=None),
    min_price: float | None = Form(default=None),
    max_price: float | None = Form(default=None),
    best_seller: bool | None = Form(default=None),
):
    # 1. Generate multimodal embedding (Order: llm, image, text)
    img = await get_image_from_file(image)
    embedding = embed_multimodal(app.state.llm, img, q)
    
    # 2. Get similar categories
    category_ids = [category] if category else await get_similar_categories(app.state.pool, embedding, top_k=10)
    
    # 3. filtering
    query, params = filter_helper(
        categories=category_ids,
        country=country,
        min_price=min_price,
        max_price=max_price,
        best_seller=best_seller,
        page=page,
        limit=limit,
        embedding=embedding
    )
    
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