import asyncpg
import io
from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image
from contextlib import asynccontextmanager

from e_commerce_recommendation.embeddings import load_model, embed_text, embed_image, embed_multimodal
from e_commerce_recommendation.configs.settings import Settings

# Load settings
settings = Settings()
DATABASE_URL = settings.SAMPLE_DATABASE_URL
IMAGE_ROOT = "/home/administrator/Desktop/datascience/github/e-commerce-recommendation/data/raw_data"


def to_pgvector(embedding: list[float]) -> str:
    """Convert embedding list to PostgreSQL vector format string."""
    return str(embedding)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    # Startup
    app.state.pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=5,
        max_size=20,
        command_timeout=60
    )
    load_model()
    print("✅ Server ready - Model loaded, DB pool created")
    
    yield
    
    # Shutdown
    await app.state.pool.close()
    print("👋 Server shutdown - DB pool closed")


app = FastAPI(
    title="E-Commerce Vector Search API",
    version="2.0.0",
    description="Multi-modal product search with Qwen3-VL embeddings",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve images as static files
app.mount("/images", StaticFiles(directory=IMAGE_ROOT), name="images")


def format_product(row, image_root: str) -> dict:
    """Format a database row into a product dict with full image URL."""
    img_rel = row["image_path"].replace(image_root, "").lstrip("/")
    # You can change 'localhost:8000' to your server's hostname if needed
    image_url = f"http://localhost:8000/images/{img_rel}"
    return {
        "id": row["id"],
        "asin": row["asin"],
        "country": row["country"],
        "title": row["title"],
        "imageUrl": image_url,
        "productURL": row["product_url"],
        "stars": float(row["stars"]) if row["stars"] else None,
        "reviews": row["reviews"],
        "price": float(row["price"]) if row["price"] else 0.0,
        "isBestSeller": row["is_best_seller"],
        "bought": row["bought_in_last_month"],
        "categories": row["labels"],
        "distance": float(row.get("distance", 0)) if "distance" in row else None
    }


# ================================================
# HEALTH CHECK
# ================================================
@app.get("/")
async def health_check():
    """API health check with endpoint documentation."""
    return {
        "status": "healthy",
        "service": "E-Commerce Vector Search API",
        "version": "2.0.0",
        "model": "Qwen3-VL-Embedding-2B",
        "endpoints": {
            "browse": "GET /browse - Browse products with filters",
            "text_search": "GET /search/text - Search by text query",
            "image_search": "POST /search/image - Search by uploaded image",
            "multimodal_search": "POST /search/multimodal - Search by text + image",
            "categories": "GET /categories - List all categories"
        }
    }


# ================================================
# CATEGORIES
# ================================================
@app.get("/categories")
async def get_categories():
    """Get all unique product categories."""
    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT DISTINCT unnest(labels) as category_id
            FROM products
            WHERE labels IS NOT NULL
            ORDER BY category_id
        """)
    
    return {
        "total": len(rows),
        "categories": [{"id": r["category_id"], "name": f"Category {r['category_id']}"} for r in rows]
    }


# ================================================
# BROWSE - SQL filtered browsing (no vector search)
# ================================================
@app.get("/browse")
async def browse(
    category: int | None = Query(None, description="Filter by category ID"),
    country: str | None = Query(None, description="Filter by country (usa, india, uk, canada)"),
    min_price: float | None = Query(None, description="Minimum price"),
    max_price: float | None = Query(None, description="Maximum price"),
    best_seller: bool | None = Query(None, description="Show only best sellers"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(40, ge=1, le=100, description="Results per page")
):
    """
    Browse products with traditional SQL filters (no AI search).
    Fast browsing by category, price, country, and best-seller status.
    """
    offset = (page - 1) * limit
    
    # Build WHERE clause dynamically
    conditions = []
    params = []
    
    if category is not None:
        params.append(category)
        conditions.append(f"${len(params)} = ANY(labels)")
    
    if country:
        params.append(country.lower())
        conditions.append(f"LOWER(country) = ${len(params)}")
    
    if min_price is not None:
        params.append(min_price)
        conditions.append(f"price >= ${len(params)}")
    
    if max_price is not None:
        params.append(max_price)
        conditions.append(f"price <= ${len(params)}")
    
    if best_seller is not None:
        params.append(best_seller)
        conditions.append(f"is_best_seller = ${len(params)}")
    
    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    
    # Add limit and offset parameters
    params.extend([limit, offset])
    limit_param = f"${len(params) - 1}"
    offset_param = f"${len(params)}"
    
    query = f"""
        SELECT 
            id, asin, country, title, image_path, product_url,
            stars, reviews, price, is_best_seller, bought_in_last_month, labels
        FROM products
        {where_clause}
        ORDER BY bought_in_last_month DESC NULLS LAST, stars DESC NULLS LAST
        LIMIT {limit_param} OFFSET {offset_param}
    """
    
    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    
    products = [format_product(r, IMAGE_ROOT) for r in rows]
    
    return {
        "page": page,
        "limit": limit,
        "total": len(products),
        "filters": {
            "category": category,
            "country": country,
            "min_price": min_price,
            "max_price": max_price,
            "best_seller": best_seller
        },
        "products": products
    }


# ================================================
# TEXT SEARCH - Vector similarity search
# ================================================
@app.get("/search/text")
async def search_text(
    q: str = Query(..., min_length=1, description="Search query text"),
    category: int | None = Query(None, description="Filter by category"),
    country: str | None = Query(None, description="Filter by country"),
    min_price: float | None = Query(None, ge=0, description="Minimum price"),
    max_price: float | None = Query(None, ge=0, description="Maximum price"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(40, ge=1, le=200, description="Results per page")
):
    """
    Search products using text query with vector similarity.
    Uses cosine distance with pgvector for semantic search.
    """
    try:
        # Generate query embedding
        query_embedding = embed_text(q)
        embedding_str = to_pgvector(query_embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")
    
    offset = (page - 1) * limit
    
    # Build WHERE clause
    conditions = []
    params = [embedding_str]  # $1 is the embedding
    
    if category is not None:
        params.append(category)
        conditions.append(f"${len(params)} = ANY(labels)")
    
    if country:
        params.append(country.lower())
        conditions.append(f"LOWER(country) = ${len(params)}")
    
    if min_price is not None:
        params.append(min_price)
        conditions.append(f"price >= ${len(params)}")
    
    if max_price is not None:
        params.append(max_price)
        conditions.append(f"price <= ${len(params)}")
    
    where_clause = "AND " + " AND ".join(conditions) if conditions else ""
    
    # Add limit and offset
    params.extend([limit, offset])
    limit_param = f"${len(params) - 1}"
    offset_param = f"${len(params)}"
    
    query = f"""
        SELECT 
            id, asin, country, title, image_path, product_url,
            stars, reviews, price, is_best_seller, bought_in_last_month, labels,
            embedding <=> $1::vector AS distance
        FROM products
        WHERE embedding IS NOT NULL
        {where_clause}
        ORDER BY embedding <=> $1::vector
        LIMIT {limit_param} OFFSET {offset_param}
    """
    
    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    
    products = [format_product(r, IMAGE_ROOT) for r in rows]
    
    return {
        "query": q,
        "page": page,
        "limit": limit,
        "total": len(products),
        "filters": {
            "category": category,
            "country": country,
            "min_price": min_price,
            "max_price": max_price
        },
        "products": products
    }


# ================================================
# IMAGE SEARCH - Upload image and find similar products
# ================================================
@app.post("/search/image")
async def search_image(
    file: UploadFile = File(..., description="Product image to search"),
    category: int | None = Query(None, description="Filter by category"),
    country: str | None = Query(None, description="Filter by country"),
    min_price: float | None = Query(None, ge=0, description="Minimum price"),
    max_price: float | None = Query(None, ge=0, description="Maximum price"),
    limit: int = Query(40, ge=1, le=200, description="Number of results")
):
    """
    Search products by uploading an image.
    Uses visual embeddings for similarity matching.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Resize if too large
        MAX_SIZE = 672
        if max(image.size) > MAX_SIZE:
            image.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)
        
        # Generate embedding
        query_embedding = embed_image(image)
        embedding_str = to_pgvector(query_embedding)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")
    
    # Build WHERE clause
    conditions = []
    params = [embedding_str]  # $1 is the embedding
    
    if category is not None:
        params.append(category)
        conditions.append(f"${len(params)} = ANY(labels)")
    
    if country:
        params.append(country.lower())
        conditions.append(f"LOWER(country) = ${len(params)}")
    
    if min_price is not None:
        params.append(min_price)
        conditions.append(f"price >= ${len(params)}")
    
    if max_price is not None:
        params.append(max_price)
        conditions.append(f"price <= ${len(params)}")
    
    where_clause = "AND " + " AND ".join(conditions) if conditions else ""
    
    # Add limit
    params.append(limit)
    limit_param = f"${len(params)}"
    
    query = f"""
        SELECT 
            id, asin, country, title, image_path, product_url,
            stars, reviews, price, is_best_seller, bought_in_last_month, labels,
            embedding <=> $1::vector AS distance
        FROM products
        WHERE embedding IS NOT NULL
        {where_clause}
        ORDER BY embedding <=> $1::vector
        LIMIT {limit_param}
    """
    
    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    
    products = [format_product(r, IMAGE_ROOT) for r in rows]
    
    return {
        "total": len(products),
        "filters": {
            "category": category,
            "country": country,
            "min_price": min_price,
            "max_price": max_price
        },
        "products": products
    }


# ================================================
# MULTIMODAL SEARCH - Combined text + image
# ================================================
@app.post("/search/multimodal")
async def search_multimodal(
    file: UploadFile = File(..., description="Product image to search"),
    q: str = Query(..., min_length=1, description="Text query to combine with image"),
    category: int | None = Query(None, description="Filter by category"),
    country: str | None = Query(None, description="Filter by country"),
    min_price: float | None = Query(None, ge=0, description="Minimum price"),
    max_price: float | None = Query(None, ge=0, description="Maximum price"),
    limit: int = Query(40, ge=1, le=200, description="Number of results")
):
    """
    Search products using both text and image (multimodal search).
    Combines visual and textual context for precise results.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Resize if too large
        MAX_SIZE = 672
        if max(image.size) > MAX_SIZE:
            image.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)
        
        # Generate combined embedding
        query_embedding = embed_multimodal(image, q)
        embedding_str = to_pgvector(query_embedding)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Multimodal embedding failed: {str(e)}")
    
    # Build WHERE clause
    conditions = []
    params = [embedding_str]  # $1 is the embedding
    
    if category is not None:
        params.append(category)
        conditions.append(f"${len(params)} = ANY(labels)")
    
    if country:
        params.append(country.lower())
        conditions.append(f"LOWER(country) = ${len(params)}")
    
    if min_price is not None:
        params.append(min_price)
        conditions.append(f"price >= ${len(params)}")
    
    if max_price is not None:
        params.append(max_price)
        conditions.append(f"price <= ${len(params)}")
    
    where_clause = "AND " + " AND ".join(conditions) if conditions else ""
    
    # Add limit
    params.append(limit)
    limit_param = f"${len(params)}"
    
    query = f"""
        SELECT 
            id, asin, country, title, image_path, product_url,
            stars, reviews, price, is_best_seller, bought_in_last_month, labels,
            embedding <=> $1::vector AS distance
        FROM products
        WHERE embedding IS NOT NULL
        {where_clause}
        ORDER BY embedding <=> $1::vector
        LIMIT {limit_param}
    """
    
    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    
    products = [format_product(r, IMAGE_ROOT) for r in rows]
    
    return {
        "query": q,
        "total": len(products),
        "filters": {
            "category": category,
            "country": country,
            "min_price": min_price,
            "max_price": max_price
        },
        "products": products
    }


# ================================================
# ERROR HANDLERS
# ================================================
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle unexpected errors gracefully."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "type": type(exc).__name__
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
