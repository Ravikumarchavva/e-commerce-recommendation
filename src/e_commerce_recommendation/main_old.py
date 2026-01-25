import os
import asyncpg
from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from PIL import Image
import io
from embeddings import load_model, embed_text, embed_image, embed_multimodal

load_dotenv()

DATABASE_URL = os.getenv("SAMPLE_DATABASE_URL")
IMAGE_ROOT = "/home/administrator/Desktop/datascience/github/e-commerce-recommendation/data/raw_data"

# PostgreSQL vector converter
def to_pgvector(vec):
    """Convert list to PostgreSQL vector format."""
    return str(vec)

app = FastAPI(title="E-Commerce Vector Search API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve images as CDN
app.mount("/images", StaticFiles(directory=IMAGE_ROOT), name="images")

# Create DB pool with optimized settings
@app.on_event("startup")
async def startup():
    app.state.pool = await asyncpg.create_pool(
        DATABASE_URL, 
        min_size=10, 
        max_size=50,
        command_timeout=60
    )
    # Load embedding model at startup
    load_model()
    print("✅ Server ready with vector search enabled")

@app.on_event("shutdown")
async def shutdown():
    await app.state.pool.close()

# ------------------------------------------------
# CATEGORIES API - Metadata for filters
# ------------------------------------------------
@app.get("/categories")
async def get_categories():
    """Get all available categories for filtering."""
    async with app.state.pool.acquire() as conn:
        # Try to get from categories table first
        try:
            rows = await conn.fetch("""
                SELECT id, name
                FROM categories
                ORDER BY name
            """)
            return [{"id": r["id"], "name": r["name"]} for r in rows]
        except:
            # Fallback: get unique category IDs from labels array
            rows = await conn.fetch("""
                SELECT DISTINCT unnest(labels) as label_id
                FROM products
                ORDER BY label_id
            """)
            return [{"id": r["label_id"], "name": f"Category {r['label_id']}"} for r in rows]


# ------------------------------------------------
# BROWSE - Fast SQL path for category/filter browsing
# ------------------------------------------------
@app.get("/browse")
async def browse(
    category: int | None = Query(None, description="Category ID to filter by"),
    country: str | None = Query(None, description="Country filter (usa, india, uk, canada)"),
    min_price: float | None = Query(None, description="Minimum price filter"),
    max_price: float | None = Query(None, description="Maximum price filter"),
    best_seller: bool | None = Query(None, description="Show only best sellers"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(40, le=100, description="Results per page")
):
    """
    Browse products with SQL filters (no vector search).
    Optimized for category browsing with price, country filters.
    """
    offset = (page - 1) * limit
    where = []
    args = []

    if category:
        args.append(category)
        where.append(f"${len(args)} = ANY(labels)")

    if country:
        args.append(country)
        where.append(f"country = ${len(args)}")

    if min_price is not None:
        args.append(min_price)
        where.append(f"price >= ${len(args)}")

    if max_price is not None:
        args.append(max_price)
        where.append(f"price <= ${len(args)}")

    if best_seller is not None:
        args.append(best_seller)
        where.append(f"is_best_seller = ${len(args)}")

    where_sql = "WHERE " + " AND ".join(where) if where else ""

    sql = f"""
    SELECT
        id,
        asin,
        country,
        title,
        image_path,
        product_url,
        stars,
        reviews,
        price,
        is_best_seller,
        bought_in_last_month,
        labels
    FROM products
    {where_sql}
    ORDER BY bought_in_last_month DESC NULLS LAST, stars DESC NULLS LAST
    LIMIT {limit} OFFSET {offset}
    """

    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch(sql, *args)

    products = []
    for r in rows:
        img_rel = r["image_path"].replace(IMAGE_ROOT, "").lstrip("/")
        products.append({
            "id": r["id"],
            "asin": r["asin"],
            "country": r["country"],
            "title": r["title"],
            "imageUrl": f"http://localhost:8000/images/{img_rel}",
            "productURL": r["product_url"],
            "stars": r["stars"],
            "reviews": r["reviews"],
            "price": float(r["price"]) if r["price"] else 0.0,
            "isBestSeller": r["is_best_seller"],
            "bought": r["bought_in_last_month"],
            "categories": r["labels"]
        })

    return {
        "page": page,
        "limit": limit,
        "total": len(products),
        "products": products
    }


# ------------------------------------------------
# TEXT SEARCH - Vector similarity search with DiskANN
# ------------------------------------------------
@app.get("/search/text")
async def search_text(
    q: str = Query(..., description="Search query text"),
    category: int | None = Query(None, description="Filter by category"),
    country: str | None = Query(None, description="Filter by country"),
    min_price: float | None = Query(None, description="Minimum price"),
    max_price: float | None = Query(None, description="Maximum price"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(40, le=200, description="Number of results")
):
    """
    Search products using text query with vector similarity (DiskANN).
    Supports label-filtered search when category is provided.
    """
    try:
        # Generate embedding for search query
        embedding = to_pgvector(embed_text(q))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

    # Build additional filters
    post_filters = []
    filter_args = []

    if country:
        filter_args.append(country)
        post_filters.append(f"country = ${{len(filter_args) + 3}}")

    if min_price is not None:
        filter_args.append(min_price)
        post_filters.append(f"price >= ${{len(filter_args) + 3}}")

    if max_price is not None:
        filter_args.append(max_price)
        post_filters.append(f"price <= ${{len(filter_args) + 3}}")

    filter_sql = ("AND " + " AND ".join(post_filters)) if post_filters else ""
    offset = (page - 1) * limit

    async with app.state.pool.acquire() as conn:
        if category:
            # Label-filtered DiskANN search
            sql = f"""
                SELECT
                    id, asin, country, title, image_path, product_url,
                    stars, reviews, price, is_best_seller,
                    bought_in_last_month, labels,
                    embedding <=> $1 AS distance
                FROM products
                WHERE $2 = ANY(labels)
                {filter_sql}
                ORDER BY embedding <=> $1
                LIMIT $3 OFFSET $4
            """
            rows = await conn.fetch(sql, embedding, category, limit, offset, *filter_args)
        else:
            # Full DiskANN search - rebuild filter numbering for non-category case
            post_filters_no_cat = []
            for i, arg in enumerate(filter_args, start=2):
                if filter_args[i-2] == country:
                    post_filters_no_cat.append(f"country = ${i}")
                elif filter_args[i-2] == min_price:
                    post_filters_no_cat.append(f"price >= ${i}")
                elif filter_args[i-2] == max_price:
                    post_filters_no_cat.append(f"price <= ${i}")
            filter_sql_no_cat = ("AND " + " AND ".join(post_filters_no_cat)) if post_filters_no_cat else ""
            
            sql = f"""
                SELECT
                    id, asin, country, title, image_path, product_url,
                    stars, reviews, price, is_best_seller,
                    bought_in_last_month, labels,
                    embedding <=> $1 AS distance
                FROM products
                WHERE true
                {filter_sql_no_cat}
                ORDER BY embedding <=> $1
                LIMIT ${len(filter_args) + 2} OFFSET ${len(filter_args) + 3}
            """
            rows = await conn.fetch(sql, embedding, *filter_args, limit, offset)

    products = []
    for r in rows:
        img_rel = r["image_path"].replace(IMAGE_ROOT, "").lstrip("/")
        products.append({
            "id": r["id"],
            "asin": r["asin"],
            "country": r["country"],
            "title": r["title"],
            "imageUrl": f"http://localhost:8000/images/{img_rel}",
            "productURL": r["product_url"],
            "stars": r["stars"],
            "reviews": r["reviews"],
            "price": float(r["price"]) if r["price"] else 0.0,
            "isBestSeller": r["is_best_seller"],
            "bought": r["bought_in_last_month"],
            "categories": r["labels"],
            "relevance": float(r["distance"])
        })

    return {
        "query": q,
        "page": page,
        "limit": limit,
        "total": len(products),
        "products": products
    }


# ------------------------------------------------
# IMAGE SEARCH - Upload image and find similar products
# ------------------------------------------------
@app.post("/search/image")
async def search_image(
    file: UploadFile = File(..., description="Product image to search"),
    category: int | None = Query(None, description="Filter by category"),
    country: str | None = Query(None, description="Filter by country"),
    limit: int = Query(40, le=100, description="Number of results")
):
    """
    Search products by uploading an image.
    Uses vector similarity with DiskANN index.
    """
    try:
        # Read and process uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Generate embedding
        embedding = to_pgvector(embed_image(image))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

    # Build filters
    post_filters = []
    filter_args = []

    if country:
        filter_args.append(country)
        post_filters.append(f"country = ${len(filter_args) + 3}")

    filter_sql = ("AND " + " AND ".join(post_filters)) if post_filters else ""

    async with app.state.pool.acquire() as conn:
        if category:
            # Label-filtered search
            sql = f"""
                SELECT
                    id, asin, country, title, image_path, product_url,
                    stars, reviews, price, is_best_seller,
                    bought_in_last_month, labels,
                    embedding <=> $1 AS distance
                FROM products
                WHERE $2 = ANY(labels)
                {filter_sql}
                ORDER BY embedding <=> $1
                LIMIT $3
            """
            rows = await conn.fetch(sql, embedding, category, limit, *filter_args)
        else:
            # Full search - rebuild filter numbering
            post_filters_no_cat = []
            for i in range(len(filter_args)):
                post_filters_no_cat.append(f"country = ${i + 2}")
            filter_sql_no_cat = ("AND " + " AND ".join(post_filters_no_cat)) if post_filters_no_cat else ""
            
            sql = f"""
                SELECT
                    id, asin, country, title, image_path, product_url,
                    stars, reviews, price, is_best_seller,
                    bought_in_last_month, labels,
                    embedding <=> $1 AS distance
                FROM products
                WHERE true
                {filter_sql_no_cat}
                ORDER BY embedding <=> $1
                LIMIT ${len(filter_args) + 2}
            """
            rows = await conn.fetch(sql, embedding, *filter_args, limit)

    products = []
    for r in rows:
        img_rel = r["image_path"].replace(IMAGE_ROOT, "").lstrip("/")
        products.append({
            "id": r["id"],
            "asin": r["asin"],
            "country": r["country"],
            "title": r["title"],
            "imageUrl": f"http://localhost:8000/images/{img_rel}",
            "productURL": r["product_url"],
            "stars": r["stars"],
            "reviews": r["reviews"],
            "price": float(r["price"]) if r["price"] else 0.0,
            "isBestSeller": r["is_best_seller"],
            "bought": r["bought_in_last_month"],
            "categories": r["labels"],
            "similarity": 1 - float(r["distance"])  # Convert distance to similarity
        })

    return {
        "total": len(products),
        "products": products
    }


# ------------------------------------------------
# MULTIMODAL SEARCH - Text + Image combined
# ------------------------------------------------
@app.post("/search/multimodal")
async def search_multimodal(
    file: UploadFile = File(..., description="Product image to search"),
    q: str = Query(..., description="Text query to combine with image"),
    category: int | None = Query(None, description="Filter by category"),
    country: str | None = Query(None, description="Filter by country"),
    limit: int = Query(40, le=100, description="Number of results")
):
    """
    Search products using both text and image (multimodal search).
    Combines visual and textual context for more precise results.
    """
    try:
        # Read and process uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Generate combined embedding
        embedding = to_pgvector(embed_multimodal(image, q))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Multimodal search failed: {str(e)}")

    # Build filters
    post_filters = []
    filter_args = []

    if country:
        filter_args.append(country)
        post_filters.append(f"country = ${len(filter_args) + 3}")

    filter_sql = ("AND " + " AND ".join(post_filters)) if post_filters else ""

    async with app.state.pool.acquire() as conn:
        if category:
            # Label-filtered search
            sql = f"""
                SELECT
                    id, asin, country, title, image_path, product_url,
                    stars, reviews, price, is_best_seller,
                    bought_in_last_month, labels,
                    embedding <=> $1 AS distance
                FROM products
                WHERE $2 = ANY(labels)
                {filter_sql}
                ORDER BY embedding <=> $1
                LIMIT $3
            """
            rows = await conn.fetch(sql, embedding, category, limit, *filter_args)
        else:
            # Full search - rebuild filter numbering
            post_filters_no_cat = []
            for i in range(len(filter_args)):
                post_filters_no_cat.append(f"country = ${i + 2}")
            filter_sql_no_cat = ("AND " + " AND ".join(post_filters_no_cat)) if post_filters_no_cat else ""
            
            sql = f"""
                SELECT
                    id, asin, country, title, image_path, product_url,
                    stars, reviews, price, is_best_seller,
                    bought_in_last_month, labels,
                    embedding <=> $1 AS distance
                FROM products
                WHERE true
                {filter_sql_no_cat}
                ORDER BY embedding <=> $1
                LIMIT ${len(filter_args) + 2}
            """
            rows = await conn.fetch(sql, embedding, *filter_args, limit)

    products = []
    for r in rows:
        img_rel = r["image_path"].replace(IMAGE_ROOT, "").lstrip("/")
        products.append({
            "id": r["id"],
            "asin": r["asin"],
            "country": r["country"],
            "title": r["title"],
            "imageUrl": f"http://localhost:8000/images/{img_rel}",
            "productURL": r["product_url"],
            "stars": r["stars"],
            "reviews": r["reviews"],
            "price": float(r["price"]) if r["price"] else 0.0,
            "isBestSeller": r["is_best_seller"],
            "bought": r["bought_in_last_month"],
            "categories": r["labels"],
            "similarity": 1 - float(r["distance"])
        })

    return {
        "query": q,
        "total": len(products),
        "products": products
    }


# ------------------------------------------------
# LEGACY PRODUCTS API (kept for backward compatibility)
# ------------------------------------------------
@app.get("/products")
async def get_products(
    page: int = Query(1, ge=1),
    limit: int = Query(40, le=100),
    category_id: int | None = None,
    country: str | None = None
):
    offset = (page - 1) * limit
    where = []
    args = []

    if category_id:
        args.append(category_id)
        where.append(f"${len(args)} = ANY(labels)")

    if country:
        args.append(country)
        where.append(f"country = ${len(args)}")

    where_sql = "WHERE " + " AND ".join(where) if where else ""

    sql = f"""
    SELECT
        asin,
        country,
        title,
        image_path,
        product_url,
        stars,
        price,
        bought_in_last_month,
        labels
    FROM products
    {where_sql}
    ORDER BY bought_in_last_month DESC
    LIMIT {limit} OFFSET {offset}
    """

    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch(sql, *args)

    products = []
    for r in rows:
        img_rel = r["image_path"].replace(IMAGE_ROOT, "").lstrip("/")
        products.append({
            "asin": r["asin"],
            "country": r["country"],
            "title": r["title"],
            "imageUrl": f"http://localhost:8000/images/{img_rel}",
            "productURL": r["product_url"],
            "stars": r["stars"],
            "price": float(r["price"]),
            "bought": r["bought_in_last_month"],
            "categories": r.get("labels", [])
        })

    return {
        "page": page,
        "limit": limit,
        "products": products
    }


# ------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------
@app.get("/")
async def health_check():
    """API health check."""
    return {
        "status": "healthy",
        "service": "E-Commerce Vector Search API",
        "version": "1.0.0",
        "endpoints": {
            "browse": "/browse",
            "text_search": "/search/text",
            "image_search": "/search/image",
            "categories": "/categories",
            "products": "/products (legacy)"
        }
    }


# ------------------------------------------------
# CATEGORY LIST FOR UI
# ------------------------------------------------
@app.get("/categories")
async def get_categories_list():
    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT DISTINCT unnest(labels) as label_id
            FROM products
            ORDER BY label_id
        """)
    return [r["label_id"] for r in rows]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)