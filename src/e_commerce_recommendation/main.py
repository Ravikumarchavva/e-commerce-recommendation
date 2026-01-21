import os
import asyncpg
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

IMAGE_ROOT = "/home/administrator/Desktop/datascience/github/e-commerce-recommendation/data/raw_data"


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve images
app.mount("/images", StaticFiles(directory=IMAGE_ROOT), name="images")

# Create DB pool
@app.on_event("startup")
async def startup():
    app.state.pool = await asyncpg.create_pool(DATABASE_URL, min_size=5, max_size=20)

@app.on_event("shutdown")
async def shutdown():
    await app.state.pool.close()

# ------------------------------------------------
# PRODUCTS API
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
        where.append(f"category_id = ${len(args)}")

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
        category_id
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
            "category_id": r["category_id"]
        })

    return {
        "page": page,
        "limit": limit,
        "products": products
    }

# ------------------------------------------------
# CATEGORY LIST FOR UI
# ------------------------------------------------
@app.get("/categories")
async def get_categories():
    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT DISTINCT category_id
            FROM products
            ORDER BY category_id
        """)
    return [r["category_id"] for r in rows]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)