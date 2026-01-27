from pydantic_settings import BaseSettings

def format_product(row, settings: BaseSettings) -> dict:
    img_rel = row["image_path"].replace(settings.IMAGE_ROOT, "").lstrip("/")
    image_url = f"/images/{img_rel}"
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
        "distance": float(row["distance"]) if "distance" in row and row["distance"] is not None else None
    }