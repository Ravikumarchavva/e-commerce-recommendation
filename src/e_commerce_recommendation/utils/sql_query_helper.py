from typing import List, Tuple, Any

def filter_helper(
    categories: List[int | None] | None,
    country: str | None,
    min_price: float | None,
    max_price: float | None,
    best_seller: bool | None,
    page: int,
    limit: int,
    embedding: List[float] | None = None
) -> Tuple[str, List[Any]]:
    offset = (page - 1) * limit
    conditions = []
    params = []

    if categories is not None:
        valid_categories = [c for c in categories if c is not None]
        if valid_categories:
            params.append(valid_categories)
            conditions.append(f"labels && ${len(params)}::smallint[]")

    if country is not None:
        params.append(country.lower())
        conditions.append(f"country = ${len(params)}")

    if min_price is not None:
        params.append(min_price)
        conditions.append(f"price >= ${len(params)}")

    if max_price is not None:
        params.append(max_price)
        conditions.append(f"price <= ${len(params)}")

    if best_seller is not None:
        params.append(best_seller)
        conditions.append(f"is_best_seller = ${len(params)}")

    order_by = "price ASC"
    if embedding is not None:
        params.append(str(embedding))
        conditions.append(f"embedding <=> ${len(params)} < 0.4")
        order_by = f"embedding <=> ${len(params)}"

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    query = f"""
        SELECT 
            id, asin, country, title, image_path, product_url,
            stars, reviews, price, is_best_seller, bought_in_last_month, labels
        FROM products
        {where_clause}
        ORDER BY {order_by}
        LIMIT {limit} OFFSET {offset}
    """
    return query, params