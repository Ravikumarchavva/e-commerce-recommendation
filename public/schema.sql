CREATE TABLE categories (
  id SMALLINT PRIMARY KEY,
  name TEXT NOT NULL
);

CREATE TABLE category_embeddings (
  category_id SMALLINT PRIMARY KEY REFERENCES categories(id),
  embedding VECTOR(2048) NOT NULL
);

CREATE INDEX category_ann_idx
ON category_embeddings
USING diskann (embedding vector_cosine_ops);

CREATE TABLE products (
  id BIGSERIAL PRIMARY KEY,
  asin TEXT NOT NULL,
  country TEXT NOT NULL,
  title TEXT,
  image_path TEXT,
  product_url TEXT,
  stars REAL,
  reviews INT,
  price NUMERIC,
  is_best_seller BOOLEAN,
  bought_in_last_month INT,

  labels SMALLINT[] NOT NULL,
  embedding VECTOR(1024) NOT NULL,

  UNIQUE (asin, country)
);

set maintenance_work_mem = '40GB';
CREATE INDEX products_vec_idx
ON products
USING diskann (embedding vector_cosine_ops, labels)
WITH (
    storage_layout = 'memory_optimized',
    num_neighbors = 64,
    search_list_size = 200,
    max_alpha = 1.2
);
show maintenance_work_mem;

CREATE INDEX products_country_idx ON products(country);
CREATE INDEX products_price_idx ON products(price);
CREATE INDEX products_bestseller_idx ON products(is_best_seller);
CREATE INDEX products_stars_idx ON products(stars);
CREATE INDEX products_reviews_idx ON products(reviews);
CREATE INDEX products_bought_last_month_idx ON products(bought_in_last_month);

ALTER TABLE products
ALTER COLUMN embedding DROP NOT NULL;

DROP INDEX IF EXISTS products_vec_idx;
ALTER TABLE products
ALTER COLUMN embedding TYPE VECTOR(2048);



DROP TABLE IF EXISTS category_embeddings;
