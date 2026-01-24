-- Sample Database Schema for E-Commerce Recommendation
-- This is a subset of 300k products for development/testing

CREATE TABLE IF NOT EXISTS categories (
  id SMALLINT PRIMARY KEY,
  name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
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
  embedding VECTOR(2048),
  
  UNIQUE (asin, country)
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS products_country_idx ON products(country);
CREATE INDEX IF NOT EXISTS products_price_idx ON products(price);
CREATE INDEX IF NOT EXISTS products_bestseller_idx ON products(is_best_seller);
CREATE INDEX IF NOT EXISTS products_stars_idx ON products(stars);
CREATE INDEX IF NOT EXISTS products_reviews_idx ON products(reviews);
CREATE INDEX IF NOT EXISTS products_bought_last_month_idx ON products(bought_in_last_month);
CREATE INDEX IF NOT EXISTS products_labels_idx ON products USING GIN(labels);

-- Vector index for similarity search (create after data is loaded)
-- CREATE INDEX products_vec_idx ON products USING diskann (embedding vector_cosine_ops);
