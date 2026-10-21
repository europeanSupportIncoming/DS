DROP TABLE IF EXISTS sales_items, stock, sales, campaigns, products, customers CASCADE;

CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    country VARCHAR(128),
    age_range VARCHAR(8),
    signup_date DATE
);

CREATE TABLE products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(256),
    category VARCHAR(128),
    brand VARCHAR(4),
    color VARCHAR(16),
    size VARCHAR(4),
    catalog_price NUMERIC(16, 2),
    cost_price NUMERIC(16, 2),
    gender VARCHAR(8)
);

CREATE TABLE stock (
    country VARCHAR(128),
    product_id INT,
    stock_quantity INT,
    PRIMARY KEY (product_id, country),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE campaigns (
    campaign_id INT PRIMARY KEY,
    campaign_name VARCHAR(64),
    start_date DATE,
    end_date DATE,
    channel VARCHAR(64),
    discount_type VARCHAR(64),
    discount_value VARCHAR(8)
);

CREATE TABLE sales (
    sale_id INT PRIMARY KEY,
    channel VARCHAR(64),
    discounted BOOLEAN,
    total_amount NUMERIC(16, 2),
    sale_date TIMESTAMP,
    customer_id INT NOT NULL,
    country VARCHAR(128),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE sales_items (
    item_id INT PRIMARY KEY,
    sale_id INT NOT NULL,
    product_id INT NOT NULL,
    quantity INT,
    original_price NUMERIC(8, 2),
    unit_price NUMERIC(8, 2),
    discount_applied NUMERIC(8, 2),
    discount_percent VARCHAR(8),
    discounted BOOLEAN,
    item_total NUMERIC(8, 2),
    sale_date DATE,
	channel VARCHAR(64),
	channel_campaigns VARCHAR(64),
	FOREIGN KEY (sale_id) REFERENCES sales(sale_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);