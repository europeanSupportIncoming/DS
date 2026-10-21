DROP TABLE IF EXISTS SALE_AND_CUSTOMER CASCADE;
DROP TABLE IF EXISTS PRODUCT_WITH_CAMPAIGN CASCADE;
DROP TABLE IF EXISTS SALE_ITEM CASCADE;
DROP TABLE IF EXISTS STOCK CASCADE;
DROP TABLE IF EXISTS SALE CASCADE;
DROP TABLE IF EXISTS CAMPAIGN CASCADE;
DROP TABLE IF EXISTS CUSTOMER CASCADE;
DROP TABLE IF EXISTS PRODUCT CASCADE;

CREATE TABLE PRODUCT (
PID VARCHAR(128) PRIMARY KEY,
Name VARCHAR(128),
Brand VARCHAR(128),
Category VARCHAR(128),
Colour VARCHAR(128),
Size VARCHAR(128),
Gender VARCHAR(8),
True_Price FLOAT,
Catalog_Price FLOAT
);

CREATE TABLE STOCK(
STID VARCHAR(128) PRIMARY KEY,
PID VARCHAR(128)
	REFERENCES PRODUCT(PID)
	ON DELETE CASCADE,
Country VARCHAR(128),
Quantity INTEGER
);

CREATE TABLE CAMPAIGN(
CID VARCHAR(128) PRIMARY KEY,
Name VARCHAR(128),
Channel VARCHAR(128),
Start_Date DATE,
End_Date DATE,
Discount_Type VARCHAR(128),
Discount_Value FLOAT
);

CREATE TABLE SALE(
SID VARCHAR(128) PRIMARY KEY,
CID VARCHAR(128)
	REFERENCES CAMPAIGN(CID)
	ON DELETE SET NULL,
Sale_Date DATE,
Campaign_Applies BOOL
	DEFAULT FALSE,
Country VARCHAR(128),
Discount_Indicator BOOL
);

CREATE TABLE CUSTOMER(
CUID VARCHAR(128) PRIMARY KEY,
Country VARCHAR(128),
Age_Range VARCHAR(128),
Signup_Date DATE
);

CREATE TABLE SALE_ITEM(
IID VARCHAR(128) PRIMARY KEY,
PID VARCHAR(128)
	NOT NULL
	REFERENCES PRODUCT(PID)
	ON DELETE CASCADE,
SID VARCHAR(128)
	NOT NULL
	REFERENCES SALE(SID)
	ON DELETE CASCADE,
Quantity INTEGER
	NOT NULL
	DEFAULT 1
	CHECK (Quantity > 0),
Original_Price FLOAT,
Unit_Price FLOAT,
Discount_Value FLOAT,
Discount_Indicator BOOL,
Final_Item_Total FLOAT
);


CREATE TABLE PRODUCT_WITH_CAMPAIGN(
PID VARCHAR(128)
	NOT NULL
	REFERENCES PRODUCT(PID),
CID VARCHAR(128)
	NOT NULL
	REFERENCES CAMPAIGN(CID),
PCID VARCHAR(256) GENERATED ALWAYS AS (PID ||'-'||CID) STORED PRIMARY KEY
);


CREATE TABLE SALE_AND_CUSTOMER(
SID VARCHAR(128)
	NOT NULL
	REFERENCES SALE(SID),
CUID VARCHAR(128)
	NOT NULL
	REFERENCES CUSTOMER(CUID),
SCID VARCHAR(256) GENERATED ALWAYS AS (SID||'-'||CUID) STORED PRIMARY KEY
);

-- First: Parent tables with no dependencies
INSERT INTO PRODUCT (PID, Name, Brand, Category, Colour, Size, Gender, True_Price, Catalog_Price) VALUES
('P001', 'Running Shoes', 'Nike', 'Footwear', 'Black', '42', 'M', 89.99, 119.99),
('P002', 'Cotton T-Shirt', 'Adidas', 'Apparel', 'White', 'L', 'U', 19.99, 29.99),
('P003', 'Jeans', 'Levis', 'Apparel', 'Blue', '32x32', 'M', 49.99, 69.99),
('P004', 'Sneakers', 'Puma', 'Footwear', 'Red', '38', 'F', 59.99, 79.99),
('P005', 'Hoodie', 'Nike', 'Apparel', 'Gray', 'XL', 'U', 39.99, 59.99);

INSERT INTO CAMPAIGN (CID, Name, Channel, Start_Date, End_Date, Discount_Type, Discount_Value) VALUES
('C001', 'Summer Sale', 'Email', '2024-06-01', '2024-08-31', 'Percentage', 20.0),
('C002', 'Black Friday', 'Social Media', '2024-11-25', '2024-11-30', 'Percentage', 30.0),
('C003', 'New Year Promo', 'App', '2024-12-20', '2025-01-10', 'Fixed', 15.0),
('C004', 'Spring Collection', 'Website', '2024-03-01', '2024-05-31', 'Percentage', 15.0),
('C005', 'Winter Clearance', 'Email', '2024-01-01', '2024-02-28', 'Percentage', 40.0);

INSERT INTO CUSTOMER (CUID, Country, Age_Range, Signup_Date) VALUES
('CUST001', 'USA', '25-34', '2024-01-15'),
('CUST002', 'Canada', '35-44', '2024-02-20'),
('CUST003', 'UK', '18-24', '2024-03-10'),
('CUST004', 'Germany', '45-54', '2024-04-05'),
('CUST005', 'France', '25-34', '2024-05-12');

-- Second: Tables that reference parent tables
INSERT INTO SALE (SID, CID, Sale_Date, Campaign_Applies, Country, Discount_Indicator) VALUES
('S001', 'C001', '2024-06-15', TRUE, 'USA', TRUE),
('S002', NULL, '2024-07-20', FALSE, 'Canada', FALSE),
('S003', 'C002', '2024-11-26', TRUE, 'UK', TRUE),
('S004', 'C003', '2024-12-25', TRUE, 'Germany', TRUE),
('S005', NULL, '2024-10-10', FALSE, 'France', FALSE);

INSERT INTO STOCK (STID, PID, Country, Quantity) VALUES
('STK001', 'P001', 'USA', 100),
('STK002', 'P002', 'Canada', 200),
('STK003', 'P003', 'UK', 150),
('STK004', 'P004', 'Germany', 75),
('STK005', 'P005', 'France', 120);

-- Third: Junction tables and tables with multiple dependencies
INSERT INTO SALE_ITEM (IID, PID, SID, Quantity, Original_Price, Unit_Price, Discount_Value, Discount_Indicator, Final_Item_Total) VALUES
('I001', 'P001', 'S001', 2, 119.99, 95.99, 24.00, TRUE, 191.98),
('I002', 'P002', 'S001', 1, 29.99, 23.99, 6.00, TRUE, 23.99),
('I003', 'P003', 'S002', 3, 69.99, 69.99, 0, FALSE, 209.97),
('I004', 'P004', 'S003', 1, 79.99, 55.99, 24.00, TRUE, 55.99),
('I005', 'P005', 'S004', 2, 59.99, 50.99, 9.00, TRUE, 101.98),
('I006', 'P001', 'S005', 1, 119.99, 119.99, 0, FALSE, 119.99);

-- Auto-generates PCID
INSERT INTO PRODUCT_WITH_CAMPAIGN (PID, CID) VALUES
('P001', 'C001'),
('P002', 'C001'),
('P003', 'C002'),
('P004', 'C003'),
('P005', 'C004');

-- Auto-generates SCID
INSERT INTO SALE_AND_CUSTOMER (SID, CUID) VALUES
('S001', 'CUST001'),
('S002', 'CUST002'),
('S003', 'CUST003'),
('S004', 'CUST004'),
('S005', 'CUST005');

SELECT * FROM PRODUCT;
SELECT * FROM CAMPAIGN;
SELECT * FROM CUSTOMER;
SELECT * FROM SALE;
SELECT * FROM STOCK;
SELECT * FROM SALE_ITEM;
SELECT * FROM PRODUCT_WITH_CAMPAIGN;