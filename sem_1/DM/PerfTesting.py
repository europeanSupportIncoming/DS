import psycopg
import time
import json
import numpy as np
import matplotlib.pyplot as plt

HOST = 'localhost'
DB_NAME = 'DM_ex_2_pk'
USER = 'postgres'
PASSWD = 'server'

conn = psycopg.connect(f'dbname={DB_NAME} user={USER} password={PASSWD}')
cur= conn.cursor()


tests = [
(1, [''], ['']),
(2, ['''CREATE INDEX idx_saledate
    ON sales (sale_date);'''], ['''DROP INDEX idx_saledate;
    ''']),
(3, ['''CREATE INDEX idx_saledate
     ON sales (sale_date);
     CREATE INDEX idx_iid
     ON sales_items (sale_id);
     '''], ['''DROP INDEX idx_saledate;
     DROP INDEX idx_iid;
     ''']),
(4, ['''CREATE INDEX idx_saledate
    ON sales (sale_date);
    CREATE INDEX idx_iid
    ON sales_items (sale_id);
    CREATE INDEX idx_catalog_price
    ON products (catalog_price);
    '''], ['''DROP INDEX idx_saledate;
    DROP INDEX idx_iid;
    DROP INDEX idx_catalog_price;
    ''']),
(5, ['''CREATE INDEX idx_saledate
    ON sales (sale_date);
    CREATE INDEX idx_iid
    ON sales_items (sale_id);
    CREATE INDEX idx_catalogue_price
    ON products (catalog_price);
    CREATE INDEX idx_total_amount
    ON sales (total_amount);
    '''], ['''DROP INDEX idx_saledate;
    DROP INDEX idx_iid;
    DROP INDEX idx_catalogue_price;
    DROP INDEX idx_total_amount;
    ''']),
         ]


query = '''
EXPLAIN (FORMAT JSON, ANALYZE)
SELECT
	c.customer_id,
	c.country,
	COUNT(DISTINCT s.sale_id) AS total_sales,
	COUNT(si.item_id) AS total_items,
	SUM(si.quantity) AS total_quantity,
	SUM(si.item_total) AS total_revenue,
	AVG(s.total_amount) AS avg_sale_value,
	MAX(s.sale_date) AS last_sale_date,
	MIN(s.sale_date) AS first_sale_date,
	SUM(s.total_amount) AS total_paid,
	COUNT(DISTINCT p.category) AS categories_bought
FROM customers c
	JOIN sales s ON c.customer_id = s.customer_id
	JOIN sales_items si ON s.sale_id = si.sale_id
	JOIN products p ON si.product_id = p.product_id
WHERE
	s.sale_date BETWEEN TIMESTAMP '2025-03-01'

	AND TIMESTAMP '2025-05-01 23:59:59'
	AND c.country IN ('Germany', 'France', 'Italy', 'Spain', 'Austria')
	AND p.catalog_price > 50
	AND s.discounted = TRUE
	AND si.discounted = TRUE
GROUP BY c.customer_id, c.country
HAVING SUM(si.item_total) > 4
ORDER BY total_revenue DESC, total_sales DESC;
'''

def run_test(t):
    tid, create, drop = t

    with psycopg.connect('dbname=DM_ex_2_pk user=postgres password=server') as conn:
        conn.autocommit = True
        cur = conn.cursor()

        cur.execute("""
        SELECT indexname FROM pg_indexes
        WHERE tablename IN ('sales', 'sales_items', 'products', 'customers')
        AND indexname LIKE 'idx_%'
        AND indexname NOT LIKE 'idx_%_key'
        """)

        existing_indices = [row[0] for row in cur.fetchall()]

        for idx in existing_indices:
            cur.execute(f'DROP INDEX IF EXISTS {idx}')

        cur.execute('DISCARD ALL')

        for q in create:
            cur.execute(q)

        cur.execute(query)
        explain_json = cur.fetchall()[0][0][0]

        exec_time_ms = float(explain_json['Plan']['Actual Total Time'])
        exec_time_sec = exec_time_ms / 1000.0


        for q in drop:
            cur.execute(q)

        cur.execute('DISCARD ALL')

        conn.rollback()

        return exec_time_sec, explain_json

data = []

exec_times = []
jsons = []

for test in tests:
    #data.append(run_test(test)[0][0][0])
    test_id = test[0]
    time_s, jsondic = run_test(test)
    exec_times.append(time_s)
    jsons.append(jsondic)


xax = [test[0] for test in tests]
yax = exec_times

plt.plot(xax, exec_times)
plt.ylabel('Execution Time (s)')
plt.grid(True, alpha=0.5)
plt.savefig('PerfPlot.png')
plt.show()

best_idx = np.argmin(exec_times)
best_test_id = tests[best_idx][0]
best_explain = jsons[best_idx]

with psycopg.connect('dbname=DM_ex_2_pk user=postgres password=server') as conn:
    conn.autocommit = True
    cur = conn.cursor()

    # Clean slate
    cur.execute('DISCARD ALL')

    # Create indices from best test
    for cmd in tests[best_idx][1]:
        if cmd.strip():
            cur.execute(cmd)

    cur.execute(query)
    raw_result = cur.fetchall()[0][0][0]

    # Drop indices
    for cmd in tests[best_idx][2]:
        cur.execute(cmd)
    cur.execute('DISCARD ALL')

with open('Plan.json', 'w') as f:
    json.dump(raw_result, f)