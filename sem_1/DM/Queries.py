import numpy as np
import psycopg
import csv
import os
import datetime

HOST = 'localhost'
DB_NAME = 'DM_ex_2_pk'
USER = 'postgres'
PASSWD = 'server'

conn = psycopg.connect(f'dbname={DB_NAME} user={USER} password={PASSWD}')
cur= conn.cursor()

queries = [
    '''select C.*
	from customers as C, sales as S
	where C.customer_id=S.customer_id and C.signup_date=S.sale_date
	order by C.customer_id desc;''',
    '''with country_counts as (
	select country, count(*) as discounted_count
	from sales
	where discounted = True
	group by country
)

select country, discounted_count
	from country_counts
	where discounted_count > (select avg(discounted_count) from country_counts)
	order by discounted_count desc;''',
    '''with relevant_sales_data as(
	select SI.product_id as product_id, S.country as country
	from sales as S, sales_items as SI
	where S.sale_id=SI.sale_id
	),

unsold_stock as (
	select ST.product_id as product_id, ST.country as country
	from stock as ST
	except
	select * from relevant_sales_data
	)

select US.product_id, P.product_name, US.country
	from unsold_stock as US, products as P
	where US.product_id=P.product_id
	order by US.country asc, US.product_id asc;
''',
    '''with category_ordered_products as (
	select 	P.category as category, P.product_id as product_id,
	P.product_name as product_name,	sum(S.quantity) as number_sold
	
	from products as P, sales_items as S
	where P.product_id=S.product_id
	
	group by P.category, P.product_id, P.product_name
),

max_sold_number as (
	select category as category, max(number_sold) as maxnumber
	from category_ordered_products
	group by category
)

select C.category, C.product_id, C.product_name, C.number_sold as quantity_sold
from category_ordered_products as C, max_sold_number as M
where C.category = M.category
and C.number_sold > M.maxnumber*0.75
order by C.category asc, C.product_id asc;
''',
    '''with months_dates as (	
	select extract(month from sale_date) as month, count(*) as sales_nr
	from sales
	group by extract(month from sale_date)
),

max_month as(
	select month as month
	from months_dates
	where sales_nr=(select max(sales_nr) from months_dates)
	group by month
),

freq as (
	select S.customer_id, count(S.customer_id) as frequency
	from sales as S
	where extract(month from S.sale_date) in (select month from max_month)
	group by S.customer_id
),

quant as (
	select S.customer_id, S.country, sum(SI.quantity) as quantity
	from sales as S
	join sales_items as SI on S.sale_id = SI.sale_id
	where extract(month from S.sale_date) in (select month from max_month)
	group by S.customer_id, S.country
)

select f.customer_id, q.country, f.frequency as purchases_in_peak_month,
q.quantity as total_items
from freq f
join quant q on f.customer_id = q.customer_id
order by q.quantity desc, f.frequency desc, f.customer_id asc
limit 15
'''
]

csv_dir = r"C:\Users\paulk\Desktop\DM\ex2\Ex2Results"
def get_rows_csv(csvpath):
    rows = []
    with open(csvpath) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            #convert to same datetime format
            try:
                y, m, d = row[-1].split('-')
                row[-1] = str(datetime.date(int(y), int(m), int(d)))
            except ValueError:
                pass
            rows.append(row)
    return rows

results = [get_rows_csv(os.path.join(csv_dir, csv_file)) for csv_file in os.listdir(csv_dir) if csv_file != 'output.csv']
def get_headers_csv(csvpath):
    with open(csvpath) as csvfile:
        reader = csv.reader(csvfile)
        return next(reader)  # First row = headers

results_headers = [get_headers_csv(os.path.join(csv_dir, csv_file)) for csv_file in os.listdir(csv_dir) if csv_file != 'output.csv']


query_names = ['Q01', 'Q02', 'Q03', 'Q04', 'Q05']
diffs = [0] * len(query_names)



# get query name, row difference count
with psycopg.connect('dbname=DM_ex_2_pk user=postgres password=server') as conn:
    cur = conn.cursor()
    for nr, query in enumerate(queries):
        cur.execute(query)
        data = cur.fetchall()
        query_headers = [desc[0] for desc in cur.description]
        for i in range(min(len(query_headers), len(results_headers[nr]))):
            if query_headers[i] != results_headers[nr][i]:
                diffs[nr] += 1
        diffs[nr] += abs(len(query_headers) - len(results_headers[nr]))
        #get same format as vsc reader
        string_data = [[str(entry) for entry in row] for row in data]
        for row in string_data:
            if row not in results[nr]:
                diffs[nr] += 1

        for row in results[nr]:
            if row not in string_data:
                diffs[nr] += 1


with open(os.path.join(csv_dir, 'QueryDiffs.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    # Write header as required
    writer.writerow(['Query Name', 'Row Difference Count'])
    for query_name, diff in zip(query_names, diffs):
        writer.writerow([query_name, diff])

