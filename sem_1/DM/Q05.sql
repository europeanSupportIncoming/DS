/*
List the top 15 customers who were most active during the business’s
historical peak sales month by retrieving their purchase frequency
and total item volume. The ”peak sales month” is the calendar month
with the highest count of unique transactions; ”purchase frequency”
is the count of distinct sales, and ”total item volume” is the sum 
of all units purchased. Return the customer ID, country,
purchase frequency, and total item volume for that month.
Sort the results by total item volume (descending), then by
purchase frequency (descending), and finally by customer ID (ascending)
to resolve ties.
*/
with months_dates as (	
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
