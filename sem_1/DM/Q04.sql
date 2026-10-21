
/*
Identify the best performing products within each category by selecting
items in the upper quartile (top 25%) based on the category-specific sales volume. Sales
volume is defined as the total units sold per product. Return the category, product ID
and name as well as the total sold quantity of these top-tier items. Sort the results in the
output alphabetically by category, then by product ID in ascending order.

discord add on: obviously, the 25 percentile mentioned here is wrong.
instead we should take values larger than 0.75 times the max value
*/
with category_ordered_products as (
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


