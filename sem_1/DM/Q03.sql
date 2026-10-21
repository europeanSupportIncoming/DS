/*
Locate the dead stock by retrieving products that are currently in
inventory within a specific country but have recorded zero sales in it.
Return the product ID and name as well as the associated country
from the stock records. Sort the results alphabetically by country,
then by product ID in ascending order
*/
with relevant_sales_data as(
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




