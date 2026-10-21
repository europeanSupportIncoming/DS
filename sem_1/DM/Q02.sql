/*
Identify countries with a higher-than-average volume of discounted sales.
Return the country names along with their respective total discounted sales figures. Sort
the results by total discounted sales in descending order.
*/
with country_counts as (
	select country, count(*) as discounted_count
	from sales
	where discounted = True
	group by country
)

select country, discounted_count
	from country_counts
	where discounted_count > (select avg(discounted_count) from country_counts)
	order by discounted_count desc;