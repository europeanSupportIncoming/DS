/*
Identify ”day-one” converters by retrieving full profiles for customers
whose purchases occurred on their corresponding sign-up dates. Return all available cus-
tomer attributes. Sort the results by customer ID in descending order.
*/
select C.*
	from customers as C, sales as S
	where C.customer_id=S.customer_id and C.signup_date=S.sale_date
	order by C.customer_id desc;