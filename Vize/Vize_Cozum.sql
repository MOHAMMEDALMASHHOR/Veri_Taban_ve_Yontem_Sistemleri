select * from Customers where CustomerID not in (select distinct CustomerID from Orders)
SELECT c.*
FROM Customers c
LEFT JOIN Orders o ON c.CustomerID = o.CustomerID
WHERE o.CustomerID IS NULL;

--select Customers.CompanyName, Customers.ContactName from Customers where Fax is NULL and CompanyName like '%A' order by CompanyName desc
--3. Soru
--declare @total int
--Select @total = count(OrderID) from Orders where CustomerID ='ALKAFI'

--IF @total < 5 
--print 'Yeni Musteri'
--else if @total >=5 and @total<=10
--print 'Sicka Alisveris Yapan'
--else
--print 'Sadik Musteri'

--4. Soru

--select C.CustomerID, C.CompanyName, Count(Distinct MONTH(OrderDate)) from Customers C join
--Orders O on C.CustomerID = O.CustomerID group by C.CustomerId,C.CompanyName 
--Having Count(Distinct Month(OrderDate)) >3

--5.

--declare cur scroll cursor for select * from Products open cur
--fetch last from cur 
--fetch prior from cur
--close cur
--deallocate cur

--6.
--delete from Products where ProductID not in (Select Distinct ProductID from [Order Details]) and CategoryID = (Select CategoryID from Categories where CategoryName ='Beverages') 

--7.
--select YEAR(OrderDate), sum([Order Details].Quantity*[Order Details].UnitPrice)  from Orders join [Order Details] on Orders.OrderID= [Order Details].OrderID 
--group by Year(OrderDate) order by Year(orderDate)

--8.
--Create VIEW kargo_odeme
--as 
--select Shippers.CompanyName, sum(Orders.Freight) as Odeme from Shippers, Orders where Shippers.ShipperID = Orders.ShipVia
--group by Shippers.CompanyName

--9.
--Create Function enpahal()
--returns table
--as 
--return select top 10 ProductName, UnitPrice from Products Order By UnitPrice desc

--10.
create function average(@categoryID int)
returns decimal(10,2)
as 
begin
declare @averagePrice Decimal(10,2)
select @averagePrice= Avg(UnitPrice) from Products where CategoryID = @categoryID
return ifnull (@averagePrice,0)
end


