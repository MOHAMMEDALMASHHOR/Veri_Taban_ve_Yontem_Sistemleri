--create database vokul
--use vokul
--create table ogrenciler(
--ono int,oad varchar(20),onot int)

--use sirket
--sel
--use northwind
--select * from Orders o where o.CustomerID = 'ALFKI'
--use sirket
----select * from personel where prim > 200 order by ad desc
--select * from personel where prim between 20 and 200
--select * from personel where prim > 20 and prim < 200
--select getdate()
--select * from personel where cast(dogum_tarihi as varchar) like '20%'
--select datepart(week,dogum_tarihi) from personel
--select datepart(year,'2021/05/16')

--select datepart(year,dogum_tarihi), count(datepart(year,dogum_tarihi)) from personel group by datepart(year,dogum_tarihi)
--SELECT cinsiyet, AVG(maas) FROM personel GROUP BY cinsiyet HAVING
--AVG(maas)>3500
----SELECT cinsiyet, AVG(maas) FROM personel where avg(maas)>3500 GROUP BY cinsiyet
--SELECT dogum_yeri, AVG(maas) FROM personel GROUP BY dogum_yeri
--HAVING COUNT(*)>2
--SELECT dogum_yeri, AVG(maas) FROM personel GROUP BY dogum_yeri
--HAVING COUNT(*)>1 and (maas/prim<10)-- Column 'personel.prim' is invalid in the HAVING clause because it is not contained in either an aggregate function or the GROUP BY clause

--select sum(maas),ad  from  personel 
--select * from personel where ad like '____'
--select cocuk.ad as 'Çoçuğun Adı',personel.ad as 'Velisinin Adı' from cocuk,personel
--where personel.personel_no=cocuk.personel_no

--select cocuk.ad from cocuk join personel on cocuk.personel_no=personel.personel_no

--use northwind
--SELECT CustomerName
--FROM Customers
--INTERSECT
--SELECT CompanyName
--FROM Suppliers;

--SELECT ProductName
--FROM Products
--WHERE SupplierID = 1
--INTERSECT
--SELECT ProductName
--FROM OrderDetails
--INNER JOIN Products ON OrderDetails.ProductID = Products.ProductID;
