--when creating function the first we use create alone without any select to declare the function
-- and then we delete the declartion in order to use the function
--CTRL + K CTRL+C
--CTRL +K CTRL+U -- UNDO
--create function Tam_bolme (@x int ,@y int)
--returns varchar(25)
--as
--begin
--Declare @message varchar(25)
--if(@x%@y=0)
--set @message='Bolunebilir'
--else
--set @message='Bolunilmez'
--return @message
--End
--select dbo.Tam_bolme(3,2)
--CREATE FUNCTION ort_maas()
--RETURNS DECIMAL(10, 2)
--AS
--BEGIN
--    DECLARE @maas DECIMAL(10, 2)
--    SELECT @maas = AVG(maas) FROM personel
--    IF @maas IS NULL
--    BEGIN
--        SET @maas = 0
--    END
--    RETURN @maas
--END


--create function toplam_prim(@cinsyet varchar(1))
--returns int
--as
--begin
--declare @toplam int
--select @toplam =sum(p.prim) from personel p where p.cinsiyet= @cinsyet
--return @toplam
--end
--select dbo.toplam_prim('K') as 'toplam pirm'

--create function unvan_biligi(@id_1 int)
--returns varchar(20)
--as
--begin
--declare @unvan_ad varchar(20)
--select @unvan_ad=u.unvan_ad  from personel p, unvan u where p.personel_no =@id_1 and p.unvan_no=u.unvan_no
--return @unvan_ad
--end

--select dbo.unvan_biligi(3) as 'Unvan Ad'

--create function p_name(@name1 varchar(20))
--returns table
--as 
--return select * from personel where ad like '%' +@name1+ '%'

--select * from p_name('Ah') 
--select * from personel

--create function maas_araligi(@maas1 int, @maas2 int)
--returns table
--as 
--return select ad,soyad,birim_no from personel where maas between @maas1 and @maas2

select * from maas_araligi(3000,3500)

--coklu ifade ile tablo 

--create function s_maas(@ilk int,@son int)
--returns @values table (
--ad nvarchar(20),
--soyad nvarchar(20),
--birim_ad nvarchar(20)
--)
--as begin 
--insert @values
--select p.ad,p.soyad,b.birim_ad from personel p, birim b
--where b.birim_no=p.birim_no and p.maas>@ilk and p.maas<@son

--CREATE FUNCTION s_maas(@ilk INT, @son INT)
--RETURNS @values TABLE
--(
--    ad NVARCHAR(50),
--    soyad NVARCHAR(50),
--    birim_ad NVARCHAR(50)
--)
--AS
--BEGIN
--    INSERT @values
--    SELECT personel.ad, personel.soyad, birim.birim_ad
--    FROM personel, birim
--    WHERE birim.birim_no = personel.birim_no
--      AND personel.maas > @ilk 
--      AND personel.maas < @son
--    RETURN
--END
--select * from dbo.s_maas(3000,3500)
--alter  FUNCTION s_maas(@ilk INT, @son INT)
--RETURNS @values TABLE
--(
--    ad NVARCHAR(50),
--    soyad NVARCHAR(50),
--    birim_ad NVARCHAR(50)
--)
--AS
--BEGIN
--    INSERT @values
--    SELECT personel.ad, personel.soyad, birim.birim_ad
--    FROM personel, birim
--    WHERE birim.birim_no = personel.birim_no
--      AND personel.maas >= @ilk 
--      AND personel.maas <= @son
--    RETURN
--END
select * from s_maas(3000,3500)