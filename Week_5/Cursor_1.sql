declare @ilce_ad nvarchar(20)
declare @il_no int
declare cursor_dyeri cursor for
select ilce_ad,il_no from ilce

open cursor_dyeri

fetch next from cursor_dyeri into @ilce_ad,@il_no

while @@FETCH_STATUS =0 --sifir ise veri cekebiliyoruz
begin
print CAST(@il_no as nvarchar) + '-' +@ilce_ad

fetch next from cursor_dyeri into @ilce_ad,@il_no
end
close cursor_dyeri
deallocate cursor_dyeri

declare @personelad nvarchar(20)
declare @personelsoyad nvarchar(20)
declare cursor_dyer cursor for
select ad,soyad from personel where dogum_yeri=3

open cursor_dyer 
fetch next from cursor_dyer into @personelad,@personelsoyad
while @@FETCH_STATUS=0
begin 
print @personelad+' '+@personelsoyad

fetch next from cursor_dyer into @personelad,@personelsoyad
end
close cursor_dyer
Deallocate cursor_dyer

DECLARE personelCursor CURSOR FOR
SELECT ad, soyad FROM personel WHERE personel_no < 5;

OPEN personelCursor;

DECLARE @ad NVARCHAR(50);
DECLARE @soyad NVARCHAR(50);

FETCH NEXT FROM personelCursor INTO @ad, @soyad;

WHILE @@FETCH_STATUS = 0
BEGIN
    PRINT @ad + ' ' + @soyad;
    FETCH NEXT FROM personelCursor INTO @ad, @soyad;
END;

CLOSE personelCursor;
DEALLOCATE personelCursor;
