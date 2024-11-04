select ad, soyad, maas, 'durum'=
case
when maas<3000 then 'dusuk gelir'
when maas>4500 then 'orta gelir'
else 'yuksek gelir'
end 
from personel order by durum