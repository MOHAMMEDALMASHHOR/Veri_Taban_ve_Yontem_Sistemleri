﻿use sirket
select * from personel
--Cannot insert explicit value for identity column in table 'personel' when IDENTITY_INSERT is set to OFF
--insert into personel (ad,soyad,cinsiyet,dogum_tarihi,dogum_yeri,baslama_tarihi,calisma_saati,maas,personel_no)values ('Muammer','Türkoğlu','E','1990-06-28',3,'2002-01-12',30,3000,250)--Cannot insert the value NULL into column 'dogum_tarihi', table 'sirket.dbo.personel'; column does not allow nulls. INSERT fails.--INSERT INTO personel(ad, soyad, cinsiyet, calisma_saati, maas) VALUES ('Mehmet','Türkoğlu','E',40,3500)--INSERT INTO ogrenci(ogr_no,ad, soyad) VALUES (1,'Ahmet','BÜYÜK'),(2,'Mehmet','TEK'),(3,'Ayşe','YILDIZ'),(4,'HATİCE','Polat'),(5,'YAVUZ','ADA');--Update personel set ad = 'Ceydanur' where personel_no =3--Join---select * from unvanselect personel.ad,personel.soyad,proje.proje_ad from personel,gorevlendirme,proje where personel.personel_no=gorevlendirme.personel_no And gorevlendirme.proje_no=proje.proje_noselect AVG(p.maas) from personel p, unvan u where p.unvan_no=u.unvan_no and u.unvan_ad= 'MÜHENDIS'--select * from ilce --select * from ilselect p.ad,p.maas,p.calisma_saati from personel p , il, ilce c where p.dogum_yeri=c.ilce_no and c.il_no=il.il_no and il.il_ad= 'ANKARA'