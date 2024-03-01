# Laporan Proyek Machine Learning - Andi Irham M

## Domain Proyek

Pesawat terbang merupakan mode transportasi yang digunakan untuk berpergian dari suatu tempat ke tempat lain yang memiliki jarak yang jauh. Pesawat terbang memiliki keunggulan dengan daya jelajah yang melebihi moda transportasi lainnya seperti kereta api dan kapal yang membutuhkan perairan untuk berlayar. Banyaknya permintaan akan penggunaan pesawat terbang, membuat munculnya pada kebutuhan akan permintaan penggunaan pesawat terbang yang juga berdampak pada harga tiket pesawat yang bervariasi, bergantung pada maskapai yang digunakan. Namun karena harga tiket pesawat yang harganya lebih mahal dibandingkan mode transportasi lainnya membuat pengguna lebih mempertimbangkan harga dari tiket pesawat[^1].

Harga tiket pesawat sangat bergantung dengan beberapa faktor, seperti maskapai yang digunakan, waktu penerbangan, kota, dan kelas penerbangan. Oleh karena itu, dengan mempertimbangkan beberapa faktor tersebut, melalui eksplorasi dataset yang tersedia, maka dapat diperkirakan harga dari tiket pesawat untuk melihat seberapa besar korelasi pengaruh faktor-faktor tersebut.

Salah satu solusi yang dapat digunakan untuk memprediksi harga tiket pesawat adalah dengan menggunakan teknik analisis data yang disebut regresi. Dengan menggunakan regresi dan memasukkan faktor-faktor penentu harga tiket pesawat diharapkan dapat memprediksi harga tiket yang diinginkan[^2],[^3].

## *Business Understanding*

Pengembangan model prediksi harga tiket pesawat memiliki potensi atau dampak berupa jadi faktor penentu keputusan oleh calon penumpang. Prediksi yang akurat akan membuat calon penumpang yakin dengan nilai dari suatu harga tiket yang ditawarkan.

### *Problem Statement*

Berasarkan dari kondisi yang telah diuraikan sebelumnya, maka diperlukan pengembangan sistem yang dapat memprediksi kemungkinan terjadinya kebakaran hutan dengan menjawab permasalahan berikut:

- Apakah harga bervariasi tergantung dengan maskapai penerbangan?
- Bagaimana harga tiket terpengaruh ketika dibeli hanya 1 atau 2 hari sebelum keberangkatan?
- Apakah harga tiket berubah berdasarkan waktu keberangkatan dan waktu kedatangan?
- Bagaimana harga berubah dengan perubahan kota asal dan kota tujuan?
- Bagaimana perbedaan harga tiket antara Kelas Ekonomi dan Kelas Bisnis?
- Bagaimana harga berubah dengan perubahan jumlah pemberhentian?
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh dalam memprediksi harga tiket pesawat?
- Bagaimana mengolah dataset sedemikian rupa agar dapat dibuat model prediksi harga tiket pesawat?
- Seberapa tinggi kemungkinan terjadinya kebakaran berdasarkan karakteristik atau fitur tertentu?

### *Goal Statement*

Untuk menjawab problem tersebut, maka akan dibuat predictive modeling dengan tujuan atau goals sebagai berikut:

- Mengetahui maskapai apa yang mempengaruhi harga tiket.
- Mengetahui harga tiket jika dibeli pada waktu 1 atau 2 hari menjelang keberangkatan.
- Mengetahui pengaruh pemilihan waktu keberangkatan dan kedatangan dengan harga tiket pesawat.
- Mengetahui pengaruh pemilihan kota asal dan tujuan terhadap perubahan harga tiket pesawat.
- Mengetahui perbedaan harga tiket kelas Ekonomi dan Bisnis.
- Mengetahui pengaruh transit terhadap harga tiket pesawat.
- Mengetahui fitur yang paling penting dan berkolerasi dengan nilai kemungkinan terjadinya kebakaran hutan berdasarkan data citra.
- Melakukan proses *data wragling* dan *data preparation* agar dapat dijalankan pada model *machine learning*.
- Membuat model *machine learning* yang dapat memprediksi kemungkinan terjadinya kebakaran berdasarkan fitur-fitur yang ada.  

### *Solution Statement*

Solusi yang dapat menjawab permasalahan dan tujuan adalah sebagai berikut:

- Eksplorasi fitur yang terdapat pada dataset dengan menggunakan teknik analisis univariat dan multivariat. Analisis univariat digunakan untuk melihat hubungan data. Analisis multivariat dilakukan untuk melihat hubungan antar fitur. Visualisasi dengan plot juga digunakan untuk memudahkan dalam penentuan fitur mana yang berguna, salah satunya menggunakan heatmap untuk melihat korelasi dari setiap fitur yang dimiliki.
- Mepersiapkan data meliputi *Data Ingesting*, *Data Cleaning*, dan *Data Formating*. 
- Menggunakan metode *Regresi* dengan memanfaatkan berbagai algoritma *machine learning* seperti Regresi Linear, *Decision Tree*, *Random Forest*, dan *Boosting*.
- Melakukan evaluasi terhadap model yang dikembangkan dengan metrik evaluasi model.

## *Data Understanding*

Dataset yang digunakan dalam analisis kali ini adalah [*Flight Price Prediction*](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction/data) yang merupakan data penerbangan yang diambil melalui *scrapping* data dari website dari tanggal 11 - 31 Maret 2022. yang terdokumentasi pada platform [kaggle.com](kaggle.com).

Detail dari file ini adalah sebagai berikut:

- Dataset terdiri dari 300.153 *records* dengan 12 fitur.
- Dataset terdiri dari 8 data kategori dan 4 data numerik.
- Tidak ada data yang missing.

### Dataset memiliki fitur-fitur sebagai berikut:

- `Airline`: Nama maskapai penerbangan.
- `Flight`: Informasi kode penerbangan
- `Source City`: Kota dari mana pesawat take off
- `Departure Time`: Informasi waktu keberangkatan.
- `Stops`: Berapa kali peasawat transit.
- `Arrival Time`: Informasi waktu kedatangan.
- `Destination City`: Kota di mana pesawat tiba.
- `Class`: Informasi seat class; terdapat kelas Bisnis dan Ekonomi.
- `Duration`: Durasi penerbangan.
- `Price`: Fitur target yang menyipan harga tiket.

### *Explanatory Data Analysis* (EDA)

Untuk dapat memahami data lebih jelas, maka dilakukan analisis data melalui metode statistik yang disebut sebagai Analisis Data Eksplanatori (*Explanatory Data Analysis*) atau disingkat EDA[^4]. EDA meliputi Analisis Data Univariat dan Multivariat.

Analisis Univariat merupakan teknik menganalisis data hanya dari satu variabel. Variabel dalam kumpulan dataset mengacu pada satu fitur/kolom. Proses ini dapat dilakukan dengan menganalisis grafik atau non grafik dengan menggunakan metode statistika. Analisis Multivariat membandingkan dua atau lebih variabel. Analisis dengan teknik ini dapat mengetahui bagaimana satu fitur/kolom dapat mempengaruhi fitur lainnya.

### Analisis Univariat Data Kategorikal

#### Fitur airline

```shell
           jumlah sampel  persentase
Vistara           126917        42.6
Air_India          79601        26.7
Indigo             43120        14.5
GO_FIRST           23173         7.8
AirAsia            16098         5.4
SpiceJet            9011         3.0
```

<p align="center"><img src="https://github.com/Andi-IM/Airline-Ticket-Predictive-Analysis/assets/21165698/6fa9ca92-252a-4479-a3a3-d503708a8647"></p>
<p align="center"><b>Gambar 1. Distribusi kategori fitur airline</b></p>

Terdapat 6 kategori pada fitur airlines, secara berurutan dari jumlahnya yang paling banyak yaitu: Vistara, Air India, Indigo, GO FIRST, AirAsia, dan SpiceJet. 

#### Fitur flight

```shell
         jumlah sampel  persentase
UK-706            3116         1.0
UK-772            2711         0.9
UK-720            2630         0.9
UK-836            2532         0.8
UK-874            2422         0.8
...                ...         ...
6E-865               1         0.0
SG-9974              1         0.0
6E-2914              1         0.0
G8-705               1         0.0
SG-9923              1         0.0

[1561 rows x 2 columns]
```

<p align="center"><img src="https://github.com/Andi-IM/Airline-Ticket-Predictive-Analysis/assets/21165698/8285ff92-dd36-473d-82b7-45957d05b0c2"></p>
<p align="center"><b>Gambar 2. Distribusi kategori untuk fitur flight</b></p>

Terdapat lebih dari 1561 kategori untuk fitur flight.

#### Fitur source_city

```shell
           jumlah sampel  persentase
Delhi              61156        20.5
Mumbai             60683        20.4
Bangalore          51548        17.3
Kolkata            45841        15.4
Hyderabad          40636        13.6
Chennai            38056        12.8
```

<p align="center"><img src="https://github.com/Andi-IM/Airline-Ticket-Predictive-Analysis/assets/21165698/a82ef317-3c20-4d94-b9e5-f4711134f00f"></p>
<p align="center"><b>Gambar 3. Distribusi kategori untuk fitur source_city</b></p>

Terdapat 6 kategori untuk fitur source_city, secara berurutan dari yang paling terbanyak yaitu Kota Delhi, Mumbai, Bangalore, Kolkata, Hyderabad, dan Chennai. Sebaran kota ini terlhat merata dengan persentase tertinggi 20.5%.

#### Fitur departure_time

```shell
               jumlah sampel  persentase
Morning                70372        23.6
Early_Morning          66189        22.2
Evening                64955        21.8
Night                  47998        16.1
Afternoon              47100        15.8
Late_Night              1306         0.4
```

<p align="center"><img src="https://github.com/Andi-IM/Airline-Ticket-Predictive-Analysis/assets/21165698/4a17c8f3-8878-49e2-9d2e-8d07d7817bc2"></p>
<p align="center"><b>Gambar 4. Distribusi kategori untuk fitur departure_time</b></p>

Terdapat 6 kategori untuk fitur departure_time, secara berurutan dari waktu pagi, pagi awal, sore, malam, siang, dan larut malam.

#### Fitur stops

```shell
             jumlah sampel  persentase
one                 249478        83.7
zero                 36004        12.1
two_or_more          12438         4.2
```

<p align="center"><img src="https://github.com/Andi-IM/Airline-Ticket-Predictive-Analysis/assets/21165698/47e097ad-e606-4786-8b97-83334381aa22"></p>
<p align="center"><b>Gambar 5. Distribusi kategori untuk fitur stops</b></p>

Pada fitur stops, sekali penerbangan adalah yang paling banyak ditemui.

#### Fitur arrival_time

```shel
               jumlah sampel  persentase
Night                  90702        30.4
Evening                77217        25.9
Morning                62708        21.0
Afternoon              37938        12.7
Early_Morning          15367         5.2
Late_Night             13988         4.7
```

<p align="center"><img src="https://github.com/Andi-IM/Airline-Ticket-Predictive-Analysis/assets/21165698/b59035e8-bb7b-4f27-a690-720598814511"></p>
<p align="center"><b>Gambar 6. Distribusi kategori untuk fitur arrival_time</b></p>

Pada fitur arrival_time, terdapat 6 kategori yang sama dengan departure_time, namun malam yang paling banyak ditemui.

#### Fitur destination_city

```shell
           jumlah sampel  persentase
Mumbai             58656        19.7
Delhi              57141        19.2
Bangalore          50686        17.0
Kolkata            49138        16.5
Hyderabad          42329        14.2
Chennai            39970        13.4
```

<p align="center"><img src="https://github.com/Andi-IM/Airline-Ticket-Predictive-Analysis/assets/21165698/f6ec6623-74f5-4f52-8fc2-0be6fd5eb0d9"></p>
<p align="center"><b>Gambar 7. Distribusi kategori untuk fitur destination_city</b></p>

Untuk fitur destination_city memiliki kategori yang sama dengan source_city, namun Kota Mumbai yang paling banyak ditemui.

#### Fitur class

```shell
          jumlah sampel  persentase
Economy          204792        68.7
Business          93128        31.3
```

<p align="center"><img src="https://github.com/Andi-IM/Airline-Ticket-Predictive-Analysis/assets/21165698/d5a1ecd6-ac49-49e9-b023-cb9e1ee611b0"></p>
<p align="center"><b>Gambar 8. Distribusi kategori untuk fitur class</b></p>

Untuk fitur class terdapat 2 kategori, yaitu: Ekonomi dan Bisnis. Kelas Ekonomi paling banyak ditemui dengan persentase 68%.

### Analisis Univariat Data Numerikal

<p align="center">
    <img src="https://github.com/Andi-IM/Airline-Ticket-Predictive-Analysis/assets/21165698/7d41a441-8af3-4126-81ad-490eceffbfe1" width="640px">
</p>
<p align="center"><b>Gambar 9. Analisis univariat untuk data numerik</b></p>

Dari histogram "price" dapat diperoleh informasi antara lain:

- Peningkatan harga tiket pesawat sebanding dengan penurunan jumlah sampel. Hal ini dapat kita lihat dengan jelas dari histogram "price" yang grafiknya mengalami penurunan seiring dengan semakin banyaknya jumlah sampel (sumbu y).
- Rentang harga tiket cukup tinggi yaitu dari skala puluhan ribu Rupee India hingga sekitar 100000 rupee.

### Analisis Multivariat

#### Analisa Pengaruh Jenis Maskapai dan Penerbangan dengan Harga Tiket Pesawat 

<p align="center"><img src="https://github.com/Andi-IM/Airline-Ticket-Predictive-Analysis/assets/21165698/f895d9a9-5398-4e39-a5c2-2323b16c9ae1" width="640px"></p>
<p align="center"><b>Gambar 10. 5 penerbangan dan maskapai teratas berdasarkan harga</b></p>

Penerbangan yang paling sering digunakan adalah UK-706 dengan Maskapai Vistara airlines.

#### Analisa Pengaruh Booking Terhadap Harga Tiket Pesawat

<p align="center"><img src="https://github.com/Andi-IM/Airline-Ticket-Predictive-Analysis/assets/21165698/33fdea58-ee4b-48fd-8d0f-e369d7c08218"></p>
<p align="center"><b>Gambar 11. Distribusi hari booking dengan tanggal keberangkatan</b></p>

Pada Gambar 11 dapat dilihat terjadi peningkatan harga 20 hari sebelum penerbangan dan penurunan tajam menjelang 1 hari penerbangan menjadi 3 kali lebih murah. Oleh karena itu kembali dianalisa apakah jika diberikan diskon akan memberikan dampak terhadap harga tiket. Diskon diberikan dengan 3 skema yaitu jika menjelang 1 hari keberangkatan, maka diberikan diskon 5 persen. Jika diatas 1 hari dan dibawah 8 hari sebesar 7%, lalu selebihnya hingga 15 hari sebesar 10%, diatas 15 hari tidak diberikan diskon karena harga yang stabil di atas 15 hari. Pemberian diskon memberikan hasil yang ditampilkan pada grafik di bawah ini.

<p align="center"><img src="https://github.com/Andi-IM/Airline-Ticket-Predictive-Analysis/assets/21165698/18850704-bf76-4824-a175-5d4031197e74"></p>
<p align="center"><b>Gambar 12. Distribusi penawaran yang dapat diberikan ke pelanggan</b></p>

Dari Gambar 12 menunjukkan ternyata pemberian diskon 1 hari menjelang penerbangan tidak begitu memberikan dampak terhadap harga karena hanya 0.6% yang mendapatkan diskon tersebut.

#### Analisa Waktu Keberangkatan dan Waktu Kedatangan Terhadap Harga Tiket Pesawat

<p align="center"><img src="https://github.com/Andi-IM/Airline-Ticket-Predictive-Analysis/assets/21165698/0f22cdae-5014-425e-af38-41004602cdb7"></p>
<p align="center"><b>Gambar 13. Distribusi boxplot waktu keberangkatan dan waktu kedatangan</b></p>

Berdasarkan distribusi pada Gambar 13, terjadi peningkatan harga pada pembelian tiket untuk keberangkatan di waktu malam dan kedatangan di waktu sore. Sangat disarankan untuk membeli tiket dengan waktu keberangkatan di tengah malam. 

#### Analisa Kota Asal dan Destinasi Penerbangan Terhadap Harga Tiket Pesawat

<p align="center"><img src="https://github.com/Andi-IM/Airline-Ticket-Predictive-Analysis/assets/21165698/5768fd66-f392-4a7e-940d-d12046d26fe0"></p>
<p align="center"><b>Gambar 14. Distribusi kota asal dan destinasi</b></p>

Berdasarkan grafik pada Gambar 14, terlihat bahwa penerbangan yang berangkat dan tiba di Kota Delhi umumnya lebih murah, sedangkan penerbangan yang berangkat dan tiba di Kota Bangalore harganya lebih mahal.

#### Analisa Harga Tiket Kelas Bisnis dan Kelas Ekonomi.

<p align="center"><img src="https://github.com/Andi-IM/Airline-Ticket-Predictive-Analysis/assets/21165698/f4d441ce-7a95-446d-a086-66dc4d3b6313"></p>
<p align="center"><b>Gambar 15. Distribusi tiket kelas Ekonomi dan Bisnis</b></p>

Pada Gambar 15, dapat dilihat bahwa Air India dan Vistara tergolong yang termahal, sedangkan AirAsia menawarkan harga yang paling bersahabat. Khusus untuk kelas bisnis, Vistara mematok harga tertinggi dibanding maskapai lain, termasuk AirAsia.

#### Analisa Jumlah Pemberhentian terhadap Harga Tiket Pesawat.

<p align="center"><img src="https://github.com/Andi-IM/Airline-Ticket-Predictive-Analysis/assets/21165698/064ca4ea-78d3-4dad-b7f0-dcf91c1e9119"></p>
<p align="center"><b>Gambar 16. Distribusi kota asal dan destinasi</b></p>

Dari data pada gambar 16, dapat dilihat pemberhentian sekali atau penerbangan dengan transit sekali adalah yang paling banyak jika dibandingkan dengan penerbangan dua kali transit dan tanpa transit. Lalu grafik dibawah ini untuk mengetahui maskapai mana yang memiliki transit sekali paling banyak.

<p align="center"><img src="https://github.com/Andi-IM/Airline-Ticket-Predictive-Analysis/assets/21165698/efbb8aae-d191-4635-9e83-e2b957ef7988"></p>
<p align="center"><b>Gambar 17. Distribusi kota asal dan destinasi</b></p>

Gambar 17 menunjukkan Vistara Airline dan Air india memiliki penerbangan dengan sekali transit paling banyak.

## Data Preparation

*Data preparation* atau *data preprocessing* adalah teknik yang digunakan untuk mengubah data mentah dalam format yang berguna dan efisien[^5]. Fungsi utama dari *data preparation* adalah untuk memastikan bahwa data mentah yang akan diproses sudah akurat yang berimplikasi pada hasil analitik yang valid. Proses *data preparation* dilakukan empat tahap persiapan data, yaitu Data Ingestion, Data Cleaning, dan Data Formating. Pada tahap *Data Ingestion*, berikut beberapa pengecekan yang dilakukan:

- Mengimport data dari format CSV ke bentuk DataFrame dengan library Pandas.
- Membaca informasi data.
- Mencari duplikasi data.
- Mencari missing value atau nilai yang hilang.
- Mencari outliers atau data yang menyimpang dari distribusi data. 

 Pada tahap *Data Cleaning*, ada beberapa metode yang dapat digunakan yaitu:

- Membuang baris data yang memiliki nilai kosong (*Dropping*)
- Mengisi nilai-nilai yang hilang (*Imputation*)
- Interpolasi menghasilkan titik-titik data baru dalam jangkauan suatu data.

Pada kasus proyek ini tidak ditemukan *missing value* maupun data duplikat. Outlier merupakan sampel data yang nilainya sangat jauh dari cakupan umum data utama[^6] sehingga perlu untuk dibuang salah satunya dengan dengan metode IQR. IQR (Inter Quartile Range) adalah metode menghapus outlier yang berada di luar jangkauan kuartil pertama dan kuartil ketiga sehingga IQR dirumuskan menjadi:

$$ IQR = Q3 - Q1 $$

Dimana Q1 adalah kuartil pertama dan Q3 adalah kuartil ketiga. Proyek ini terdapat outlier pada durasi penerbangan seperti yang dilihat pada Gambar 18:

<p align="center"><img src="https://github.com/Andi-IM/Airline-Ticket-Predictive-Analysis/assets/21165698/ddae7578-ebd0-4d8e-bc28-f3193e37ab55" width="640px"></p>
<p align="center"><b>Gambar 18. Visualisasi Boxplot untuk melihat outlier</b></p>

Setelah outlier dihilangkan maka jumlah data berkurang menjadi 297.920 sampel.

Pada data formatting, data yang bersifat kategorikal diubah menjadi numerik dengan tujuan untuk mempersiapkan data yang dapat dijalankan oleh model machine learning dengan optimal dan mengamankan data untuk mencegah akses yang tak diizinkan[^7]. Salah satu bentuk teknik yang digunakan adalah Label Encoding yang mengubah kategori secara secara berurutan sesuai dengan posisinya. Library yang dapat digunakan adalah LabelEncoder dari sklearn. Sehingga bentuk data yang telah diubah menjadi seperti ini:

**Tabel 1. Bentuk Format yang sudah di-encode menggunakan LabelEncoder**
|index|airline|source\_city|departure\_time|stops|arrival\_time|destination\_city|class|duration|days\_left|price|
|---|---|---|---|---|---|---|---|---|---|---|
|0|4|2|2|2|5|5|1|2\.17|1|5953|
|1|4|2|1|2|4|5|1|2\.33|1|5953|
|2|0|2|1|2|1|5|1|2\.17|1|5956|
|3|5|2|4|2|0|5|1|2\.25|1|5955|
|4|5|2|4|2|4|5|1|2\.33|1|5955|

Data akan dibagi menjadi 2 kelompok besar yatu dataset training dan dataset testing menggunakan library sklearn.preprocesing. Rasio yang digunakan untuk pemabgian adalah 70:30 sehingga dataset menjadi:

- 210.107 sampel untuk dataset train
- 90.046 sampel untuk dataset test

## Modeling

Kasus yang sedang dipahami pada proyek ini adalah mencari tahu hubungan antara kolom harga dengan kolom-kolom lain yang memperngaruhi harga sehingga pemodelan regresi adalah metode yang tepat. Regresi adalah proses indetifikasi relasi dan pengaruhnya pada nilai-nilai objek yang bertujuan untuk menemukan suatu fungsi yang memodelkan data dengan meminimalkan error[^8]. 

Ada banyak variasi dari model regresi dan berikut model yang akan dikembangkan dan diuji pada proyek ini:

- **Regresi Linear**

    Regresi Linar adalah teknik analisis data yang memprediksi nilai data yang tidak diketahui dengan menggunakan data lain yang terkait dan diektahui dimana secara matematis dimodelkan sebagai persamaan linear. Regresi linear mencoba untuk memodelkan hubungan antara dua variabel dengan mencocokkan persama linier dengan data yang diamati[^9]. Satu variabel dianggap sebagai penjelas dan yang lainnya dianggap sebagai variabel dependen. Regresi linear memiliki keuntungan untuk permasalahan yang memiliki hubungan linear karena algoritma ini adalah yang paling kompleks dibandingkan yang lain yang juga mencoba menemukan hubungan antar variabel independen dan dependen.  Regresi Linear akan menjadi kelemahan jika hubungan antar variabel tidak linier. Selain itu hasil prediksi regresi linear merupakan nilai estimasi sehingga kemungkinan tidak sesuainya sangat tinggi. 

- **Decision Tree**

  Decision Tree merupakan satu dari banyaknya pendekatan praktikal *supervised learning*. Metode Decision Tree ini dapat digunakan untuk tugas regresi maupun tugas klasifikasi dengan praktis. ALgoritma ini mengubah fakta yang sangat besar menjadi pohon keputusan yang merepresentasikan aturan. Pohon keputusan ini juga berguna untuk mengeksplorasi data, menemukan hubungan tersembunyi antara sejumlah calon variabel input dan sebuah variabel target[^10],[^11]. Decision tree memiliki kelebihan dan kelemahan sebagai berikut[^12]: 

  Kelebihan Decision Tree:

  - Dapat digunakan untuk masalah klasifikasi dan regresi dengan baik.
  - Dapat menangkap hubungan non-linear
  - Dapat menangkap informasi tanpa harus melakukan transformasi data.
  - Berguna dalam eksplorasi data. 

  Kekurangan Decision Tree:

  - Kurang mampu menggeneralisir apabila data berjumlah jutaan
  - Kurang cocok untuk dataset dengan banyak fitur
  - Mudah overfit
      
- **Random Forest**
  
  Algoritma Random Forest adalah algoritma yang sering digunakan karena sederhana dan memiliki stabilitas yang mumpuni. Algoritma ini termasuk varian teknik *bagging*[^13]. Algoritma ini merupakan kombinasi dari decision tree sedemikian hingga setiap pohon bergantung pada nilai vektor acak yang disampling secara independen dan dengan distribusi yang sama untuk semua pohon dalam hutan tersebut. Kekuatan random forest terletak pada seleksi fitur yang acak untuk memilah setiap *node*, yang mampu menghasilkan tingkat kesalahan relatif rendah. Meskipun ini adalah penyempurnaan dari algoritma decision tree, algoritma random forest juga memiliki beberapa kelebihan dan kekurangan antara lain[^14]:

  Kelebihan Random Forest:

  - Robustness Random Forest dapat menangani data yang mengandung kesalahan (noise) dan data tidak normal (outlier) dengan baik. Hal ini membuatnya tidak mudah overfit, dan memberikan hasil prediksi yang baik pada data baru. 
  - Akurasi Tinggi: Random Forest merupakan algoritma machine learning yang paling akurat. Ia dapat digunakan untuk masalah klasifikasi dan regresi, serta dapat bekerja denagn baik pada data kategorikal dan data kontinu.
  - Cepat: Meskipun tergolong pada algoritma kompleks, Random Forest bekerja dengan cepat dan dapat menangani kumpulan data yang besar. Proses *Training*\-nya dapat dipercepat dengan proses paralel. 
  - Memberikan informasi fitur penting: Random Forest menyediakan ukuran *feature importance*, yang berguna untuk memilih fitur penting dan memahami data.

   Kekurangan Rnadom Forest:

  - Masih bisa overfitting: Meskipun algoritma ini lebih jarang terjadi overfitting dibandingkan Decision Tree tunggal, hal ini tetap bisa terjadi jika jumlah pohon terlalu banyak atau pohon terlalu *deep* dan kompleks.
  - Multi interpretasi: Random Forest lebih sulit dipahami dibandingkan Decision Tree tunggal karena terdiri dari banyak phon. Hal ini membuat proses memahami bagaimana algoritma menghasilkan prediksi tertentu menjadi lebih rumit. 
  - Waktu training yang lebih lama, jika jumlah dan kedalaman pohon yang tinggi.
  - Penggunaan memori yang lebih besar karena ia menyimpan banyak pohon. Ini akan menjadi masalah jika ukuran dataset sangat besar.  
     
- **Boosting**

  Boosting adalah algoritma yang bekerja dengan menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk suatu model yang kuat (strong ensemble learner). Algoritma ini muncul dari gagasan mengenai apakah algoritma yang sederhana seperti regresi linear dan decision tree dapat dimodifikasi untuk dapat meningkatkan performa. Ada beberapa algoritma boosting yang akan digunakan pada proyek ini diantaranya:
  
  - AdaBoost
 
    Adaboost adalah metode adaptive boosting yang diperkenalkan oelh Freud dan Schapire (1995)[^15]. Adaboost bekerja mengobservasi bobot dan memberi tugas bobot yang tinggi ke model yang belum dapat memahami dataset secara iteratif hingga model memiliki akurasi yang diinginkan. Kelebihan dari AdaBoost yaitu relatif lebih mudah untuk diimplementasikan dan waktu pengujian yang relatif cepat sehingga cocok dipakai dalam implementasi kondisi *real time*. Kekurangan dari AdaBoost yaitu membutuhkan hyperparameter tuning yang tepat untuk memberikan performa yang baik.

  - Gradient Boosting
    
    Gradient Boosting adalah salah satu algoritma *boosting*, dimana menghasilkan model prediksi dari *weak learner* berbentuk decision tree. Gradient boosting melatih decision tree untuk meminimalkan *loss*[^16]. Algoritma ini menggunakan pendekatan iteratif, di mana setiap iterasi bertujuan untuk meningkatkan model sebelumnya dengan menambahkan model baru. tahapan yang dilakukan pada algoritma Gradient Boosting adalah sebagai berikut:
     
    1. **Inisialisasi model**: Tahap pertama adalah inisialisasi model, model awal dibuat sebagai model konstan yang merupakan rata-rata atau median dari target variabel.
    2. **Membuat weak model**: Tahap ini weak model dibuat sebagai model yang mampu memprediksi eror dari model sebelumnya. Biasanya berupa model decision tree yang memiliki satu atau dua percabangan.
    3. **Menghitung residual error**: Setelah model lemah dibuat, residual error dihutung sebagai selisih antara nilai prediksi dari model sebelumnya dan nilai asli dari target variable.
    4. **Menyusun kembali data training**: data training diubah dengan menggunakan residual error sebagai target variable.
    5. **Membuat model baru**: Model baru dibuat dengan memprediksi residual error yang dihasilkan dari model sebelunya.
    6. **Menggabungkan model**: Model baru yang dibuat pada tahap sebelunmnya digabungkan dengan model sebelumnya untuk membentuk model yang lebih baik.
    7. **Iterasi berulang**: Tahap di atas dilakukan kembali berulang-ulang hingga mencapai kondisi berhenti yang diharapkan, seperti jumlah iterasi yang ditentukan atau tidak terdapat perubahan nilai residual error yang signifikan.
   
    Kelebihan Gradient Boosting:

    - Akurasi yang tinggi
    - Tidak memerlukan persyaratan yang ketat
    - Kecepatan komputasi yang cepat

    Kekurangan Gradient Boosting

    - Memerlukan tuning yang cermat
    - Mudah overfit
    - Memerlukan data yang banyak 
         
  - XG Boost

    Extreme Gradient Boosting (XGBoost) adlaah pengembangan lebih lanjut dari Gradient Boosting. Sama halnya dengan Gradient Boosting, XG Boost juga menggunakan algoritma Decision Tree sebagai *base learner* dan membangun ekspansi aditif dari objective function untuk meminimalkan *loss*. Namun, XGBoost memiliki skalabilitas yang lebih baik dan mampu melakukan optimasi lebih cepat dariapda Gradient Boosting[^17].
  
## Evaluation

Proyek ini menggunakan *machine learning* dengan kasus regresi oleh karena itu metrik yang digunakan adalah metrik yang membandingkan hasil prediksi dengan nilai sebenarnya. Model dikatakan baik jika memiliki nilai error yang kecil atau perbandingan antara hasil prediksi dengan nilai sebenarnya tidak jauh atau mendekati. Adapun metrik yang digunakan sebagai alat ukur performa model antara lain **MAE**, **MSE**, **MAPR**, dan **R<sup>2</sup>**[^18]. 

Mean Absolute Error atau disingkat MAE adalah rata-rata perbedaan absolut antara nilai prediksi dengan nilai sebenarnya. Sebuah model dikatakan memiliki performa baik apabila nilai MAE semakin kecil atau sama dengan 0. MAE didefenisikan sebagai persamaan berikut:

$$ MAE = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat y_i| $$

Dimana:

- N = jumlah data
- $\hat y_i$ = nilai prediksi
- $y_i$ = nilai sebenarnya

**Mean Squared Error** atau disingkat MSE adalah rata-rata dari perbedaan kuadrat antara nilai prediksi dengan nilai sebenarnya. Sebuah model dikatakan memiliki performa yang baik apabila nilai MSE semakin kecil atau sama dengan 0. MSE MSE didefenisikan sebagai persamaan berikut:

$$ MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat y_i)^2 $$

Dimana:

- N = jumlah data
- $\hat y_i$ = nilai prediksi
- $y_i$ = nilai sebenarnya


**Mean Absolute Percentage Error** atau disingkat MAPE adalah rata-rata dari selisih persentase antara nliai prediksi dan nlai aktual. Dengan kata lain, MAPE menghitung berapa rata-rata kesalahan dalam prediksi sebagai presentase aktual. Semakin kecil nilai MAPE, maka model tersebut dikatakan memliki performa yang bagus. MAPE didefenisikan sebagai persamaan berikut:

$$ MAPE = \frac{1}{N} \sum_{i=1}^{N} |\frac{\hat y_i - y_i}{y_i} | \times 100% $$

Dimana:

- N = jumlah data
- $\hat y_i$ = nilai prediksi
- $y_i$ = nilai sebenarnya

R<sup>2</sup> menjelaskan sejauh mana varians suatu variabel menjelaskan varians variabel lainnya. Dengan kata lain, R<sup>2</sup> mengukur proporsi varians variabel terikat yang dijelaskan oleh variabel bebas. R<sup>2</sup> adalah metrik populer yang digunakan dalam mengidentifikasi akurasi model dengan nilai 0 hingga 1, semakin mendekati 1 berarti model regresi memiliki performa yang baik dan sebaliknya jika mendekati 0 berarti model tidak memiliki performa yang baik. R<sup>2</sup> didefinisikan sebagai persamaan berikut:

$$ R^2 = 1 - \frac{SSE}{SST} $$

$$ SSE = \sum_{i=1}^{N} (\hat y_i - y_i)^2 $$

$$ SST = \sum_{i=1}^{N} (\bar y_i - y_i)^2 $$

Keterangan:
- SSE = Sum of Squared Error, adalah jumlah kuadrat dari perbedaan antara nilai prediksi dan nilai sebenarnya
- SST = Total Sum of Squares, adalah jumlah kuardat dari perbedaan antara nilai prediksi dan rata-rata nilai sebenarnya.
- N = jumlah data
- $\hat y_i$ = nilai prediksi
- $y_i$ = nilai sebenarnya
   
Tabel di bawah ini merupakan perbandingan dari masing-masing model

**Tabel 2. Hasil Evaluasi Model Machine Learning**
|Model|MAE|MSE|MAPR|R2 Squared|
|:---|---|---|---|---|
|Linear Regression|4633\.650914958173|48327818\.91012481|0\.4401630312483704|0\.9060025119113978|
|Decision Tree Regressor|2533\.197494930205|20060289\.411819987|0\.17047683169632916|0\.9609827867765326|
|Random Forest Regressor|2447\.7819146012303|18874886\.723902855|0\.16557343378561687|0\.9632883920686917|
|Gradient Boosting Regressor|2963\.104288124166|24177320\.326169077|0\.21118401455007288|0\.9529751718446106|
|AdaBoostRegressor|3634\.494092801859|33299409\.99753702|0\.30562280935598|0\.9352327300261173|
|XGBRegressor|**2041\.2474563602389**|**12286355\.686425244**|**0\.15052727511266395**|**0\.9761030686190323**|

Berdasarkan tabel di atas dapat dilihat bahwa XGB Regressor menampilkan performa yang paling baik dengan nilai R<sup>2</sup> sebesar 0.98.

Secara lebih jauh perbandingan metrik untuk masing-masing model dapat dilihat pada gambar berikut ini.

<p align="center"><img src="https://github.com/Andi-IM/Airline-Ticket-Predictive-Analysis/assets/21165698/3a19f161-aaff-4bbc-8d06-b9ded0e2ed1c" width="640px"></p>
<p align="center"><b>Gambar 19. Visualisasi perbandingan evaluasi masing-masing model.</b></p>

Selain itu dilakukakn perbandingan nilai y_true terhadap prediksi harga tiket pesawat dari 6 model yang telah dibuat. Seperti yang dapat dilihat pada tabel di bawah ini:

**Tabel 3. Hasil Prediksi Model Machine Learning**
|index|y\_true|Linear Regression|Decision Tree Regressor|Random Forest Regressor|Gradient Boosting Regressor|AdaBoostRegressor|XGBRegressor|
|---|---|---|---|---|---|---|---|
|0|7426|5943\.08|5295\.59|5244\.35|4906\.64|5559\.15|5356\.7001953125|
|1|4179|5348\.5|5295\.59|5244\.35|5063\.07|5559\.15|5105\.1201171875|
|2|6542|2259\.42|5295\.59|5336\.71|4757\.33|5559\.15|5294\.39990234375|
|3|9105|9936\.22|11302\.31|11373\.39|9946\.43|10731\.9|10736\.3603515625|
|4|8011|5396\.36|5295\.59|5336\.71|4696\.73|5559\.15|6237\.490234375|

Untuk mempermudah analisis, kita bisa lihat perbandingannya pada Gambar 20:

<p align="center"><img src="https://github.com/Andi-IM/Airline-Ticket-Predictive-Analysis/assets/21165698/f4482f7e-aff5-4d8d-9067-f06cc24af205"></p>
<p align="center"><b>Gambar 20. Visualisasi perbandingan hasil prediksi dengan data testing masing-masing model.</b></p>

Berdasarkan hasil evaluasi, dapat dilihat bahwa prediksi harga tiket pesawat dengan Extreme Gradient Boost Regressor memberikan hasil yang paling mendekati y_true dapat dilihat dari visualisasi di atas yang mendominasi. 

## Kesimpulan

Dari hasil evaluasi diatas, dapat disimpulkan bahwa: 

- Maskapai memiliki pengaruh besar terhadap harga tiket pesawat adalah Vistara Airlines dengan Penerbangan tersibuknya, UK-706
- Booking tiket pesawat pada 1 hingga 2 hari menjelang keberangkatan sangat sedikit dengan persentase 0.6% sehingga tidak diperlukan pemberian diskon karena tidak begitu mempengaruhi jumlah penumpang. 
- Waktu keberangkatan dan kedatangan mempengaruhi harga tiket dengan rata-rata termurah di waktu tengah malam.
- Kota asal dan tujuan juga mempengaruhi, dengan harga tiket dari dan ke Kota Delhi yang paling murah dan harga tiket dari dan ke kota Bangalore yang paling mahal.
- Pada data umumnya orang-orang membeli tiket dengan sekali transit dan dengan maskapai Vistara dan Air India yang terbanyak.
- Dari banyak algoritma regresi yang dikembangkan, Aglrotitma Extreme Gradient Boost atau XG Boost memiliki performa yang baik.

## References

[^1]: R. H. Pranata, “PENERAPAN ALGORITMA JARINGAN SYARAF TIRUAN UNTUK MEMPREDIKSI HARGA TIKET PESAWAT,” Jurnal Sistem Komputer Musirawas (JUSIKOM), vol. 3, no. 2, p. 122, Dec. 2018, doi: https://doi.org/10.32767/jusikom.v3i2.334.

[^2]: Agarwal, Umang & Gupta, Smriti & Goyal, Madhav. (2022). House Price Prediction using Linear Regression. 10.13140/RG.2.2.11175.62887.

[^3]: Li, Xinshu. (2022). Prediction and Analysis of Housing Price Based on the Generalized Linear Regression Model. Computational Intelligence and Neuroscience. 2022. 1-9. 10.1155/2022/3590224.

[^4]: C. Chatfield, “Exploratory data analysis,” European Journal of Operational Research, vol. 23, no. 1, pp. 5–13, Jan. 1986, doi: https://doi.org/10.1016/0377-2217(86)90209-2.

[^5]: Laraswati. (2022). Tahapan Data Preparation agar Data Lebih Mudah Diproses, diakses pada tanggal 28 Februari 2024, https://blog.algorit.ma/data-preparation/ 

[^6]: Max Kuhn. (2013). Applied Predictive Modeling http://appliedpredictivemodeling.com/

[^7]: Chaudary R.A. (2023). An Introduction to Data Encoding and Decoding in Data Science. Diakses pada 28 Februari 2024, https://www.sitepoint.com/data-encoding-decoding-data-science-introduction.

[^8]: Suyanto. (2022). Machine Learning Tingkat Dasar dan Lanjut.

[^9]: Iqbal M. (2019). Pengertian Regresi Linear serta Keuntungan dan Kerugian.

[^10]: Gurucharan M. K. (2020). Machine Learning Basics: Decision Tree Regression. Diakses 29 Februari 2024. https://towardsdatascience.com/machine-learning-basics-decision-tree-regression-1d73ea003fda

[^11]: Pangestu, A. (2020). Application Based of E-Commerce Poverty Prediction Data Processing. 6(2), pp. 1729-1740.

[^12]: Educba (2023). Decision Tree Advantages and Disadvantages. diakses pada 29 Februari 2024. https://www.educba.com/decision-tree-advantages-and-disadvantages/

[^13]: Kelleher, John D, et al. "Machine Learning for Predictive Data Analytics". MIT Press. 2020. Tersedia: [tautan informasi buku](https://machinelearningbook.com/).

[^14]: Wang Y. (2023), "What Are The Advantages And Disadvantages Of Random Forest?". Rebelion Research. Tersedia: [tautan informasi](https://www.rebellionresearch.com/what-are-the-advantages-and-disadvantages-of-random-forest).

[^15]: Kelleher, John D, et al. "Machine Learning for Predictive Data Analytics". MIT Press. 2020. Tersedia: tautan informasi buku.

[^16]: Trivusi. (2023) Gradient Boosting: Pengeritan, Cara Kerja dan Kegunaannya. Diaskes 29 Januari 2024. Terseida: https://www.trivusi.web.id/2023/03/algoritma-gradient-boosting.html

[^17]: Chen,  T.,  &  Guestrin,  C.  (2016).  XGBoost:  A  Scalable  Tree  Boosting  System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794. https://doi.org/10.1145/2939672.2939785

[^18]: Shweta Goyal. (2021). Evaluation Metrics for Regression Models. diakses pada tanggal 28 Februari 2024, https://medium.com/analytics-vidhya/evaluation-metrics-for-regression-models-c91c65d73af
