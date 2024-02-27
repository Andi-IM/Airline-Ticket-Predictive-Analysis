# Laporan Proyek Machine Learning - Andi Irham M

## Domain Proyek

Pesawat terbang merupakan mode transportasi yang digunakan untuk berpergian dari suatu tempat ke tempat lain yang memiliki jarak yang jauh. Pesawat terbang memiliki keunggulan dengan daya jelajah yang melebihi moda transportasi lainnya seperti kereta api dan kapal yang membutuhkan perairan untuk berlayar. Banyaknya permintaan akan penggunaan pesawat terbang, membuat munculnya pada kebutuhan akan permintaan penggunaan pesawat terbang yang juga berdampak pada harga tiket pesawat yang bervariasi, bergantung pada maskapai yang digunakan. Namun karena harga tiket pesawat yang harganya lebih mahal dibandingkan mode transportasi lainnya membuat pengguna lebih mempertimbangkan harga dari tiket pesawat[^1].

Harga tiket pesawat sangat bergantung dengan beberapa faktor, seperti maskapai yang digunakan, waktu penerbangan, kota, dan kelas penerbangan. Oleh karena itu, dengan mempertimbangkan beberapa faktor tersebut, melalui eksplorasi dataset yang tersedia, maka dapat diperkirakan harga dari tiket pesawat untuk melihat seberapa besar korelasi pengaruh faktor-faktor tersebut.

Salah satu solusi yang dapat digunakan untuk memprediksi harga tiket pesawat adalah dengan menggunakan teknik analisis data yang disebut regresi. Dengan menggunakan regresi dan memasukkan faktor-faktor penentu harga tiket pesawat diharapkan dapat memprediksi harga tiket yang diinginkan [^2], [^3].

## Business Understanding

Pengembangan model prediksi harga tiket pesawat memiliki potensi atau dampak berupa jadi faktor penentu keputusan oleh calon penumpang. Prediksi yang akurat akan membuat calon penumpang yakin dengan nilai dari suatu harga tiket yang ditawarkan.

### Problem Statement

Berasarkan dari kondisi yang telah diuraikan sebelumnya, maka diperlukan pengembangan sistem yang dapat memprediksi kemungkinan terjadinya kebakaran hutan dengan menjawab permasalahan berikut:

- Apakah harga bervariasi tergantung dengan maskapai penerbangan?
- Bagaimana harga tiket terpengaruh ketika dibeli hanya 1 atau 2 hari sebelum keberangkatan?
- Apakah harga tiket berubah berdasarkan waktu keberangkatan dan waktu kedatangan?
- Bagaimana harga berubah dengan perubahan kota asal dan kota tujuan?
- Bagaimana perbedaan harga tiket antara Kelas Ekonomi dan Kelas Bisnis?
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh dalam memprediksi harga tiket pesawat?
- Bagaimana mengolah dataset sedemikian rupa agar dapat dibuat model prediksi harga tiket pesawat?
- Seberapa tinggi kemungkinan terjadinya kebakaran berdasarkan karakteristik atau fitur tertentu?

### Goal Statement

Untuk menjawab problem tersebut, maka akan dibuat predictive modeling dengan tujuan atau goals sebagai berikut:

- Mengetahui maskapai apa yang mempengaruhi harga tiket.
- Mengetahui harga tiket jika dibeli pada waktu 1 atau 2 hari menjelang keberangkatan.
- Mengetahui pengaruh pemilihan waktu keberangkatan dan kedatangan dengan harga tiket pesawat.
- Mengetahui pengaruh pemilihan kota asal dan tujuan terhadap perubahan harga tiket pesawat.
- Mengetahui perbedaan harga tiket kelas Ekonomi dan Bisnis.
- Mengetahui fitur yang paling penting dan berkolerasi dengan nilai kemungkinan terjadinya kebakaran hutan berdasarkan data citra.
- Melakukan proses *data wragling* dan *data preparation* agar dapat dijalankan pada model machine learning.
- Membuat model machine learning yang dapat memprediksi kemungkinan terjadinya kebakaran berdasarkan fitur-fitur yang ada.  

### Solution Statement

Solusi yang dapat menjawab permasalahan dan tujuan adalah sebagai berikut:

- Eksplorasi fitur yang terdapat pada dataset dengan menggunakan teknik analisis univariat dan multivariat. Analisis univariat digunakan untuk melihat hubungan data. Analisis multivariat dilakukan untuk melihat hubungan antar fitur. Visualisasis dengan plot juga digunakan untuk memudahkan dalam penentuan fitur mana yang berguna, salah satunya menggunakan heatmap untuk melihat korelasi dari setiap fitur yang dimiliki.
- Mepersiapkan data dengan melakukan proses Data Wragling yang meliputi Data Gathering, Data Assessing dan Data Cleaning. 
- Menggunakan metode *Regresi* dengan memanfaatkan algoritma machine learning seperti KNN, Random Forest, Boosting, dan SVM.
- Mengunakan ***Root Mean Squared Error*** sebagai metrik untuk melihat akurasi dari model yang akan dibangun.  

## Data Understanding

Dataset yang digunakan dalam analisis kali ini adalah [*Flight Price Prediction*](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction/data) yang merupakan data penerbangan yang diambil melalui scapping data dari website dari tanggal 11 - 31 Maret 2022. yang terdokumentasi melalui platform [kaggle.com](kaggle.com).

Detail dari file ini adalah sebagai berikut:

- Dataset terdiri dari 300.153 *records* dengan 12 fitur.
- Dataset terdiri dari 8 data kategori dan 4 data numerik.
- Tidak ada data yang missing.

### Dataset memiliki fitur-fitur sebagai berikut:

- Airline: Nama Perusahaan Penerbangan.
- Flight: Informasi kode penerbangan
- Source City: Kota dari mana pesawat take off
- Departure Time: Informasi waktu keberangkatan.
- Stops: Berapa kali peasawat transit.
- Arrival Time: Informasi waktu kedatangan.
- Destination City: Kota di mana pesawat tiba.
- Class: Informasi seat class; terdapat kelas Bisnis dan Ekonomi.
- Duration: Durasi penerbangan.
- Price: Fitur target yang menyipan harga tiket.

### Explanatory Data Analysis

Untuk dapat memahami data lebih jelas, maka dilakukan analisis data melalui metode statistik yang disebut sebagai Analisis Data Eksplanatori (*Explanatory Data Analysis*) atau disingkat EDA [^4]. EDA meliputi Analisis Data Univariat dan Multivariat.

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

Terdapat 6 kategori untuk fitur departure_time, secara berurutan dari waktu pagi, pagi awal, sore, malam, siang, dan larut malam.

#### Fitur stops

```shell
             jumlah sampel  persentase
one                 249478        83.7
zero                 36004        12.1
two_or_more          12438         4.2
```

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

Untuk fitur destination_city memiliki kategori yang sama dengan source_city, namun Kota Mumbai yang paling banyak ditemui.

#### Fitur class

```shell
          jumlah sampel  persentase
Economy          204792        68.7
Business          93128        31.3
```

Untuk fitur class terdapat 2 kategori, yaitu: Ekonomi dan Bisnis. Kelas Ekonomi paling banyak ditemui dengan persentase 68%.

### Analisis Univariat Data Numerikal

<p align="center">
    <img src="https://github.com/Andi-IM/Airline-Ticket-Predictive-Analysis/assets/21165698/7d41a441-8af3-4126-81ad-490eceffbfe1" width="640px">
</p>
<p align="center"><b>Analisis univariat untuk data numerik</b></p>

Dari histogram "price" dapat diperoleh informasi antara lain:

- Peningkatan harga tiket pesawat sebanding dengan penurunan jumlah sampel. Hal ini dapat kita lihat dengan jelas dari histogram "price" yang grafiknya mengalami penurunan seiring dengan semakin banyaknya jumlah sampel (sumbu y).
- Rentang harga tiket cukup tinggi yaitu dari skala puluhan ribu rupee India hingga sekitar 100000 rupee.

### Analisis Multivariat

#### 7 Kota asal dan tujuan berdasarkan harga

|index|source\_city|destination\_city|price|
|---|---|---|---|
|4|Bangalore|Mumbai|299261197|
|25|Mumbai|Bangalore|298260354|
|14|Delhi|Mumbai|295931282|
|29|Mumbai|Kolkata|282022007|
|27|Mumbai|Delhi|277303264|
|24|Kolkata|Mumbai|253178558|
|10|Delhi|Bangalore|250537591|

Dari tabel di atas dapat dilihat bahwa pemilihan kota dapat mempengaruhi harga dari tiket pesawat

#### 5 Penerbangan dan Maskapai Teratas Berdasarkan Harga

<p align="center"><img src="https://github.com/Andi-IM/Airline-Ticket-Predictive-Analysis/assets/21165698/f895d9a9-5398-4e39-a5c2-2323b16c9ae1" width="640px"></p>

Penerbangan yang paling sering digunakan adalah UK-706 dengan Maskapai Vistara airlines.




Hasil analisis multivariat antar fitur numerikal dapat dilihat pada diagram matriks korelasi berikut ini:

<p align="center"><img src="https://github.com/Andi-IM/Airline-Ticket-Predictive-Analysis/assets/21165698/a51093c9-0d28-4c83-85e4-f8a0d85ec7ae" width="640px"></p>
<p align="center">Matriks korelasi fitur numerik</p>

Dari gambar di atas, dapat dilihat bahwa durasi memiliki korelasi yang paling tinggi terhadap relasi. Fitur days_left dapat dibuang karena memiliki korelasi yang paling lemah terhadap harga.

## Data Preparation

Pada proses *data preparation* dilakukan empat tahap persiapan data, yaitu:

- Encoding fitur kategori.
- Reduksi dimensi dengan Principal Component Analysis (PCA).
- Pembagian dataset dengan fungsi train_test_split dari library sklearn.
- Standarisasi.

## Modeling

- *Random Forest*
  
  Algoritma Random Forest adalah algoritma yang sering digunakan karena sederhana dan memiliki stabilitas yang mumpuni. Algoritma ini termasuk varian teknik *bagging*. Algoritma ini merupakan kombinasi pohon keputusan sedemikian hingga setiap pohon bergantung pada nilai vektor acak yang disampling secara independen dan dengan distribusi yang sama untuk semua pohon dalam hutan tersebut. Kekuatan random forest terletak pada seleksi fitur yang acak untuk memilah setiap *node*, yang mampu menghasilkan tingkat kesalahan relatif rendah..

## Evaluation

Proyek ini menggunakan machine learning dengan kasus regresi oleh karena itu metrik yang digunakan adalah metrik yang membandingkan hasil prediksi dengan nilai sebenarnya. Model dikatakan baik jika memiliki nilai error yang kecil atau perbandingan antara hasil prediksi dengan nilai sebenarnya tidak jauh atau mendekati.

Root Mean Squared Error atau disingkat RMSE digunakan dengan menghitung nilai akar dari rata-rata kuadrat perbedaan antara nilai prediksi dengan nilai sebenarnya di dataset. RMSE didefenisikan sebagai persamaan berikut:

$$
\begin{align}
RMSE = \sqrt{\dfrac{1}{n}\Sigma(\hat y_i - y_i)^2}
\end{align}
$$

$Keterangan:$

- $N$ = jumlah dataset
- $\hat y_i$ = nilai prediksi
- $y_i$ = nilai sebenarnya

## Kesimpulan

Dapat dilihat dari keempat model yang digunakan dapat disimpulkan model random forest memiliki nilai error yang kecil.

## References

[^1]: R. H. Pranata, “PENERAPAN ALGORITMA JARINGAN SYARAF TIRUAN UNTUK MEMPREDIKSI HARGA TIKET PESAWAT,” Jurnal Sistem Komputer Musirawas (JUSIKOM), vol. 3, no. 2, p. 122, Dec. 2018, doi: https://doi.org/10.32767/jusikom.v3i2.334.

[^2]: Agarwal, Umang & Gupta, Smriti & Goyal, Madhav. (2022). House Price Prediction using Linear Regression. 10.13140/RG.2.2.11175.62887.

[^3]: Li, Xinshu. (2022). Prediction and Analysis of Housing Price Based on the Generalized Linear Regression Model. Computational Intelligence and Neuroscience. 2022. 1-9. 10.1155/2022/3590224.

[^4]: C. Chatfield, “Exploratory data analysis,” European Journal of Operational Research, vol. 23, no. 1, pp. 5–13, Jan. 1986, doi: https://doi.org/10.1016/0377-2217(86)90209-2.
