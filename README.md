# Laporan Proyek Machine Learning - Andi Irham M

## Domain Proyek

Pesawat terbang merupakan mode transportasi yang digunakan untuk berpergian dari suatu tempat ke tempat lain yang memiliki jarak yang jauh. Pesawat terbang memiliki keunggulan dengan daya jelajah yang melebihi moda transportasi lainnya seperti kereta api dan kapal yang membutuhkan perairan untuk berlayar. Banyaknya permintaan akan penggunaan pesawat terbang, membuat munculnya pada kebutuhan akan permintaan penggunaan pesawat terbang yang juga berdampak pada harga tiket pesawat yang bervariasi, bergantung pada maskapai yang digunakan. Namun karena harga tiket pesawat yang harganya lebih mahal dibandingkan mode transportasi lainnya membuat pengguna lebih mempertimbangkan harga dari tiket pesawat[^1].

Harga tiket pesawat sangat bergantung dengan beberapa faktor, seperti maskapai yang digunakan, waktu penerbangan, kota, dan kelas penerbangan. Oleh karena itu, dengan mempertimbangkan beberapa faktor tersebut, melalui eksplorasi dataset yang tersedia, maka dapat diperkirakan harga dari tiket pesawat untuk melihat seberapa besar korelasi pengaruh faktor-faktor tersebut.

Salah satu solusi yang dapat digunakan untuk memprediksi harga tiket pesawat adalah dengan menggunakan teknik analisis data yang disebut regresi. Dengan menggunakan regresi dan memasukkan faktor-faktor penentu harga tiket pesawat diharapkan dapat memprediksi harga tiket yang diinginkan [^2], [^3].

## Business Understanding

Pengembangan model prediksi harga tiket pesawat memiliki potensi atau dampak berupa jadi faktor penentu keputusan oleh calon penumpang. Prediksi yang akurat akan membuat calon penumpang yakin dengan nilai dari suatu harga tiket yang ditawarkan.

### Problem Statement

Berasarkan dari kondisi yang telah diuraikan sebelumnya, maka diperlukan pengembangan sistem yang dapat memprediksi kemungkinan terjadinya kebakaran hutan dengan menjawab permasalahan berikut:

- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh dalam memprediksi kemungkinan kebakaran hutan?
- Bagaimana mengolah dataset sedemikian rupa agar dapat dibuat model prediksi harga tiket pesawat?
- Seberapa tinggi kemungkinan terjadinya kebakaran berdasarkan karakteristik atau fitur tertentu?

### Goal Statement

Untuk menjawab problem tersebut, maka akan dibuat predictive modeling dengan tujuan atau goals sebagai berikut:

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

Untuk dapat memahami data lebih jelas, maka dilakukan analisis data melalui metode statistik yang disebut sebagai Analisis Data Eksplanatori (*Explanatory Data Analysis*) atau disingkat EDA [^2]. EDA meliputi Analisis Data Univariat dan Multivariat.

Analisis Univariat merupakan teknik menganalisis data hanya dari satu variabel. Variabel dalam kumpulan dataset mengacu pada satu fitur/kolom. Proses ini dapat dilakukan dengan menganalisis grafik atau non grafik dengan menggunakan metode statistika. Analisis Multivariat membandingkan dua atau lebih variabel. Analisis dengan teknik ini dapat mengetahui bagaimana satu fitur/kolom dapat mempengaruhi fitur lainnya.

Berikut adalah hasil EDA, dimana Gambar 1 merupakan EDA Analisis Univariat dan Gambar 2 merupakan EDA Analisis Multivariat.

## Data Preparation

Pada proses *data preparation* dilakukan empat tahap persiapan data, yaitu:

- Encoding fitur kategori.
- Reduksi dimensi dengan Principal Component Analysis (PCA).
- Pembagian dataset dengan fungsi train_test_split dari library sklearn.
- Standarisasi.

## Modeling

Model yang digunakan untuk memprediksi kemungkinan kebakaran berdasarkan data satelit antara lain:

- *K-Nearest Neighbors*
  
  K-Nearest Neighbours (kNN) adalah algoritma yang paling simple. Metode ini bekerja dengan cara mencari sejumlah *k* pola (di antara semua pola latih yang ada di semua kelas) yang terdekat dengan pola masukan, kemudian menentukan kelas keputusan berdasarkan jumlah pola terbanyak di antara *k* pola tersebut (voting). KNN dapat digunakan untuk kasus klasifikasi maupun regresi.

   **Tahapan Cara Kerja kNN**
  - Menentukan jumlah tetangga terdekat *k*.
  - Menghitung jarak antara data testing ke data training.
  - Mengurutkan data berdasarkan data yang mempunyai jarak terkecil (bisa menggunakan manhattan, eucledian ataupun minkowski)
  - Menentukan kelompok testing berdasarkan label pada *k*.

  Pada proyek ini menggunakan *n_neighbors = 50* dengan catatan pemilihan nilai *k* sangat penting dan berpengaruh dengan performa model. Metrik jarak juga memiliki keunggulan masing-masing, dan di proyek ini akan menggunakan metode Eucledian untuk menghitung jarak. Dan metode evaluasi selanjutnya akan dibahas pada [Evaluasi Model](#evaluasi-model) .

  **Kelebihan kNN**
  - *Mudah digunakan* dengan kompleksitas algoritma yang tidak sebegitunya tinggi.
  - *Mudah beradaptasi* - Algoritma ini menyimpan seluruh data dalam penyimpanan memori. Ketika sebuah contoh baru atau titik data ditambahkan, kNN secara otomatis menyesuaikan diri berdasarkan contoh baru dan turut berkontribusi pada prediksi masa depan.
  - *Sedikit pengaturan hyperparameter* -  Dalam training algoritma ini hanya memerlukan parameter k

  **Kekurangan kNN**
  - *Tidak dapat diskalakan* - Algoritma kNN sering juga dikatakan algoritma pemalas (*Lazy Algorithm*) karena ia tidak secara eksplisit mempelajari model dari data pelatihan. Akibatnya, kNN membutuhkan daya komputasi dan penyimpanan data selama fase prediksi. Hal ini membuat kNN memakan waktu dan sumber daya.
  - *Kutukan Dimensionalitas* - Algoritma kNN rentan terhadap "fenomena puncak" yang terkait dengan kutukan dimensionalitas. Ini berarti kNN menghadapi kesulitan dalam mengklasifikasikan titik data secara akurat ketika dimensi data menjadi terlalu tinggi.
  - *Rentan Overfit* - karena rentan dengan "Kutukan dimensionalitas", algoritma kNN juga rentan dengan masalah overfitting. Oleh karena itu, teknik pemilihan fitur dan reduksi dimensionalitas umumnya diterapkan untuk masalah ini.

- *Random Forest*
  
  Algoritma Random Forest adalah algoritma yang sering digunakan karena sederhana dan memiliki stabilitas yang mumpuni. Algoritma ini termasuk varian teknik *bagging*. Algoritma ini merupakan kombinasi pohon keputusan sedemikian hingga setiap pohon bergantung pada nilai vektor acak yang disampling secara independen dan dengan distribusi yang sama untuk semua pohon dalam hutan tersebut. Kekuatan random forest terletak pada seleksi fitur yang acak untuk memilah setiap *node*, yang mampu menghasilkan tingkat kesalahan relatif rendah..
- *Boosting*

  Boosting dikemukakan oleh Robert E. Schapire pada tahun 1990. Sesuai dengan namanya, metode boosting bekerja dengan cara memperkuat (*boost*) sebuah model klasifikasi awal yang lemah, secara sekuensial menggunakan penyamplingan objek data bootstrap berdasarkan pembobotan dinamis.
  Algoritma boosting sudah ada sejak puluhan tahun lalu. Kembali terkenal sejak adanya peningkatan dalam kompetisi machine learning atau data science. Algoritma ini sangat powerful dalam meningkatkan akurasi prediksi. Algoritma boosting sering mengungguli model yang lebih sederhana seperti logistic regression dan random forest. Beberapa pemenang kompetisi kaggle menyatakan bahwa mereka menggunakan algoritma boosting atau kombinasi beberapa algoritma boosting dalam modelnya. Meskipun demikian, hal ini tetap bergantung pada kasus per kasus, ruang lingkup masalah, dan dataset yang digunakan.
Dilihat caranya berkembang, algoritma boosting terdiri dari dua metode:
  - Adaptive boosting
  - Gradient boosting
- *SVM*

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

[^2]: C. Chatfield, “Exploratory data analysis,” European Journal of Operational Research, vol. 23, no. 1, pp. 5–13, Jan. 1986, doi: https://doi.org/10.1016/0377-2217(86)90209-2.

[^3] Agarwal, Umang & Gupta, Smriti & Goyal, Madhav. (2022). House Price Prediction using Linear Regression. 10.13140/RG.2.2.11175.62887.

[^4] Li, Xinshu. (2022). Prediction and Analysis of Housing Price Based on the Generalized Linear Regression Model. Computational Intelligence and Neuroscience. 2022. 1-9. 10.1155/2022/3590224.
