# **Laporan Proyek Machine Learning - Gabril Hozanna**

## **Project Overview**
Pesatnya pertumbuhan pengumpulan data telah menyebabkan era baru informasi. Data digunakan untuk membuat sistem yang lebih efisien dan di sinilah Sistem Rekomendasi berperan. Sistem Rekomendasi adalah jenis sistem penyaringan informasi karena sistem Rekomendasi meningkatkan kualitas hasil pencarian dan menyediakan item yang lebih relevan dengan item pencarian atau terkait dengan riwayat pencarian penggunaan.

Sistem Rekomendasi digunakan untuk memprediksi peringkat atau preferensi yang akan diberikan pengguna pada suatu item. Hampir setiap perusahaan teknologi besar telah menerapkannya dalam beberapa bentuk atau yang lain: Amazon menggunakannya untuk menyarankan produk kepada pelanggan, YouTube menggunakannya untuk memutuskan video mana yang akan diputar selanjutnya di autoplay, dan Facebook menggunakannya untuk merekomendasikan halaman yang disukai dan orang untuk diikuti. Selain itu, perusahaan seperti Netflix dan Spotify sangat bergantung pada efektivitas mesin rekomendasi mereka untuk bisnis dan kesuksesan mereka. Di Netflix sendiri, menerapkan sistem rekomendasi untuk merekomendasikan film dan serial ke pengguna. Pendapatan Netflix pada kuartal II-2021 ini mengalami kenaikan 19 persen dari tahun ke tahun (YoY). Pendapatan perusahaan di tahun 2020 berada di angka `US$6,1` milliar (sekitar Rp.88,5 trilliun), naik menjadi `US$7,3` miliar (kira-kira Rp.105,9 trilliun) [InfoKomputer](https://infokomputer.grid.id/read/122837009/contoh-implementasi-teknologi-ai-di-platform-streaming-film-netflix?page=all).

Film adalah salah satu media hiburan yang populer di masyarakat. Sampai September 2021, tercatat telah ada 8,313,921 judul film yang telah rilis [IMDb Statistics](https://www.imdb.com/pressroom/stats/). Banyaknya judul-judul yang telah rilis membuat masyarakat kesulitan untuk menemukan film mana yang mereka ingin tonton. Untuk mengatasi masalah tersebut, perlu adanya informasi mengenai film yang akan memudahkan masyarakat untuk menemukan film yang cocok dengan preferensi user, oleh sebab itu user perlu sebuah sistem yang dapat memberikan rekomendasi film. 

## **Business Understanding**

### **Problem Statements**

* Bagaimana menyajikan sejumlah rekomendasi film dengan teknik content-based filtering?
* Bagaimana menyajikan sejumlah rekomendasi film dengan teknik collaborative filtering?

### **Goals**
* Untuk menyajikan sejumlah rekomendasi film dengan teknik content-based filtering
* Untuk menyajikan sejumlah rekomendasi film dengan teknik collaborative filtering

### **Solution approach**

Berdasarkan problem statements yang telah disebutkan maka dapat menggunakan teknik:
1. **Content-based filtering** memanfaatkan informasi beberapa item / data untuk direkomendasikan kepada pengguna sebagai referensi yang terkait dengan informasi yang digunakan sebelumnya. Tujuan dari content based recommendation agar dapat memprediksi persamaan dari sejumlah informasi yang didapat dari pengguna. Content based filtering menggunakan konsep perhitungan vector, TF-IDF, dan Cosine Similarity yang intinya dikonversikan dari data / text menjadi berbentuk vektor. Content based filtering membutuhkan deskripsi item/data yang baik. 

    Untuk teknik content-based filtering menggunakan `CountVectorizer`. CountVectorizer akan mengambil kata-kata dari setiap kalimat dan menciptakan vocabulary dari semua kata unik dalam kalimat. Vocabulary ini kemudian dapat digunakan untuk membuat feature vector dari jumlah kata. Perbedaan antara `TfidfVectorizer()` dengan `CountVectorizer()` adalah `TfidfVectorizer()` mengembalikan float atau `returns floats` sedangkan CountVectorizer() mengembalikan int atau `return int`.

2. **Collaborative filtering** memanfaatkan transaksi suatu produk / item yang didasarkan kepada perilaku / kebiasaan si pengguna. Tujuannya agar pengguna yang sama dan item yang serupa dapat disukai oleh pengguna sebagai rekomendasi pilihan. Collaborative Filtering membutuhkan banyak feedback dari pengguna agar sistem berfungsi dengan baik.

    Untuk teknik collaborative filtering menerapkan model `BaseLine`, `SVD (Matrix Factorization)` dan `KNN` dari library `Surpise`. Library surpise sendiri digunakan untuk sistem rekomendasi. 
    * **Baseline Model**: Ini adalah algoritma dasar (basic) yang tidak banyak bekerja tetapi masih berguna untuk membandingkan akurasi. Digunakan untuk memprediksi perkiraan dasar untuk pengguna dan item tertentu.
    * **SVD Model**: Seperti yang dipopulerkan oleh [Simon Funk](https://sifter.org/~simon/journal/20061211.html) (Netflix Prize). Algoritma ini setara dengan Probabilistic Matrix Factorization [Sumber](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#matrix-factorization-based-algorithms). Singular value decomposition (SVD) adalah metode faktorisasi matriks yang menggeneralisasikan dekomposisi eigen ([eigendecomposition atau eigenvectors/eigenvalues](https://medium.com/swlh/eigenvalues-and-eigenvectors-5fbc8b037eed)) dari matriks persegi (n x n) ke matriks apa pun (n x m) [Sumber](https://en.wikipedia.org/wiki/Singular_value_decomposition). 
    SVD mirip dengan Principal Component Analysis (PCA) tetapi lebih umum. PCA mengasumsikan bahwa inputannya berupa matriks persegi, SVD tidak memiliki asumsi ini. Secara umum rumusnya adalah `M=UΣVᵗ`. SVD dapat menangani matriks dengan jumlah kolom dan baris yang berbeda. Rumus PCA adalah `M=𝑄𝚲𝑄ᵗ` dimana menguraikan matriks menjadi matriks ortogonal 𝑄 dan matriks diagonal 𝚲. SVD melakukan hal serupa tetapi tidak kembali (return) ke basis yang sama saat memulai transformasi. Tidak bisa dilakukan karena matriks asli M bukan matriks persegi [Sumber](https://towardsdatascience.com/simple-svd-algorithms-13291ad2eef2).
    * **KNN Model**: Algoritma k-Nearest Neighbor adalah algoritma supervised
  learning dimana hasil dari instance yang baru diklasifikasikan berdasarkan mayoritas dari kategori k-tetangga terdekat. Tujuannya adalah untuk mengklasifikasikan obyek baru berdasarkan atribut dan sample-sample dari training data. Algoritma k-Nearest Neighbor menggunakan Neighborhood Classification sebagai nilai prediksi dari nilai instance yang baru.
      * Kelebihan:
        1. Sangat nonlinear, bersifat nonparametrik dimana didefinisikan sebagai model nonparametrik karena model yang tidak mengasumsikan apa-apa mengenai distribusi instance di dalam dataset.
        2. Mudah dipahami dan diimplementasikan.
      * Kekurangan:
        1. Perlu menunjukkan parameter K (jumlah tetangga terdekat).
        2. Tidak menangani nilai hilang (missing value) secara implisit.
        3. Sensitif terhadap data pencilan (outlier).
        4. Rentan terhadap variabel yang non-informatif.
        5. Rentan terhadap dimensionalitas yang tinggi.
        6. Rentan terhadap perbedaan rentang variabel.
        7. Nilai komputasi yang tinggi. [Sumber](https://informatikalogi.com/algoritma-k-nn-k-nearest-neighbor/#3)
      
## **Data Understanding**

![Dataset](https://raw.githubusercontent.com/gabrielhozana/movie_recommender/main/photo/dataset.png)
**Konteks**

Dataset [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset) berisi metadata untuk 45.000 film yang terdaftar di Dataset Full MovieLens. Kumpulan data terdiri dari film yang dirilis pada atau sebelum Juli 2017. Dataset berisi cast, crew, plot keywords, budget, revenue, posters, release dates, languages, production companies, countries, penghitungan suara TMDB, dan rata-rata vote.

Dataset ini juga memiliki file yang berisi 26 juta ratings dari 270.000 pengguna untuk 45.000 film. Ratings berada pada skala 1-5 dan telah diperoleh dari situs web resmi GroupLens.

<br>

**Konten**

Dataset ini terdiri dari file-file berikut:
* **movies_metadata.csv**: File Metadata Film utama. Berisi informasi tentang 45.000 film dari dataset Full MovieLens. Fitur terdiri dari posters, backdrops, budget, revenue, release dates, languages, production countries dan companies.

* **keywords.csv**: Berisi keyword plot film untuk film MovieLens. Tersedia dalam bentuk Objek JSON.

* **credits.csv**: Terdiri dari Informasi cast dan crew untuk semua film. Tersedia dalam bentuk Objek JSON.

* **links.csv**: File yang berisi ID TMDB dan IMDB dari semua film dari dataset Full MovieLens.

* **links_small.csv**: Berisi ID TMDB dan IMDB dari subset kecil dari 9.000 film dari dataset Lengkap.

* **ratings_small.csv**: Subset dari 100,000 ratings dari 700 pengguna di 9,000 film.


**Dataset**

Untuk dataset yang digunakan adalah `movies_metadata, keywords, credits, links_small dan ratings_small`. Dimana masing-masing dataset terdapat kolom/fitur yang berbeda. Untuk lebih jelasnya dapat dilihat dibawah ini:

1. **movies_metadata**

    Terdiri dari 45466 data namun masih terdapat banyak sekali data yang missing value (null/Nan) seperti pada kolom `overview, tagline, title`. Berikut ini adalah uraian beberapa kolom pada dataset yang akan digunakan:
    * **id**: merupakan id pada film.
    * **title**: merupakan judul pada film.
    * **original_title**: merupakan judul asli pada film
    * **tagline**: merupakan slogan atau catchphrases untuk film. Tagline dapat merujuk pada plot film.
    * **overview**: merupakan gambaran singkat terkait film.
  
2. **credits**
  
    Terdiri dari 45476 data dan tidak terdapat missing value (null/nan). Berikut ini adalah uraian kolom pada dataset.
    * **cast**: berisi terkait pemeran pada film.
    * **crew**: berisi terkait kru pada film.
    * **id**: merupakan id pada film.

3. **keywords**
  
    Terdiri dari 46419 data dan tidak terdapat missing value (null/nan). Berikut ini adalah uraian kolom pada dataset.
    * **id**: merupakan id pada film.
    * **keywords**: berisi berisi kata kunci atau keyword pada film.

4. **links_small**
  
    Terdiri dari 9125 data dan terdapat missing value pada kolom `tmdId`. Berikut ini adalah uraian kolom pada dataset.
    * **movieId**: merupakan id pada film.
    * **imdbId**: merupakan id pada database IMDB.
    * **tmdbId**: merupakan id pada database TMDB.
5. **ratings_small**
  
    Terdiri dari 100004 data dan tidak terdapat missing value (null/nan). Berikut ini adalah uraian kolom pada dataset. 
    * **userId**: merupakan id pengguna.
    * **movieId**: merupakan id pada film.
    * **rating**: merupakan rating yang diberikan oleh user terhadap film.
    * **timestamp**: berisi timestamp pada film.

    Pada kolom/fitur rating dapat dilihat pada gambar dibawah ini, dimana merupakan visualisasi dari distribusi rating film yang diberikan oleh pengguna. Jika dilihat, nilai rating film terbanyak adalah rating 4.   
    ![Rating](https://raw.githubusercontent.com/gabrielhozana/movie_recommender/main/photo/1.png)

**Exploratory Data Analysis (EDA)**

Dilakukan Exploratory Data Analysis (EDA) untuk mendapatkan sebuah insight dari dataset. Adapun tahapan yang dilakukan adalah:
* Melakukan analisis deskriptif `describe`. 
* Mengecek informasi data `info` dan missing value.
* Melakukan visualisasi data pada dataset `ratings`.

## **Data Preparation**

Teknik yang digunakan pada tahapan ini, yaitu:
1. `Drop`. Dengan menggunakan fungsi drop, maka dapat membuang atau menghapus kolom atau data yang diinginkan. T
    * Terdapat data anomali pada dataset `movies_metadata` maka perlu membuang data anomali tersebut.
    * Melakukan drop kolom/fitur `timestamp` pada dataframe `ratings` karena tidak akan dipakai.
3. `astype`. Astype berfungsi untuk mengubah tipe data. Dalam hal ini dilakukan perubahan tipe data pada dataset:
    * movies_metadata: mengubah tipe data pada kolom/fitur `id` ke int karena tipe datanya berupa object.
    * link_small: mengubah tipe data pada kolom/fitur `tmdbId` ke int karena datanya bukan berupa desimal namun tipe datanya berupa float.
4. `isin`. Isin digunakan untuk menyamakan dataframe dengan values sehingga mendapatkan output yang relevan dengan values. Pada tahap ini dilakukan penyamaan dataframe `movies_metadata` dengan:
    * `links_small` untuk mendapatkan sebuah dataframe baru dalam hal ini membuat dataframe `smd`. Jadi hanya mengambil sebagian data dari kesamaan dataset Metadata dengan Links Small.
    * `crew, cast dan keywords` untuk mendapatkan sebuah dataframe baru `smd2`.
6. `fillna`. Digunakan untuk mengisi data yang missing value atau null. Pada kolom/fitur `tagline` terdapat nan sehingga diisi dengan `''`.
7. Menggabungkan kolom/fitur pada dataframe `smd`. Tagline film adalah slogan atau catchphrases untuk film. Biasanya menyertakan permainan kata-kata yang cerdas, frasa pendek, satu atau dua kalimat. Tagline dapat merujuk pada plot film atau menyarankan pengalaman yang akan dialami sebagai penonton. Jadi, akan menggabungkan kolom/fitur tagline dengan kolom/fitur overview untuk memperoleh kolom/fitur description.
8. `merge`. Fungsi merge digunakan untuk menggabungkan dataset/dataframe. Akan menggunakan credits dan keywords, jadi akan digabungkan dataframe ini dengan dataframe metadata movies. 
9. Ekstraksi data objek (json) `smd2`. Akan mengubah kolom/fitur cast, crew, keywords menjadi hanya berisi satu list dengan cast, crew, keywords bukan dictionary.
10. Hapus Spasi dan Ubah ke Lowercase pada `smd2`. Dengan cara ini, mesin tidak akan bingung misalnya membedakan Johnny Depp dan Johnny Galecki. Dapat menggunakan fungsi `lower` dan `replace`.
11. Mention `director` sebanyak 3 kali. Mention Sutradara 3 kali untuk memberikan bobot yang lebih besar dibandingkan dengan seluruh pemeran karena sutradara mempengaruhi kualitas film lebih dari peran lainnya.
12. `Stemming` data. Jadi akan mengonversi setiap kata dengan teknik `stemming` sehingga kata-kata seperti Dog dan Dogs dianggap sama. Dalam hal ini dilakukan tahap stemming pada kolom `keywords` pada `smd2`.
13. Train-Test-Split(). Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus dilakukan sebelum membuat model. Mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru. 

## **Modeling**

### Model Development dengan Content Based Filtering

Sebelumnya telah melakukan data preparation, dimana telah mendapatkan dua dataframe baru. Kedua dataframe yang didapatkan akan dibuat model berdasarkan kolom/fitur yang berbeda. Adapun dataframe yang didapatkan beserta kolom/fitur yang akan digunakan:

* `smd`: Movie Overviews dan Taglines.
* `smd2`: Movie Cast, Crew, Keywords dan Genre.

Dari kedua dataframe tersebut selanjutnya akan membuat sistem rekomendasi content based filtering dengan menghitung `cosine similarity` dari setiap data di dataset menggunakan fungsi [cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html). Sebelum itu di definisikan variabel vectorizer sebagai CountVectorizer(), dengan stop_words=’english’ berfungsi untuk menghilangkan kata semacam: i, you, the, a, this, is dan sejenisnya. Tujuannya sebelum menghitung cosine similarity, terlebih dahulu data diubah kedalam bentuk vektor. Kemudian hasil dari perhitungannya disimpan pada dataframe baru `cosine_sim` untuk dataframe `smd` dan `cosine_sim2` untuk dataframe `smd2`. 

Untuk hasil rekomendasi yang diberikan, dibuat fungsi get_recommendations dimana fungsi tersebut akan memberikan rekomendasi terhadap suatu judul film dengan description/overall yang sama dengan judul film yang dimasukkan (inputan).


