# Analisis Proposal vs Sistem — Naive Bayes Tren Penyakit Puskesmas Sarang 1

**Berkas yang dianalisis:** proposal (`.docx`), kode sistem (`.rar`: FastAPI + scikit-learn), dan `Rekap_Bulanan_DUMMY_2023-2025.csv`.
**Batasan:** data hanya CSV di atas, tanpa tambahan data lain. Seluruh saran di bawah hanya memakai kolom CSV tersebut.

---

## 1. Ringkasan

- Sistem **sudah sesuai** dengan inti proposal: label tren Naik/Stabil/Turun, Categorical Naive Bayes + Laplace, split waktu 80/20, pembanding Mayoritas/Decision Tree/KNN, metrik akurasi–presisi–recall–F1 makro–confusion matrix, dan prediksi bulan berikutnya.
- Ada **3 hal yang dijanjikan proposal tetapi belum benar/ada** di sistem: (1) validasi silang deret waktu yang berjalan salah, (2) analisis probabilitas bersyarat antarkelas, (3) pemilihan 10 penyakit terbanyak.
- Ada **beberapa hal di sistem yang tidak tertulis di proposal** (parameter `min_abs_change`, algoritma Persistence, kode_icd10 sebagai atribut, web app + login). Tidak bertabrakan, tetapi harus dicantumkan di proposal agar konsisten.
- **Bagian proposal yang tidak punya dasar data** (register ISPA 110 pasien, Skenario A/B, MTBS, status gizi, desa) tidak bisa dijalankan dengan CSV ini dan perlu direvisi.
- Hasil model pada data dummy: **F1 makro NB 0,4638** (akurasi 0,4952), lebih tinggi dari Decision Tree 0,3980, KNN 0,3886, Persistence 0,2329, Mayoritas 0,1905. Namun uji hanya 105 baris (Jun–Des 2025) dan data dummy, jadi hanya bukti alur berjalan.

---

## 2. Analisis CSV

| Aspek | Temuan |
|---|---|
| Ukuran | 540 baris = 15 diagnosis × 36 bulan (Jan 2023–Des 2025) |
| Kualitas | Tidak ada nilai kosong, tidak ada duplikat, jumlah laki+perempuan dan jumlah 5 kelompok umur selalu sama dengan `jumlah_kasus` |
| Penanda | Semua baris `catatan = DATA DUMMY` |
| Cuaca | `curah_hujan_mm` dan `suhu_rata2` bernilai sama untuk semua penyakit dalam satu bulan (data tingkat bulan, bukan tingkat penyakit) |
| Kunjungan | `total_kunjungan` sama per bulan; total kasus 15 penyakit ≈ 45–55% dari kunjungan |
| Top-10 (total kasus) | J00, I10, K30, M79.1, R50.9, J06.9, J10, K05.2, E11, M79.10 |
| Terendah | A15 (TB), total 412 kasus |
| Kode mirip | M79.1 vs M79.10 (Myalgia), J10 vs J11 (Influenza) — berpotensi tumpang tindih |
| Distribusi label (10%, min 3) | Stabil 199, Naik 172, Turun 154 (525 label; 15 baris bulan pertama tanpa label) |

Catatan: dengan data ini kelas **relatif seimbang** (±38/33/29%). Ini berbeda dari proposal, yang mengasumsikan kelas sangat tidak seimbang (79/9/12%) pada data register ISPA.

Sensitivitas ambang (dihitung dari CSV):

| threshold | min_abs_change | Stabil | Naik | Turun |
|---|---|---|---|---|
| 10% | 0 (rumus proposal murni) | 191 | 176 | 158 |
| 10% | 3 (default sistem) | 199 | 172 | 154 |
| 15% | 3 | 260 | 143 | 122 |
| 20% | 5 | 336 | 110 | 79 |

---

## 3. Analisis Proposal

**Inti yang dijanjikan:** prediksi tren (Naik/Stabil/Turun) kasus penyakit terbanyak, per bulan, di Puskesmas Sarang 1 dengan Naive Bayes; 3 rumusan masalah (tahapan, kinerja, atribut berpengaruh); 4 tujuan khusus; metodologi CRISP-DM dengan 9 tahap (7.5).

**Kekuatan:** rumusan jelas, ada pencegahan kebocoran data, evaluasi pakai F1 makro, ada pembanding.

**Inkonsistensi internal proposal (perlu direvisi):**

1. **Data awal ≠ tujuan.** Bagian 7.2 memakai register ISPA balita (110 pasien, klasifikasi MTBS), sedangkan judul dan 7.5 membahas tren bulanan semua penyakit. Proposal sendiri mengakui ini ("dikonsultasikan kepada dosen pembimbing"). Dengan CSV rekap bulanan, masalah ini selesai, tetapi teks harus diubah.
2. **Skenario A/B** (umur, gizi, desa, gejala, laju napas) tidak dapat dibuat dari CSV karena kolomnya tidak ada.
3. **Tabel 1–3, Gambar 1–3, Lampiran A dan B** menggambarkan data register, bukan CSV.
4. **Stratified k-fold** dan penyebutan kelas tidak seimbang 79/9/12% tidak berlaku pada CSV.
5. **Placeholder belum terisi:** nama dosen pembimbing, periode data lengkap, data resmi 10 penyakit terbanyak, sumber data lengkap.
6. **Data dummy tidak disebut sama sekali.** Sistem sudah memakai banner dummy; proposal perlu menyatakan bahwa CSV dummy hanya untuk pengembangan/uji sistem dan hasil penelitian memakai data asli.
7. **Perangkat (7.4)** menyebut Python + pandas + scikit-learn + matplotlib + Colab/Jupyter + Excel; sistem memakai FastAPI, SQLAlchemy, JWT, Chart.js, Docker.

---

## 4. Analisis Sistem

**Arsitektur:** FastAPI + SQLAlchemy (SQLite), scikit-learn, Jinja2 + Chart.js, JWT dengan peran Admin/Viewer. Layanan: `ingest` → `labeling` → `features` → `training` → `evaluation` → `predict`.

**Alur ML (sesuai kode):**
1. Label: `delta = (kasus_t − kasus_{t−1}) / max(kasus_{t−1}, 1)`; Naik/Turun jika melewati `threshold_pct` **dan** selisih ≥ `min_abs_change`.
2. Atribut hanya dari bulan t−1 dan t−2 (+ kalender) → tidak ada kebocoran. Diskretisasi kuantil 33/67% hanya dari data latih.
3. Split waktu: data valid 35 periode → 28 latih (2023-02..2025-05), 7 uji (2025-06..2025-12).
4. Tuning alpha ∈ {0,1; 0,5; 1; 2} dengan `TimeSeriesSplit`.
5. Latih NB + 4 pembanding, hitung metrik, drop-one-feature, simpan model `.joblib`, prediksi bulan berikutnya (Jan 2026).

**Pengujian:** `pytest` 9 dari 9 lulus (saya jalankan ulang). Tes kebocoran dan split kronologis ada.

---

## 5. Kesesuaian Sistem terhadap Proposal

### 5.1 Yang sudah sesuai

| Proposal | Sistem |
|---|---|
| Kelas Naik/Stabil/Turun, ambang 10% | `threshold_pct` default 10% |
| Atribut lag 1–2 bulan sebelumnya | `label_lag1`, `label_lag2` |
| Atribut bulan, musim, curah hujan, proporsi umur, proporsi jenis kelamin | Tersedia semua (musim Nov–Apr hujan, Mei–Okt kemarau) |
| Split waktu 80/20 | `test_ratio` 0,20, tanpa pengacakan |
| Categorical NB + Laplace + tuning alpha | `CategoricalNB`, alpha {0,1; 0,5; 1; 2} |
| Pembanding: mayoritas, Decision Tree, KNN | Ada, plus Persistence |
| Confusion matrix, akurasi, presisi, recall, F1 makro | Ada, per kelas dan ekspor CSV |
| Retrain tanpa atribut tertentu | Drop-one-feature |
| Prediksi periode berikutnya | Ada, 3 probabilitas berjumlah 1,0 |
| Anonimisasi/privasi data | Kolom identitas ditolak (422) |
| Demo program pada Seminar Hasil | Web app lengkap |

### 5.2 Yang TIDAK sesuai / belum ada (proposal menjanjikan, sistem belum)

| # | Proposal | Kondisi sistem | Tingkat |
|---|---|---|---|
| 1 | "Validasi silang deret waktu (time series split)" | **Salah urut.** Baris latih terurut per penyakit, lalu waktu, sehingga `TimeSeriesSplit` membagi per blok penyakit, bukan per waktu. Saya cek: setiap fold validasinya mencakup 2023-01 s.d. 2025-12 dan berisi penyakit berbeda-beda (fold 1: E11, I10, J00; fold 5: R50.9, R51...). Alpha terpilih jadi tidak valid secara temporal. Skor CV juga tidak disimpan/ditampilkan. | **Tinggi** |
| 2 | Atribut berpengaruh dianalisis lewat "perbandingan probabilitas bersyarat antarkelas" (7.6, rumusan masalah 3) | Hanya drop-one-feature. Tabel P(kategori ∣ kelas) belum ada di API maupun halaman Hasil. | **Tinggi** |
| 3 | "Menetapkan sepuluh penyakit dengan kasus terbanyak" (7.5) | Model memakai **semua 15** diagnosis. Top-10 hanya di grafik dashboard. | Sedang |
| 4 | Atribut "total kunjungan" | Hanya `rasio_kasus_kunjungan_lag1` (kasus ÷ kunjungan); `total_kunjungan` mentah tidak menjadi atribut. | Rendah |
| 5 | Musim (hujan/kemarau) sebagai atribut | Tersedia, tetapi **tidak dipilih secara default** (`musim` dikeluarkan dari default). | Rendah |
| 6 | Preprocessing: pembersihan, penyeragaman nama/kode penyakit | Sistem hanya memvalidasi dan menolak; tidak ada laporan preprocessing atau penyeragaman. | Rendah |
| 7 | Gaussian NB untuk atribut numerik (6.3) | Tidak ada; semua numerik didiskretisasi ke Categorical NB. Ini masih diperbolehkan oleh 7.5 ("diskretisasi apabila diperlukan"), tinggal ditulis jelas. | Rendah |

### 5.3 Yang ada di sistem tetapi tidak tertulis di proposal (tidak bertabrakan, perlu dicantumkan)

| Fitur sistem | Catatan |
|---|---|
| `min_abs_change` (default 3) | Proposal hanya mengenal persentase. Setting `min_abs_change = 0` menghasilkan rumus proposal murni. |
| Baseline **Persistence** | Tambahan; proposal hanya menyebut mayoritas + 1–2 algoritma lain. |
| `kode_icd10` sebagai atribut (satu model gabungan) | Tidak disebut di 7.5; berperan penting (delta F1 +0,034 pada run 1). |
| `suhu_lag1` | Proposal hanya menyebut curah hujan sebagai atribut (suhu disebut di 7.3 sebagai data pendukung). |
| Web app, login JWT/RBAC, SQLite, Docker, Chart.js | Tidak ada di 7.4 (Perangkat). |
| Banner data dummy, pratinjau label, riwayat run, hapus dataset | Pendukung operasional. |

### 5.4 Temuan kode lain (bukan soal proposal, tetapi memengaruhi hasil)

1. Kategori tak dikenal pada data uji/prediksi dipaksa ke indeks 0 (`X == 999 → 0`), padahal sudah disediakan satu kategori cadangan lewat `min_categories`. Ini bisa menggeser probabilitas diam-diam.
2. Saat prediksi, `label_lag1` kosong diganti `"Stabil"`, sedangkan saat latih diganti `"tidak_ada"` (tidak konsisten).
3. Uji hanya Jun–Des 2025, sehingga puncak musim hujan (Jan–Mar) tidak pernah diuji.
4. Drop-one tidak stabil: run 1 dan run 4 memberi urutan atribut berbeda, banyak delta negatif (mis. `suhu_lag1` −0,1136 pada run 4).
5. Deteksi PII memakai pencocokan substring: kolom seperti `nama_klinik` atau `suhu_normal` ikut ditolak.
6. `SECRET_KEY` default ditulis di `config.py`, `.env` ikut dalam arsip, kata sandi default ada di README. Aman untuk lokal, tidak untuk produksi.

---

## 6. Saran Pengembangan (tidak bertabrakan dengan proposal, hanya CSV)

### Prioritas 1 — menutup kekurangan terhadap proposal

1. **Perbaiki validasi silang waktu.** Urutkan latih per (tahun, bulan), lalu buat fold berbasis periode (expanding window: latih s.d. bulan k, validasi bulan k+1..k+m). Simpan dan tampilkan skor F1 makro per fold dan rata-rata ± simpangan baku.
2. **Tabel probabilitas bersyarat P(kategori ∣ kelas).** Ambil dari `nb_model.feature_log_prob_` dan `category_count_`, tampilkan per atribut di halaman Hasil, plus ekspor CSV. Ini menjawab rumusan masalah 3 dan bagian 7.6.
3. **Opsi Top-N penyakit (default 10).** Hitung peringkat dari **periode latih saja** (agar tidak bocor), dengan pilihan "Semua penyakit" tetap tersedia.
4. **Preset "Sesuai proposal".** Satu klik mengisi: ambang 10%, `min_abs_change` = 0, atribut bulan, musim, curah hujan, proporsi umur, proporsi laki-laki, kunjungan, label lag 1–2. Hasil preset ini dibandingkan dengan setelan default untuk menunjukkan efek `min_abs_change`.
5. **Tambah atribut `total_kunjungan` lag1** (didiskretisasi 3 kategori) dan beri catatan: `bulan_target` dan `musim` redundan, pilih salah satu per percobaan.
6. **Perbaiki dua inkonsistensi kode** (kategori tak dikenal, fallback `label_lag1`).

### Prioritas 2 — memperkuat hasil untuk pembahasan/artikel

7. **Evaluasi walk-forward berulang** (beberapa titik awal uji, mencakup Jan–Mar) dan laporkan rata-rata ± simpangan baku F1 makro. Menjawab keterbatasan uji 7 bulan.
8. **Tabel sensitivitas ambang** (5/10/15/20%) otomatis: distribusi kelas dan F1 NB pada tiap ambang. Mendukung kalimat proposal bahwa ambang "ditetapkan bersama dosen pembimbing dan petugas Puskesmas".
9. **Ablasi per kelompok atribut** (kalender, lag label, demografi, kunjungan, cuaca) di samping drop-one. Lebih stabil daripada satu atribut per satu.
10. **Kinerja per penyakit** (F1 per kode ICD-10) dan per kelas, untuk melihat penyakit mana yang sulit diprediksi.
11. **Halaman "Contoh Perhitungan Naive Bayes"** untuk satu baris prediksi: prior, likelihood tiap atribut, skor, posterior. Ini setara Lampiran B proposal tetapi memakai CSV, dan memenuhi klaim "hasil mudah ditelusuri".
12. **Penjelasan per prediksi:** tampilkan 3 atribut dengan kontribusi log-probabilitas terbesar pada tiap penyakit.
13. **Laporan preprocessing:** jumlah baris, nilai kosong, duplikat, baris terbuang akibat lag, distribusi label. Memenuhi tahap preprocessing 7.5 secara terdokumentasi.

### Prioritas 3 — kerapian

14. Dashboard: filter periode, tabel dapat diurutkan (sudah ada di PRD, belum ada di UI).
15. Ekspor "tabel hasil siap artikel" (metrik 5 algoritma + confusion matrix NB) untuk format AIMRaD.
16. Perketat deteksi PII (cocokkan nama kolom utuh/awalan), pindahkan `SECRET_KEY` dan kata sandi ke `.env` yang tidak disertakan, ganti kata sandi default.

### Sebaiknya dihindari (bertabrakan dengan proposal atau data)

- Klasifikasi pasien individual, Skenario A/B, MTBS (tidak ada datanya dan sudah masuk non-tujuan PRD).
- Mengganti ke regresi/peramalan angka (ARIMA, LSTM) sebagai model utama; proposal mendefinisikan tugasnya sebagai klasifikasi tren.
- Mengubah kelas target atau menambah kelas.
- Pembagian data acak.
- Menggabungkan kode ICD (mis. M79.1 + M79.10) pada dataset utama; jika ingin, jadikan analisis tambahan terpisah, bukan pengganti.
- Menambah data eksternal (BMKG tambahan, dsb.), karena data dibatasi pada CSV.

---

## 7. Revisi Teks Proposal yang Disarankan

1. **7.2 Sampel:** ganti dengan "data rekap bulanan 15 diagnosis ICD-10, 36 bulan (Jan 2023–Des 2025), 540 baris", dan nyatakan sebagai data dummy untuk pengembangan sistem; data asli menyusul dengan format kolom yang sama.
2. **Hapus atau pindahkan** Tabel 1–3, Gambar 1–3, Skenario A/B, MTBS ke "studi pendahuluan" (atau hapus) dan ganti dengan tabel profil CSV (bagian 2 dokumen ini).
3. **7.4 Perangkat:** tambahkan FastAPI, SQLAlchemy, Chart.js; sebutkan matplotlib/Colab hanya jika memang dipakai.
4. **7.5:** tulis `min_abs_change`, Persistence sebagai baseline tambahan, `kode_icd10` dan `suhu` sebagai atribut, serta bahwa numerik didiskretisasi (tanpa Gaussian NB).
5. **7.5 Pembagian data:** pastikan kalimat time series split sesuai implementasi setelah perbaikan saran #1.
6. **Lampiran A (kamus data) dan B (contoh hitung):** ganti dengan kamus data 16 kolom CSV dan contoh hitung dari halaman saran #11.
7. Isi semua placeholder `[...]` dan sebutkan batas data: dummy tidak boleh dipakai sebagai kesimpulan kesehatan.