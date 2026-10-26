# PRD: Sistem Prediksi Tren Kasus Penyakit dengan Naive Bayes
**Proyek:** Penerapan Algoritma Naive Bayes untuk Prediksi Tren Kasus Penyakit Terbanyak di Puskesmas Sarang 1
**Versi:** 1.0 | **Bahasa antarmuka:** Indonesia | **Bahasa kode/komentar:** Inggris atau Indonesia (konsisten)

---

## 0. Instruksi untuk AI Developer (baca dulu)

Kamu bertindak sebagai senior full-stack developer. Bangun sistem sesuai PRD ini secara lengkap.

1. Berikan **kode lengkap per file** sesuai struktur di Bagian 11, urut dari konfigurasi sampai frontend. Jangan ada placeholder, `TODO`, atau potongan "..." pada kode.
2. Sertakan `requirements.txt`, `.env.example`, `README.md` (cara instal dan jalankan), skrip `seed`, dan tes `pytest`.
3. Jika ada keputusan yang tidak tertulis di PRD, pilih opsi paling sederhana, lalu tulis keputusan itu di bagian "Asumsi" pada README.
4. Jangan menambah fitur di luar PRD. Jangan mengganti tech stack tanpa alasan tertulis.
5. Setelah selesai, jalankan mental-check terhadap **Kriteria Penerimaan (Bagian 13)** dan laporkan item mana yang terpenuhi.

---

## 1. Ringkasan Produk

Aplikasi web untuk memprediksi **tren jumlah kasus per penyakit pada bulan berikutnya** (kelas **Naik / Stabil / Turun**) memakai **Naive Bayes**, dari data rekap bulanan kunjungan Puskesmas per diagnosis ICD-10. Hasil dipakai petugas dan peneliti sebagai pendukung perencanaan obat, bahan medis habis pakai, dan penyuluhan.

**Pengguna:** (1) Admin/peneliti: mengunggah data, melatih model, mengevaluasi. (2) Petugas Puskesmas (viewer): melihat dashboard dan prediksi.

## 2. Tujuan dan Non-Tujuan

**Tujuan**
- G1. Mengunggah dan memvalidasi data rekap bulanan.
- G2. Membentuk label tren secara otomatis dengan ambang yang bisa diatur.
- G3. Melatih dan mengevaluasi Naive Bayes, dibandingkan dengan baseline dan algoritma pembanding.
- G4. Menampilkan prediksi tren bulan berikutnya per penyakit beserta probabilitasnya.
- G5. Menjelaskan atribut apa yang paling berpengaruh.

**Non-tujuan (jangan dibuat)**
- Klasifikasi pasien individual, input data pasien, atau penyimpanan data identitas.
- Prediksi angka jumlah kasus (regresi). Yang diprediksi hanya kelas tren.
- Integrasi langsung ke SIMPUS, aplikasi mobile, notifikasi email/WhatsApp.

## 3. Konteks dan Batasan Data (penting)

- Data asli Puskesmas Sarang 1 belum lengkap. Sistem **dikembangkan dengan data dummy** (`Rekap_Bulanan_DUMMY_2023-2025.csv`), lalu diganti data asli dengan format kolom yang sama.
- Baris dengan `catatan = "DATA DUMMY"` harus memunculkan **banner peringatan permanen** di semua halaman hasil: *"Data ini dummy. Hasil hanya untuk uji sistem, bukan kesimpulan kesehatan."*
- Data asli yang diharapkan: minimal **36 bulan** (Jan 2023-Des 2025). Jika kurang dari 24 bulan, sistem menampilkan peringatan "data terlalu pendek untuk evaluasi yang andal" tetapi tetap boleh melatih.
- **Dilarang menyimpan data identitas.** Unggahan yang memuat kolom bernama `nama`, `nik`, `alamat`, `no_rm`, `orang_tua` (atau variasinya) harus **ditolak** dengan pesan jelas.

## 4. Spesifikasi Data Masukan

File: CSV (UTF-8) atau XLSX, satu baris = satu penyakit pada satu bulan.

| Kolom | Wajib | Tipe | Aturan |
|---|---|---|---|
| `tahun` | Ya | int | 2015-2100 |
| `bulan` | Ya | int | 1-12 |
| `kode_icd10` | Ya | string | Huruf besar, contoh `J06.9` |
| `nama_penyakit` | Ya | string | |
| `jumlah_kasus` | Ya | int | >= 0 |
| `kasus_laki`, `kasus_perempuan` | Tidak | int | Jika ada, jumlahnya harus = `jumlah_kasus` |
| `kasus_<1th`, `kasus_1-5th`, `kasus_6-14th`, `kasus_15-59th`, `kasus_60th+` | Tidak | int | Jika ada, jumlahnya harus = `jumlah_kasus` |
| `total_kunjungan` | Tidak | int | >= total kasus semua penyakit pada bulan itu |
| `curah_hujan_mm` | Tidak | float | >= 0 |
| `suhu_rata2` | Tidak | float | 15-40 |
| `catatan` | Tidak | string | `DATA DUMMY` memicu banner |

Nama kolom di DB dinormalisasi (`kasus_lt1th`, `kasus_1_5th`, `kasus_6_14th`, `kasus_15_59th`, `kasus_gte60th`).

**Aturan validasi:** kolom wajib lengkap; tidak ada duplikat (`tahun`, `bulan`, `kode_icd10`); tiap penyakit punya bulan berurutan tanpa celah (celah dilaporkan sebagai error, bukan diisi otomatis); nilai negatif ditolak. Laporan validasi mencantumkan nomor baris dan jenis kesalahan, dalam format JSON dan tampilan tabel.

## 5. Spesifikasi Machine Learning

### 5.1 Pembentukan label
Per penyakit, urut waktu. Untuk bulan *t*:
`delta = (kasus_t - kasus_{t-1}) / max(kasus_{t-1}, 1)`
- **Naik** jika `delta > +threshold_pct` **dan** `(kasus_t - kasus_{t-1}) >= min_abs_change`
- **Turun** jika `delta < -threshold_pct` **dan** `(kasus_{t-1} - kasus_t) >= min_abs_change`
- **Stabil** selain itu.

Parameter default: `threshold_pct = 10%`, `min_abs_change = 3` (agar fluktuasi kecil pada penyakit berkasus sedikit, seperti TB, tidak dihitung sebagai tren). Keduanya bisa diubah di UI.

### 5.2 Tugas prediksi dan pencegahan kebocoran
Memprediksi **label bulan *t*** hanya dari informasi yang tersedia sampai **bulan *t-1***. Atribut yang boleh dipakai:

| Atribut | Sumber | Catatan |
|---|---|---|
| `kode_icd10` | kategorikal | Satu model gabungan untuk semua penyakit |
| `bulan_target` (1-12) atau `musim` (hujan: Nov-Apr, kemarau: Mei-Okt) | kalender | Diketahui sebelumnya |
| `label_lag1`, `label_lag2` | label bulan t-1 dan t-2 | |
| `proporsi_laki_lag1` | kasus_laki / jumlah_kasus pada t-1 | Jika kolom ada |
| `proporsi_umur_*_lag1` | per kelompok umur pada t-1 | Jika kolom ada |
| `rasio_kasus_kunjungan_lag1` | kasus / total_kunjungan pada t-1 | Jika kolom ada |
| `curah_hujan_lag1`, `suhu_lag1` | nilai bulan t-1 | **Bukan** nilai bulan t (tidak diketahui saat prediksi) |

Dilarang memakai `jumlah_kasus` bulan *t* atau nilai apa pun dari bulan *t* sebagai atribut. Sediakan **tes otomatis** yang membuktikan tidak ada kebocoran.

Atribut numerik didiskretisasi menjadi 3 kategori (rendah/sedang/tinggi) memakai kuantil yang **dihitung hanya dari data latih**, lalu dipakai `CategoricalNB`. Missing value pada atribut opsional: kategori `tidak_ada`.

### 5.3 Pembagian data dan pelatihan
- **Split berdasarkan waktu**: `test_ratio` default 20% bulan terakhir sebagai data uji. Dilarang mengacak.
- Validasi tambahan: `TimeSeriesSplit` (default 5 fold) pada data latih.
- Model utama: `CategoricalNB` (scikit-learn), `alpha` (Laplace) default 1.0, dapat dituning pada rentang {0.1, 0.5, 1, 2} lewat TimeSeriesSplit.
- `random_state = 42` di semua tempat yang relevan.

### 5.4 Pembanding (wajib ditampilkan berdampingan)
1. **Mayoritas**: selalu menebak kelas terbanyak di data latih.
2. **Persistence**: menebak label bulan t-1 untuk bulan t.
3. `DecisionTreeClassifier` (max_depth=4).
4. `KNeighborsClassifier` (k=5).

### 5.5 Evaluasi
Per model: accuracy, precision, recall, F1 per kelas, **F1 macro**, dan confusion matrix. Laporkan juga jumlah data latih/uji dan distribusi kelas. Kesimpulan otomatis di UI: jika F1 macro Naive Bayes tidak lebih tinggi dari baseline terbaik, tampilkan teks *"Model belum lebih baik dari baseline"*.

### 5.6 Pengaruh atribut
- Tabel probabilitas bersyarat `P(kategori | kelas)` untuk tiap atribut.
- **Drop-one-feature**: latih ulang tanpa satu atribut, laporkan selisih F1 macro (urut terbesar).

### 5.7 Prediksi
Untuk bulan setelah data terakhir, hasilkan per penyakit: kelas prediksi (Naik/Stabil/Turun) dan probabilitas ketiga kelas. Simpan bersama `model_run_id`.

## 6. Kebutuhan Fungsional

| ID | Kebutuhan | Prioritas |
|---|---|---|
| FR-01 | Login admin (JWT); viewer hanya baca | Must |
| FR-02 | Unggah CSV/XLSX, validasi, tampilkan laporan, simpan dataset | Must |
| FR-03 | Tolak unggahan yang berisi kolom identitas | Must |
| FR-04 | Daftar dataset dan detail (jumlah baris, rentang bulan, penyakit, flag dummy) | Must |
| FR-05 | Pratinjau label (distribusi Naik/Stabil/Turun) untuk threshold tertentu sebelum melatih | Must |
| FR-06 | Latih model dengan parameter (threshold, min_abs_change, alpha, test_ratio, pilihan atribut) | Must |
| FR-07 | Halaman hasil: metrik, confusion matrix, perbandingan model, pengaruh atribut | Must |
| FR-08 | Jalankan prediksi bulan berikutnya, tampilkan tabel dan grafik | Must |
| FR-09 | Dashboard: grafik jumlah kasus bulanan per penyakit, top-N penyakit, filter periode | Must |
| FR-10 | Ekspor prediksi dan metrik ke CSV | Should |
| FR-11 | Riwayat model run (bisa dibandingkan) | Should |
| FR-12 | Banner data dummy | Must |
| FR-13 | Hapus dataset dan model run oleh admin | Could |

## 7. Arsitektur dan Tech Stack

- **Backend:** Python 3.11+, **FastAPI**, SQLAlchemy 2.x, Pydantic v2, Alembic (migrasi).
- **ML:** pandas, NumPy, scikit-learn, joblib (simpan model).
- **Database:** SQLite untuk pengembangan; harus bisa pindah ke MySQL/PostgreSQL hanya dengan mengganti `DATABASE_URL`.
- **Frontend:** Jinja2 + HTML/CSS + **Chart.js** (CDN), disajikan oleh FastAPI. Tidak memakai framework SPA.
- **Auth:** JWT (python-jose), password di-hash (passlib/bcrypt). Akun admin awal dari variabel lingkungan.
- **Pengujian:** pytest + httpx TestClient.
- **Deploy:** `uvicorn`; sediakan `Dockerfile` dan `docker-compose.yml` sederhana.

## 8. Skema Database

```sql
users(id PK, username UNIQUE, password_hash, role ENUM('admin','viewer'), created_at)

datasets(id PK, name, original_filename, is_dummy BOOL, row_count INT,
         start_period CHAR(7), end_period CHAR(7), disease_count INT,
         uploaded_by FK users.id, uploaded_at)

diseases(id PK, icd10 UNIQUE, name)

monthly_cases(id PK, dataset_id FK, disease_id FK, year INT, month INT,
  cases INT, male INT NULL, female INT NULL,
  age_lt1 INT NULL, age_1_5 INT NULL, age_6_14 INT NULL, age_15_59 INT NULL, age_gte60 INT NULL,
  total_visits INT NULL, rainfall_mm FLOAT NULL, temp_avg FLOAT NULL,
  UNIQUE(dataset_id, disease_id, year, month))

model_runs(id PK, dataset_id FK, created_by FK, created_at,
  threshold_pct FLOAT, min_abs_change INT, alpha FLOAT, test_ratio FLOAT,
  features_json TEXT, train_rows INT, test_rows INT, train_period TEXT, test_period TEXT,
  class_distribution_json TEXT, model_path TEXT, status ENUM('running','done','failed'), error TEXT NULL)

model_metrics(id PK, model_run_id FK, algorithm ENUM('naive_bayes','majority','persistence','decision_tree','knn'),
  accuracy FLOAT, f1_macro FLOAT, precision_json TEXT, recall_json TEXT, f1_json TEXT, confusion_matrix_json TEXT)

feature_influence(id PK, model_run_id FK, feature TEXT, f1_macro_without FLOAT, delta_f1 FLOAT)

predictions(id PK, model_run_id FK, disease_id FK, target_year INT, target_month INT,
  predicted_class ENUM('Naik','Stabil','Turun'), p_naik FLOAT, p_stabil FLOAT, p_turun FLOAT, created_at)
```

Index: `monthly_cases(dataset_id, disease_id, year, month)`, `predictions(model_run_id)`.

## 9. Spesifikasi API (prefix `/api`)

| Method | Endpoint | Fungsi | Role |
|---|---|---|---|
| POST | `/auth/login` | Kembalikan JWT | publik |
| GET | `/health` | Status layanan | publik |
| POST | `/datasets/upload` | Unggah file, validasi, simpan. Respons: `dataset_id` + laporan validasi (422 bila gagal) | admin |
| GET | `/datasets` | Daftar dataset | semua |
| GET | `/datasets/{id}` | Detail dataset | semua |
| DELETE | `/datasets/{id}` | Hapus dataset beserta turunannya | admin |
| GET | `/diseases` | Daftar penyakit pada dataset (`?dataset_id=`) | semua |
| GET | `/monthly-cases` | Filter `dataset_id, icd10, from, to` | semua |
| GET | `/dashboard/summary` | Top-N penyakit, total kasus per bulan | semua |
| POST | `/labels/preview` | Body: `dataset_id, threshold_pct, min_abs_change`. Respons: distribusi label per penyakit dan total | admin |
| POST | `/models/train` | Latih (sinkron, selesai < 30 detik pada 540 baris). Respons: `model_run_id` | admin |
| GET | `/models` | Riwayat model run | semua |
| GET | `/models/{id}` | Parameter dan ringkasan | semua |
| GET | `/models/{id}/metrics` | Metrik semua algoritma | semua |
| GET | `/models/{id}/confusion-matrix` | Matriks tiap algoritma | semua |
| GET | `/models/{id}/feature-influence` | Pengaruh atribut dan tabel probabilitas bersyarat | semua |
| POST | `/predictions/run` | Body: `model_run_id`. Hasilkan prediksi bulan berikutnya | admin |
| GET | `/predictions?model_run_id=` | Ambil prediksi | semua |
| GET | `/export/predictions.csv?model_run_id=` | Ekspor CSV | semua |
| GET | `/export/metrics.csv?model_run_id=` | Ekspor CSV | semua |

**Format error seragam:** `{"error": {"code": "VALIDATION_ERROR", "message": "...", "details": [...]}}` dengan status HTTP yang tepat (400/401/403/404/422).

## 10. Antarmuka Pengguna (halaman)

1. **Login**
2. **Dashboard**: banner dummy (jika ada), kartu ringkas (periode data, jumlah penyakit), grafik garis kasus bulanan (pilih penyakit), bar top-10 penyakit.
3. **Unggah Data**: form unggah, tabel laporan validasi, tautan unduh template CSV.
4. **Dataset**: daftar dan detail.
5. **Latih Model**: form parameter, tombol "Pratinjau Label" (grafik distribusi), tombol "Latih".
6. **Hasil Model**: tabel perbandingan 5 algoritma, bar F1 macro, confusion matrix (heatmap tabel), pengaruh atribut, teks kesimpulan otomatis.
7. **Prediksi**: tabel penyakit | prediksi | probabilitas (3 kelas, bar horizontal), kode warna (Naik=merah, Stabil=abu, Turun=hijau), tombol ekspor CSV.

Kebutuhan UI: responsif (laptop dan ponsel), bahasa Indonesia, tabel dapat diurutkan, loading indicator saat melatih.

## 11. Struktur Proyek

```
prediksi-tren-puskesmas/
├── app/
│   ├── main.py                # FastAPI app, router, static, templates
│   ├── config.py              # Settings dari .env
│   ├── database.py            # Engine, session
│   ├── models.py              # SQLAlchemy models
│   ├── schemas.py             # Pydantic schemas
│   ├── security.py            # JWT, hashing, dependency role
│   ├── routers/               # auth, datasets, cases, models, predictions, export, pages
│   ├── services/
│   │   ├── ingest.py          # baca CSV/XLSX, normalisasi kolom, validasi, tolak PII
│   │   ├── labeling.py        # pembentukan label tren
│   │   ├── features.py        # atribut lag, diskretisasi (fit hanya di data latih)
│   │   ├── training.py        # NB + pembanding, TimeSeriesSplit, tuning alpha
│   │   ├── evaluation.py      # metrik, confusion matrix, drop-one-feature
│   │   └── predict.py         # prediksi bulan berikutnya
│   ├── templates/             # base.html, login, dashboard, upload, datasets, train, results, predictions
│   └── static/                # css, js (Chart.js helper)
├── migrations/                # Alembic
├── data/
│   └── Rekap_Bulanan_DUMMY_2023-2025.csv
├── scripts/
│   └── seed.py                # buat admin awal + muat data dummy sebagai dataset
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## 12. Kebutuhan Non-Fungsional

- **Privasi:** tidak ada kolom identitas pasien di DB mana pun; penolakan PII pada unggahan; log tidak mencetak isi data.
- **Keamanan:** password di-hash, JWT kedaluwarsa (default 8 jam), validasi tipe dan ukuran file (maks 5 MB), pencegahan SQL injection lewat ORM, secret hanya dari `.env`.
- **Reprodusibilitas:** hasil pelatihan identik untuk data dan parameter yang sama (`random_state=42`); parameter tiap run tersimpan.
- **Kinerja:** pelatihan penuh pada 540 baris < 30 detik; halaman dashboard < 2 detik.
- **Kualitas kode:** type hints, docstring singkat, pemisahan router dan service, tanpa logika ML di router.
- **Observabilitas:** logging terstruktur untuk unggah, latih, dan error.

## 13. Kriteria Penerimaan

1. `docker-compose up` (atau `uvicorn`) berjalan; `scripts/seed.py` memuat data dummy dan membuat admin.
2. Unggahan CSV dummy sukses, dan banner dummy tampil di dashboard, hasil, dan prediksi.
3. Unggahan file berkolom `nama` atau `nik` ditolak dengan 422 dan pesan jelas.
4. Unggahan dengan duplikat, bulan bolong, atau jumlah umur tidak sama dengan jumlah kasus menghasilkan laporan kesalahan per baris.
5. Pratinjau label berubah mengikuti `threshold_pct` dan `min_abs_change`.
6. Pelatihan menghasilkan metrik untuk 5 algoritma (NB, mayoritas, persistence, Decision Tree, KNN) dan confusion matrix.
7. Tes otomatis membuktikan tidak ada kebocoran: mengubah nilai `jumlah_kasus` bulan target tidak mengubah atribut yang dipakai memprediksi bulan itu.
8. Split data berurutan waktu: semua bulan uji lebih baru daripada semua bulan latih.
9. Halaman hasil menampilkan teks "Model belum lebih baik dari baseline" bila F1 macro NB <= baseline terbaik.
10. Prediksi bulan berikutnya menampilkan 15 penyakit dengan probabilitas yang berjumlah 1,0 per baris.
11. Viewer tidak bisa mengunggah, melatih, atau menghapus (403).
12. `pytest` lulus, mencakup: labeling, kebocoran, validasi unggahan, deteksi PII, auth/role, endpoint train dan predict.

## 14. Data Uji

Gunakan file `Rekap_Bulanan_DUMMY_2023-2025.csv` (540 baris: 15 diagnosis x 36 bulan). Pola yang sengaja ada: penyakit pernapasan naik di musim hujan (puncak Januari), diare naik di musim kemarau, tren naik sekitar 4% per tahun, dan noise acak. Seluruh baris bertanda `catatan = DATA DUMMY`. Hasil model pada data ini hanya untuk menguji alur sistem.

## 15. Di Luar Lingkup Rilis Ini (catatan pengembangan lanjut)

Klasifikasi derajat ISPA balita dari register pasien, integrasi SIMPUS, notifikasi, peramalan angka dengan model deret waktu.
