# Sistem Prediksi Tren Kasus Penyakit Terbanyak dengan Naive Bayes
**Puskesmas Sarang 1 Kabupaten Rembang**

Aplikasi web berbasis machine learning untuk memprediksi **tren jumlah kasus per penyakit pada bulan berikutnya** (kelas **Naik / Stabil / Turun**) menggunakan algoritma **Categorical Naive Bayes** dan dibandingkan berdampingan dengan 4 algoritma baseline/pembanding (**Mayoritas**, **Persistence**, **Decision Tree**, dan **K-Nearest Neighbors**), dari data rekap bulanan kunjungan per diagnosis ICD-10.

---

## 1. Arsitektur & Teknologi

Sistem dibangun mengacu pada **PRD** dan diselaraskan dengan tata kelola **[rule.md](file:///c:/laragon/www/naive_bayes_puskesmas/rule.md)**:
- **Backend Core**: Python 3.11+ / 3.13, **FastAPI**, **SQLAlchemy 2.x**, **Pydantic v2**.
- **Machine Learning**: `scikit-learn` (`CategoricalNB`, `DecisionTreeClassifier`, `KNeighborsClassifier`, `TimeSeriesSplit`), `pandas`, `NumPy`, `joblib`.
- **Database**: SQLite (default lokal) dengan dukungan migrasi mudah ke MySQL/PostgreSQL via variabel `DATABASE_URL`.
- **Frontend & Presentasi**: HTML5 semantik, CSS Modular (Clean White Canvas, Soft Blue Button Interactions, Slim Custom Scrollbar sesuai standar `rule.md`), **Font Awesome 6**, dan visualisasi interaktif **Chart.js** (CDN). Disajikan langsung via Jinja2 tanpa framework SPA berat.
- **Autentikasi & Keamanan**: JWT (JSON Web Token), hashing password menggunakan `bcrypt`, kontrol akses berbasis peran (RBAC: Admin & Viewer), penolakan kolom PII (nama, NIK, dsb), dan banner permanen data dummy.
- **Pengujian**: `pytest` + `httpx TestClient` (100% lulus seluruh skenario pengujian).

---

## 2. Struktur Proyek

```
naive_bayes_puskesmas/
├── app/
│   ├── main.py                # Inisialisasi FastAPI, middleware, static, router
│   ├── config.py              # Konfigurasi Pydantic Settings dari .env
│   ├── database.py            # Engine SQLAlchemy & SessionLocal
│   ├── models.py              # Skema ORM database (User, Dataset, Case, ModelRun, Prediction)
│   ├── schemas.py             # Pydantic validation schemas
│   ├── security.py            # Hashing Bcrypt, JWT token, dependencies RBAC
│   ├── routers/               # auth, datasets, cases, models, predictions, export, pages
│   ├── services/
│   │   ├── ingest.py          # Parsing CSV/XLSX, validasi kontinuitas, penolakan PII
│   │   ├── labeling.py        # Rumus pembentukan label tren (Naik/Stabil/Turun)
│   │   ├── features.py        # Ekstraksi lag fitur, pencegahan kebocoran, diskretisasi
│   │   ├── training.py        # Latih CategoricalNB + 4 pembanding, TimeSeriesSplit, tuning
│   │   ├── evaluation.py      # Metrik akurasi, F1 macro, confusion matrix, drop-one-feature
│   │   └── predict.py         # Inferensi prediksi bulan berikutnya (probabilitas = 1.0)
│   ├── templates/             # Jinja2: base, login, dashboard, upload, datasets, train, results, predictions
│   └── static/                # Modular CSS (base, layout, scrollbar, components) & JS
├── data/                      # Direktori data rekap kasus bulanan
│   └── Rekap_Bulanan_DUMMY_2023-2025.csv
├── dataset/                   # Dataset sumber dari pengguna
│   └── Rekap_Bulanan_DUMMY_2023-2025.csv
├── scripts/
│   └── seed.py                # Pembuat user awal, pemuat dataset dummy, dan inisialisasi model
├── tests/                     # Pengujian otomatis pytest
│   ├── conftest.py
│   ├── test_labeling.py       # Uji rumus label tren
│   ├── test_leakage.py        # Uji pencegahan kebocoran & split waktu kronologis
│   ├── test_ingest.py         # Uji penolakan PII & validasi data
│   ├── test_auth_roles.py     # Uji otorisasi RBAC (Viewer vs Admin)
│   └── test_api_flow.py       # Uji alur pelatihan & inferensi prediksi
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .env
├── rule.md                    # Pedoman arsitektur, kode, keamanan & layout
└── PRD_Sistem_Prediksi_Tren_Penyakit_Puskesmas_Sarang1.md
```

---

## 3. Akun Pengguna Awal (Default Credentials)

| Peran (Role) | Nama Pengguna | Kata Sandi | Hak Akses |
|---|---|---|---|
| **Admin** | `admin` | `admin_puskesmas_sarang1` | Akses penuh: Unggah dataset, latih model, eksekusi prediksi, hapus data |
| **Viewer** | `viewer` | `viewer_puskesmas_sarang1` | Akses baca: Dashboard analitik, metrik evaluasi model, hasil prediksi |

---

## 4. Cara Instalasi & Menjalankan

### Opsi A: Menjalankan di Lingkungan Lokal (Windows / Laragon)

1. **Pastikan Python Aktif**:
   Aplikasi menggunakan lingkungan virtual `.venv` yang telah dibuat.
2. **Jalankan Seeder (Pengisian Data Awal & Model Run)**:
   ```powershell
   & ".\.venv\Scripts\python.exe" scripts/seed.py
   ```
3. **Jalankan Server Web Uvicorn**:
   ```powershell
   & ".\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```
4. **Buka Peramban (Browser)**:
   Akses `http://127.0.0.1:8000/` dan masuk menggunakan akun admin atau viewer.

### Opsi B: Menjalankan Menggunakan Docker

```bash
docker-compose up --build -d
```
Akses di `http://localhost:8000/`.

---

## 5. Menjalankan Pengujian Otomatis (`pytest`)

Jalankan seluruh rangkaian uji kepatuhan:
```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests -v
```

---

## 6. Asumsi & Keputusan Teknis

1. **Pencegahan Kebocoran Data (Data Leakage)**:
   - Data bulan target *t* hanya menyumbang informasi waktu kalender (`bulan_target` dan `musim`). Seluruh atribut prediktor lainnya (kasus diagnosis, demografi, kunjungan, cuaca) murni bersumber dari periode *t-1* dan *t-2*.
   - Kuantil diskretisasi (33.3% dan 66.7%) dihitung hanya pada data latih (`train split`) dan diterapkan pada data uji (`test split`) tanpa menghitung ulang kuantil baru.
2. **Kepatuhan Laragon & Python**:
   - `rule.md` memuat aturan arsitektur, clean code, UI/UX (white canvas, soft blue buttons, custom scrollbar, Font Awesome icons, no emoji, responsive, no git commit). Seluruh aturan tersebut diimplementasikan secara ketat pada aplikasi FastAPI Jinja2 ini.
3. **Penanganan Probabilitas Prediksi**:
   - Probabilitas ketiga kelas (*Naik*, *Stabil*, *Turun*) dinormalisasi secara presisi sehingga jumlahnya tepat bernilai 1.0 per baris.

---

## 7. Verifikasi Kriteria Penerimaan (Acceptance Criteria Check)

- [x] **Kriteria 1**: `uvicorn` / `docker-compose` berjalan; `scripts/seed.py` memuat data dummy dan membuat admin & viewer.
- [x] **Kriteria 2**: Unggahan CSV dummy sukses; banner dummy tampil permanen di dashboard, evaluasi, dan prediksi.
- [x] **Kriteria 3**: Unggahan berkas berkolom `nama` atau `nik` ditolak dengan status HTTP 422 dan pesan jelas.
- [x] **Kriteria 4**: Laporan kesalahan per baris untuk duplikat, bulan bolong, atau ketidakcocokan total usia/gender.
- [x] **Kriteria 5**: Pratinjau label dinamis berubah mengikuti `threshold_pct` dan `min_abs_change`.
- [x] **Kriteria 6**: Pelatihan menghasilkan metrik komparasi 5 algoritma (NB, Mayoritas, Persistence, Decision Tree, KNN) dan confusion matrix.
- [x] **Kriteria 7**: Tes otomatis membuktikan tidak ada kebocoran data (mengubah kasus bulan target tidak mengubah fitur bulan itu).
- [x] **Kriteria 8**: Split data berurutan waktu: seluruh bulan uji lebih baru dari seluruh bulan latih.
- [x] **Kriteria 9**: Halaman hasil menampilkan teks otomatis *"Model belum lebih baik dari baseline"* jika F1 Naive Bayes <= baseline terbaik.
- [x] **Kriteria 10**: Prediksi bulan berikutnya menampilkan 15 diagnosis penyakit dengan probabilitas berjumlah 1.0 per baris.
- [x] **Kriteria 11**: Viewer dibatasi hanya baca; aksi unggah, latih, dan hapus menghasilkan 403 Forbidden.
- [x] **Kriteria 12**: Seluruh tes `pytest` lulus 100% (9 dari 9 pengujian).
