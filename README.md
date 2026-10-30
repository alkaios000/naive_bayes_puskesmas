# Sistem Prediksi Tren Kasus Penyakit Terbanyak dengan Naive Bayes
**Puskesmas Sarang 1 Kabupaten Rembang**

Aplikasi web berbasis machine learning untuk memprediksi **tren jumlah kasus per penyakit pada bulan berikutnya** (kelas **Naik / Stabil / Turun**) menggunakan algoritma **Categorical Naive Bayes** yang dibandingkan berdampingan dengan 4 algoritma pembanding (**Mayoritas**, **Persistence**, **Decision Tree**, dan **K-Nearest Neighbors**), berdasarkan data rekapitulasi bulanan diagnosis penyakit ICD-10.

---

## 1. Arsitektur & Teknologi

Sistem dibangun dengan arsitektur web modern yang modular, aman, teruji, dan siap untuk deployment maupun kebutuhan riset akademik:

- **Backend Core**: Python 3.11+ / 3.13, **FastAPI**, **SQLAlchemy 2.x**, **Pydantic v2**.
- **Machine Learning**: `scikit-learn` (`CategoricalNB`, `DecisionTreeClassifier`, `KNeighborsClassifier`, `TimeSeriesSplit`), `pandas`, `NumPy`, `joblib`.
- **Database**: SQLite (default lokal) dengan dukungan migrasi mudah ke PostgreSQL / MySQL via variabel `DATABASE_URL`.
- **Frontend & Presentasi**: HTML5 semantik, CSS Modern Terkurasi (Desain bersih tanpa gaya AI-slop generik, palet warna berharmoni, sticky navigation tabs, kontras tinggi, typography Inter/Segoe UI), **Font Awesome 6**, dan visualisasi interaktif **Chart.js**. Disajikan via engine templating Jinja2.
- **Autentikasi & Keamanan**: JWT (JSON Web Token), hashing password menggunakan `bcrypt` (`passlib`), kontrol akses berbasis peran (RBAC: Admin & Viewer), penolakan kolom PII (nama, NIK, alamat), pencegahan kebocoran data (*zero data leakage*), dan banner permanen penanda dataset dummy.
- **Pengujian Otomatis**: `pytest` + `httpx TestClient` (**12 dari 12 pengujian lulus 100%**).

---

## 2. Struktur Proyek

```
naive_bayes_puskesmas/
├── app/
│   ├── main.py                # Inisialisasi FastAPI, middleware, static, router
│   ├── config.py              # Konfigurasi Pydantic Settings dari .env
│   ├── database.py            # Engine SQLAlchemy & SessionLocal SQLite
│   ├── models.py              # Skema ORM database (User, Dataset, Case, ModelRun, Prediction)
│   ├── schemas.py             # Pydantic validation schemas (I/O, User CRUD, Model Run)
│   ├── security.py            # Hashing Bcrypt, JWT token, dependencies RBAC
│   ├── routers/               # auth, users, datasets, cases, models, predictions, export, pages
│   ├── services/
│   │   ├── ingest.py          # Parsing CSV/XLSX, validasi kontinuitas, penolakan PII
│   │   ├── labeling.py        # Rumus pembentukan label tren (Naik/Stabil/Turun)
│   │   ├── features.py        # Ekstraksi lag fitur, pencegahan kebocoran, diskretisasi kuantil
│   │   ├── training.py        # Latih CategoricalNB + 4 pembanding, TimeSeriesSplit, tuning
│   │   ├── evaluation.py      # Metrik evaluasi, confusion matrix, rincian dinamis tahap 1-6
│   │   └── predict.py         # Inferensi prediksi bulan berikutnya (probabilitas = 1.0)
│   ├── templates/             # Jinja2: base, login, dashboard, users, upload, datasets, train, results, predictions
│   └── static/                # Modular CSS (base, layout, components, sticky tabs) & JS
├── dataset/                   # Direktori sumber dataset
│   ├── Rekap_Bulanan_2023-2025.csv       # Dataset historis riil Puskesmas Sarang 1
│   └── Rekap_Bulanan_DUMMY_2023-2025.csv # Dataset dummy untuk pengujian & simulasi
├── models_storage/            # Artefak model serialisasi machine learning (.joblib)
├── uploads/                   # Direktori target upload berkas dataset dari antarmuka
├── scripts/
│   └── seed.py                # Inisialisasi tabel, pembuatan akun default, & pemuatan dataset awal
├── tests/                     # 8 berkas pengujian otomatis pytest (12 skenario pengujian)
│   ├── conftest.py            # Fixture test client & pembersihan otomatis SQLite
│   ├── test_labeling.py       # Uji rumus penentuan label tren
│   ├── test_leakage.py        # Uji pencegahan kebocoran & split waktu kronologis
│   ├── test_ingest.py         # Uji penolakan PII & validasi data
│   ├── test_auth_roles.py     # Uji otorisasi RBAC (Viewer vs Admin)
│   ├── test_users.py          # Uji manajemen pengguna CRUD & RBAC
│   ├── test_api_flow.py       # Uji alur pelatihan & inferensi prediksi
│   └── test_proposal_enhancements.py # Uji Time Series expanding window & fitur proposal
├── trash/                     # Berkas non-esensial / arsip referensi (diabaikan oleh Git)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .env
├── .gitignore
└── README.md                  # Dokumentasi resmi arsitektur sistem & panduan instalasi
```

---

## 3. Akun Pengguna Awal (Default Credentials)

Sistem mengadopsi kontrol akses berbasis peran (*Role-Based Access Control*):

| Peran (Role) | Nama Pengguna | Kata Sandi | Hak Akses |
|---|---|---|---|
| **Admin** | `admin` | `admin_puskesmas_sarang1` | **Penuh**: Unggah dataset, latih model, eksekusi prediksi, kelola pengguna (CRUD User), hapus data. |
| **Viewer** | `viewer` | `viewer_puskesmas_sarang1` | **Baca (Read-Only)**: Akses dashboard analitik, metrik evaluasi model, rincian perhitungan, dan hasil prediksi. |

> **Fitur Manajemen Pengguna**: Admin dapat menambah pengguna baru, mengubah kata sandi, mengubah role, atau menonaktifkan akun langsung dari menu **Pengguna** di antarmuka web.

---

## 4. Cara Menjalankan Aplikasi

### Opsi A: Lingkungan Lokal (Windows / Laragon / PowerShell)

1. **Pastikan Lingkungan Virtual Aktif**:
   Jika menggunakan virtual environment `.venv`:
   ```powershell
   # Aktivasi di PowerShell
   .\.venv\Scripts\Activate.ps1
   ```
   *(Atau jalankan langsung melalui `.\.venv\Scripts\python.exe`)*

2. **Jalankan Seeder (Inisialisasi Tabel & Data Awal)**:
   ```powershell
   & ".\.venv\Scripts\python.exe" scripts/seed.py
   ```

3. **Jalankan Server Web FastAPI (Uvicorn)**:
   ```powershell
   & ".\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

4. **Buka Peramban (Browser)**:
   - **Aplikasi Web**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
   - **Dokumentasi API Interaktif (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - **Redoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### Opsi B: Menggunakan Docker & Docker Compose

```bash
docker-compose up --build -d
```
Aplikasi langsung dapat diakses di `http://localhost:8000`.

---

## 5. Menjalankan Pengujian Otomatis (`pytest`)

Jalankan seluruh skenario uji integritas dan validasi ilmiah:
```powershell
& ".\.venv\Scripts\python.exe" -m pytest -v
```
Output pengujian mencakup 12 uji skenario yang memastikan zero leakage, RBAC, rumus pelabelan, time series expanding window, dan API flow.

---

## 6. Fitur Unggulan & Kepatuhan Ilmiah Proposal

1. **Pencegahan Kebocoran Data (Zero Data Leakage)**:
   - Data bulan target $t$ hanya menyumbang informasi waktu kalender (`bulan_target` dan `musim`). Seluruh prediktor lainnya (kasus diagnosis, demografi gender/usia, kunjungan, cuaca) murni bersumber dari periode $t-1$ dan $t-2$.
   - Kuantil diskretisasi (33.3% dan 66.7%) dihitung hanya pada data latih (`train split`) dan diterapkan pada data uji (`test split`) tanpa menghitung ulang kuantil baru.
2. **Evaluasi Berurutan Waktu (Expanding Window Time Series)**:
   - Pengujian model menggunakan split kronologis, di mana seluruh data uji berasal dari masa depan relatif terhadap data latih, mencegah *look-ahead bias*.
3. **Rincian Perhitungan Dinamis Tahap 1 s.d. 6**:
   - Halaman **Evaluasi & Metrik** menyediakan rincian perhitungan matematis Naive Bayes langkah-demi-langkah yang dihitung secara dinamis dari model run aktif:
     - **Tahap 1**: Pembentukan data latih & ekstraksi fitur lag.
     - **Tahap 2**: Pelabelan tren dinamis berbasis persentase perubahan & ambang batas.
     - **Tahap 3**: Diskretisasi kuantil batas bawah/atas (*Rendah*, *Sedang*, *Tinggi*).
     - **Tahap 4**: Probabilitas Prior $P(C_k)$ per kelas.
     - **Tahap 5**: Probabilitas Likelihood Kondisional $P(X_i \mid C_k)$ dengan Laplace smoothing ($\alpha=1$).
     - **Tahap 6**: Perhitungan Posterior, argmax kelas, dan normalisasi probabilitas tepat $1.0$.
   - Dilengkapi **Sticky Sub-navigation Tab Bar** sehingga navigasi antar tahap tetap nyaman saat dokumen di-scroll ke bawah.
4. **Manajemen Pengguna Terintegrasi**:
   - Menu khusus Admin untuk mengelola akun pengguna, status aktif/nonaktif, dan audit role.

---

## 7. Status Kriteria Penerimaan (Acceptance Criteria Check)

- [x] **Kriteria 1**: `uvicorn` / `docker-compose` berjalan lancar; `scripts/seed.py` memuat data awal dan akun Admin & Viewer.
- [x] **Kriteria 2**: Unggahan CSV/XLSX tervalidasi; banner penanda dummy tampil permanen pada data dummy.
- [x] **Kriteria 3**: Unggahan berkas berkolom privasi/PII (`nama`, `nik`, dll.) otomatis ditolak dengan HTTP 422.
- [x] **Kriteria 4**: Laporan kesalahan per baris untuk data duplikat, bulan terlewat (*timeline gap*), atau ketidaksesuaian gender/usia.
- [x] **Kriteria 5**: Pratinjau label dinamis interaktif sesuai parameter `threshold_pct` dan `min_abs_change`.
- [x] **Kriteria 6**: Pelatihan menghasilkan metrik komparasi 5 algoritma (CategoricalNB, Mayoritas, Persistence, Decision Tree, KNN) serta Confusion Matrix.
- [x] **Kriteria 7**: Uji otomatis membuktikan tidak ada kebocoran data (*zero data leakage*).
- [x] **Kriteria 8**: Split data berurutan waktu kronologis (Expanding Window).
- [x] **Kriteria 9**: Notifikasi otomatis *"Model belum lebih baik dari baseline"* jika performa Naive Bayes belum melampaui baseline terbaik.
- [x] **Kriteria 10**: Prediksi bulan berikutnya menampilkan probabilitas dinormalisasi berjumlah 1.0 per baris.
- [x] **Kriteria 11**: Role Viewer dibatasi hanya-baca (aksi unggah, latih, dan hapus menghasilkan HTTP 403 Forbidden).
- [x] **Kriteria 12**: Seluruh 12 pengujian otomatis pada `pytest` lulus 100% tanpa error.
- [x] **Kriteria 13**: Modul Manajemen Pengguna (CRUD User) berfungsi penuh untuk Admin.
- [x] **Kriteria 14**: Rincian matematis tahap 1-6 tersaji dinamis dengan topbar & sticky tab bar yang rapi.
