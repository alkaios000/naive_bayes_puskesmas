# Panduan Instalasi Aplikasi di Laptop Baru
## Sistem Prediksi Tren Kasus Penyakit — Puskesmas Sarang 1

Dokumen ini berisi daftar perangkat lunak yang perlu dipasang serta langkah-langkah lengkap untuk memindahkan dan menjalankan aplikasi **Prediksi Tren Kasus Penyakit (Naive Bayes)** di laptop atau komputer baru.

> **Catatan Arsitektur**: Meskipun folder proyek ini diletakkan di direktori Laragon (`laragon\www`), aplikasi ini dibangun dengan **Python (FastAPI)** dan database default **SQLite** (bawaan Python). Anda **tidak wajib menginstal Laragon, XAMPP, Apache, PHP, ataupun MySQL** di laptop baru.

---

## 1. Apa Saja yang Perlu Diinstal di Laptop Baru?

### Jalur A: Instalasi Standar (Python Lokal — Direkomendasikan)
1. **Python 3.11 – 3.13 (64-bit)** *(Wajib)*
   - Versi rujukan proyek: Python `3.11` (seperti di `Dockerfile`) hingga `3.13` (lingkungan pengembangan lokal `Python 3.13.0`).
   - Unduh dari situs resmi: [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - **PENTING (Windows)**: Saat installer dijalankan, centang opsi **`Add python.exe to PATH`** sebelum menekan *Install Now*.
2. **Git** *(Opsional — wajib jika mengambil proyek via `git clone`)*
   - Unduh dari: [https://git-scm.com/downloads](https://git-scm.com/downloads)
   - Tidak diperlukan jika proyek dipindahkan menggunakan file ZIP/RAR atau flashdisk.
3. **Dependensi Python (`requirements.txt`)** *(Diinstal ke dalam `.venv` pada langkah instalasi)*
   - Framework Web & Server: `fastapi`, `uvicorn[standard]`, `jinja2`, `python-multipart`
   - Database & Validasi: `sqlalchemy`, `pydantic`, `pydantic-settings`
   - Keamanan & Autentikasi: `python-jose[cryptography]`, `passlib`, `bcrypt`
   - Machine Learning & Pengolahan Data: `pandas`, `numpy`, `scikit-learn`, `joblib`, `openpyxl` (untuk file `.xlsx`)
   - Pengujian Otomatis: `pytest`, `httpx`
4. **Browser Modern & Koneksi Internet**
   - Browser: Google Chrome, Microsoft Edge, atau Mozilla Firefox.
   - Koneksi internet diperlukan saat instalasi package (`pip install`) serta saat membuka tampilan web karena font dan pustaka UI (*Plus Jakarta Sans* dari Google Fonts, *Font Awesome 6.5.1* dari `cdnjs.cloudflare.com`, dan *Chart.js* dari `cdn.jsdelivr.net`) dimuat melalui CDN.

### Jalur B: Instalasi Alternatif Menggunakan Docker (Tanpa Instal Python)
1. **Docker Desktop** (Windows/macOS) atau **Docker Engine + Docker Compose** (Linux).
   - Unduh dari: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
   - Jika memilih jalur ini, Python dan seluruh package di `requirements.txt` akan dipasang otomatis di dalam container (`python:3.11-slim`).

---

## 2. Langkah-Langkah Instalasi (Jalur Standar: Python Lokal)

### Langkah 1: Verifikasi Python (dan Git) di Laptop Baru
Buka terminal (**PowerShell** atau **Command Prompt** di Windows, atau **Terminal** di macOS/Linux), lalu cek versi:

```powershell
python --version
pip --version
```

Pastikan muncul versi Python `3.11.x` s.d. `3.13.x`. Jika menggunakan Git, cek juga:
```powershell
git --version
```

---

### Langkah 2: Pindahkan Proyek ke Laptop Baru
Pilih salah satu cara berikut sesuai cara Anda membawa kode aplikasi:

#### Opsi 1 — Clone dari GitHub
```powershell
git clone https://github.com/alkaios000/naive_bayes_puskesmas.git
cd naive_bayes_puskesmas
```
*(File `.venv/`, `.env`, `naive_bayes_puskesmas.db`, `models_storage/`, dan `uploads/` tidak ikut di dalam Git karena masuk `.gitignore`, sehingga akan dibuat pada langkah berikutnya.)*

#### Opsi 2 — Salin Folder / Ekstrak ZIP atau RAR
1. Salin atau ekstrak folder proyek `naive_bayes_puskesmas` ke lokasi tujuan di laptop baru (misalnya `C:\laragon\www\naive_bayes_puskesmas` atau `C:\projects\naive_bayes_puskesmas`).
2. Buka terminal lalu arahkan ke dalam folder proyek:
   ```powershell
   cd C:\laragon\www\naive_bayes_puskesmas
   ```
3. **PENTING jika menyalin folder secara utuh**: Jika folder `.venv` dari laptop lama ikut tersalin, **hapus folder `.venv` tersebut terlebih dahulu** karena path Python di dalam `.venv` terikat pada laptop asal:
   ```powershell
   Remove-Item -Recurse -Force .venv
   ```

---

### Langkah 3: Buat dan Aktifkan Virtual Environment (`.venv`)
Buat virtual environment baru agar dependensi proyek terisolasi dan tidak bentrok dengan proyek Python lain:

```powershell
python -m venv .venv
```

Aktifkan `.venv` sesuai terminal yang digunakan:
- **Windows PowerShell**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(Jika muncul error `running scripts is disabled on this system`, jalankan perintah berikut terlebih dahulu lalu aktifkan ulang:)*
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\.venv\Scripts\Activate.ps1
  ```
- **Windows Command Prompt (CMD)**:
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
- **Git Bash / macOS / Linux**:
  ```bash
  source .venv/Scripts/activate   # Git Bash (Windows)
  # atau
  source .venv/bin/activate       # macOS / Linux
  ```

Setelah aktif, nama `(.venv)` akan muncul di awal baris terminal.

---

### Langkah 4: Instal Seluruh Dependensi Python
Perbarui `pip` dan instal semua pustaka dari `requirements.txt` *(membutuhkan koneksi internet)*:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Tunggu hingga seluruh paket (`fastapi`, `uvicorn`, `pandas`, `scikit-learn`, `openpyxl`, dll.) selesai terpasang tanpa error.

---

### Langkah 5: Siapkan File Konfigurasi Lingkungan (`.env`)
Aplikasi membaca konfigurasi dari file `.env` (berdasarkan `app/config.py`). Jika file `.env` belum ada, salin dari `.env.example`:

- **Windows PowerShell**:
  ```powershell
  Copy-Item .env.example .env
  ```
- **Windows CMD**:
  ```cmd
  copy .env.example .env
  ```
- **macOS / Linux / Git Bash**:
  ```bash
  cp .env.example .env
  ```

> **Catatan Konfigurasi `.env`**:
> - Default database sudah menggunakan SQLite (`DATABASE_URL="sqlite:///./naive_bayes_puskesmas.db"`), sehingga langsung siap pakai tanpa setup server database terpisah.
> - Jika aplikasi dipakai di lingkungan produksi atau jaringan bersama, ubah nilai `SECRET_KEY` menjadi string acak yang aman.
> - Jika ingin mengganti ke MySQL atau PostgreSQL, ubah `DATABASE_URL` di `.env` dan instal driver tambahannya (misalnya `pip install pymysql` untuk MySQL atau `pip install psycopg2-binary` untuk PostgreSQL).
> - Direktori `uploads/` dan `models_storage/` akan otomatis dibuat oleh sistem jika belum ada.

---

### Langkah 6: Inisialisasi Database, Akun Default, & Data Awal
Jalankan skrip seeder (di Windows PowerShell/CMD, atau cukup `python scripts/seed.py` jika `.venv` sudah aktif di Windows/macOS/Linux):

```powershell
.\.venv\Scripts\python.exe scripts/seed.py
```

Skrip `scripts/seed.py` akan menjalankan 4 tahap:
1. `[1/4]` Membuat tabel database di `naive_bayes_puskesmas.db`.
2. `[2/4]` Membuat akun login default (`admin` dan `viewer`) sesuai nilai di `.env`.
3. `[3/4]` Memuat dataset dummy `dataset/Rekap_Bulanan_DUMMY_2023-2025.csv` (jika berkas tersebut tersedia).
4. `[4/4]` Melatih model awal dan menghasilkan prediksi bulan berikutnya secara otomatis (jika dataset dummy berhasil dimuat).

> **Catatan Penting (Jika File Dataset Dummy Tidak Ada)**:
> - Jika Anda menyalin folder proyek lokal yang hanya berisi `dataset/Rekap_Bulanan_2023-2025.csv` (tanpa `Rekap_Bulanan_DUMMY_2023-2025.csv`), skrip `seed.py` **tetap berhasil** membuat tabel database dan akun `admin` & `viewer`, namun tahap `[3/4]` dan `[4/4]` akan dilewati.
> - Pada kondisi tersebut, silakan login sebagai **Admin** setelah server berjalan, buka menu **Unggah Dataset**, unggah file `dataset/Rekap_Bulanan_2023-2025.csv`, lalu buka menu **Latih Model** untuk melatih model pertama kali.
> - Jika Anda menyalin `naive_bayes_puskesmas.db` dan `models_storage/` dari laptop lama yang versi Python/scikit-learn-nya berbeda, disarankan menghapus `naive_bayes_puskesmas.db` dan isi `models_storage/` lalu menjalankan ulang `scripts/seed.py` (atau latih ulang model melalui UI) agar artefak `.joblib` kompatibel dengan versi `scikit-learn` di laptop baru.

---

### Langkah 7: Jalankan Server Web Aplikasi
Jalankan server FastAPI menggunakan Uvicorn (atau `python -m uvicorn ...` jika `.venv` sudah aktif):

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Server siap digunakan jika muncul output:
```text
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

Buka browser di laptop baru dan akses:
- **Aplikasi Web**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Health Check API**: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
- **Dokumentasi API (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

#### Akun Login Default:
| Peran (Role) | Username | Password | Hak Akses |
|---|---|---|---|
| **Admin** | `admin` | `admin_puskesmas_sarang1` | Penuh (unggah dataset, latih model, prediksi, kelola pengguna, hapus data) |
| **Viewer** | `viewer` | `viewer_puskesmas_sarang1` | Hanya-baca (dashboard, hasil evaluasi, hasil prediksi) |

*(Untuk mengakses aplikasi dari komputer/HP lain dalam satu jaringan Wi-Fi/LAN yang sama, jalankan Uvicorn dengan `--host 0.0.0.0 --port 8000` tanpa `--reload`, lalu buka `http://<IP-LAPTOP>:8000`.)*

---

### Langkah 8: Verifikasi Pengujian Otomatis (`pytest`) *(Opsional)*
Untuk memastikan seluruh komponen sistem berjalan normal di laptop baru:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

> **Catatan**: 2 dari 12 skenario uji (`test_api_flow.py` dan `test_proposal_enhancements.py`) memeriksa keberadaan berkas `dataset/Rekap_Bulanan_DUMMY_2023-2025.csv`. Jika proyek di-clone dari Git (file dummy ada), seluruh 12 pengujian akan lulus. Jika file dummy tidak ada di folder `dataset/`, 10 pengujian akan lulus dan 2 pengujian tersebut akan gagal karena berkas dummy tidak ditemukan.

---

## 3. Langkah Alternatif: Instalasi Menggunakan Docker

Jika Anda tidak ingin menginstal Python dan dependensi secara manual di laptop baru, gunakan Docker:

1. Pastikan **Docker Desktop** sudah terinstal dan sedang berjalan (*Docker Engine running*).
2. Jika file `naive_bayes_puskesmas.db` belum ada di folder proyek (misalnya setelah `git clone`), **buat file kosong terlebih dahulu** agar Docker tidak membuatnya menjadi folder saat *bind mount* (`docker-compose.yml`):
   - **PowerShell**:
     ```powershell
     if (-not (Test-Path naive_bayes_puskesmas.db)) { New-Item -ItemType File -Path naive_bayes_puskesmas.db | Out-Null }
     ```
   - **macOS / Linux / Git Bash**:
     ```bash
     touch naive_bayes_puskesmas.db
     ```
3. Siapkan file `.env` dari `.env.example` (opsional, karena variabel lingkungan utama juga sudah didefinisikan di `docker-compose.yml`).
4. Buka terminal di folder proyek, lalu build dan jalankan container:
   ```powershell
   docker compose up -d --build
   ```
5. Jika database (`naive_bayes_puskesmas.db`) masih kosong, jalankan seeder di dalam container:
   ```powershell
   docker compose exec web python scripts/seed.py
   ```
6. Buka aplikasi di browser: [http://localhost:8000](http://localhost:8000)

Perintah pendukung Docker:
- Melihat log aplikasi: `docker compose logs -f`
- Menghentikan aplikasi: `docker compose down`

---

## 4. Ringkasan Cepat (Checklist Laptop Baru)

1. [ ] Instal **Python 3.11–3.13** (centang *Add python.exe to PATH* saat instalasi di Windows).
2. [ ] Clone atau salin folder proyek ke laptop baru (hapus `.venv` lama jika ikut tersalin).
3. [ ] Buat virtual environment: `python -m venv .venv` dan aktifkan.
4. [ ] Instal package: `pip install -r requirements.txt`.
5. [ ] Buat `.env`: `Copy-Item .env.example .env`.
6. [ ] Inisialisasi DB & user: `.\.venv\Scripts\python.exe scripts/seed.py`.
7. [ ] Jalankan server: `.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`.
8. [ ] Buka `http://127.0.0.1:8000` dan login (`admin` / `admin_puskesmas_sarang1`).

---

## 5. Troubleshooting Umum di Laptop Baru

1. **`python` atau `pip` tidak dikenali (`not recognized`)**
   - Penyebab: Opsi *Add python.exe to PATH* belum dicentang saat menginstal Python, atau terminal belum dibuka ulang setelah instalasi.
   - Solusi: Tutup dan buka ulang terminal, atau jalankan ulang installer Python lalu pilih *Modify* dan pastikan *Add Python to environment variables* aktif.
2. **Error `Activate.ps1 cannot be loaded because running scripts is disabled` di PowerShell**
   - Solusi: Jalankan `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` terlebih dahulu, atau langsung jalankan aplikasi tanpa aktivasi menggunakan `.\.venv\Scripts\python.exe`.
3. **Error `ModuleNotFoundError: No module named 'fastapi'` (atau modul lain)**
   - Penyebab: Menjalankan perintah memakai Python global, bukan Python dari `.venv`, atau belum menjalankan `pip install -r requirements.txt`.
   - Solusi: Pastikan selalu menggunakan `.\.venv\Scripts\python.exe` atau pastikan `(.venv)` sudah aktif sebelum menjalankan `pip install -r requirements.txt` dan `uvicorn`.
4. **Port `8000` bentrok / sudah dipakai aplikasi lain**
   - Solusi: Ubah port menjadi misalnya `8080`:
     ```powershell
     .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
     ```
     Lalu buka `http://127.0.0.1:8080`.
5. **Grafik (Chart.js) atau ikon (Font Awesome) tidak tampil di browser**
   - Penyebab: Laptop belum terhubung ke internet (atau akses ke `cdn.jsdelivr.net` / `cdnjs.cloudflare.com` terblokir), sehingga skrip CDN di `app/templates/base.html` gagal dimuat.
   - Solusi: Pastikan koneksi internet aktif lalu muat ulang halaman (`Ctrl + F5`).
