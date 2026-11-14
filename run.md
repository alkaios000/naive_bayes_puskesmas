# Panduan Menjalankan Aplikasi
## Sistem Prediksi Tren Kasus Penyakit — Puskesmas Sarang 1

Dokumen ini berisi panduan praktis dan lengkap untuk menjalankan aplikasi di komputer lokal (Windows/Laragon) maupun menggunakan Docker.

---

### 1. Cara Cepat Menjalankan (Quickstart)

Jika virtual environment (`.venv`) dan dependensi sudah terpasang, cukup buka terminal (**PowerShell** atau **Command Prompt**) di folder proyek:

```powershell
# Jalankan server aplikasi
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Setelah server aktif, buka peramban (browser) Anda:
- **Aplikasi Web**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Dokumentasi API (Swagger/OpenAPI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

#### Akun Login Default:
| Peran (Role) | Username | Password | Hak Akses |
|---|---|---|---|
| **Admin** | `admin` | `admin_puskesmas_sarang1` | Penuh: unggah dataset, latih model, inferensi prediksi, hapus data |
| **Viewer** | `viewer` | `viewer_puskesmas_sarang1` | Baca: dashboard analitik, evaluasi model, hasil prediksi |

---

### 2. Panduan Langkah Demi Langkah (Step-by-Step)

Gunakan langkah ini jika Anda baru mengkloning repositori atau memulai dari awal:

#### Langkah 1: Buka Terminal di Direktori Proyek
Pastikan terminal berada di direktori root aplikasi:
```powershell
cd c:\laragon\www\naive_bayes_puskesmas
```

#### Langkah 2: Buat & Aktifkan Lingkungan Virtual (Virtual Environment)
Jika folder `.venv` belum ada:
```powershell
# Membuat virtual environment dengan Python
python -m venv .venv
```

Aktifkan virtual environment:
- **PowerShell**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(Jika muncul error execution policy di PowerShell, jalankan sekali: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*
- **Command Prompt (CMD)**:
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
- **Git Bash**:
  ```bash
  source .venv/Scripts/activate
  ```

#### Langkah 3: Pasang Dependensi (Packages)
Pastikan semua pustaka yang dibutuhkan terpasang:
```powershell
pip install -r requirements.txt
```

#### Langkah 4: Konfigurasi File Lingkungan (`.env`)
Pastikan file `.env` sudah tersedia di folder root. Jika belum ada, salin dari `.env.example`:
```powershell
Copy-Item .env.example .env
```

#### Langkah 5: Jalankan Seeder Database
Inisialisasi tabel basis data, buat akun `admin` & `viewer`, muat dataset riwayat 2023–2025 (540 baris), dan lakukan pelatihan model awal:
```powershell
.\.venv\Scripts\python.exe scripts/seed.py
```
*Output akan menampilkan status: `=== SEEDING BERHASIL SELESAI ===`.*

#### Langkah 6: Jalankan Server Web
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Tanda server telah siap digunakan:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

---

### 3. Menjalankan Pengujian Otomatis (`pytest`)

Untuk memverifikasi seluruh fungsionalitas sistem (keamanan PII, pencegahan data leakage, split waktu kronologis, validasi label, dan fitur proposal):

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```
*Hasil: 10 pengujian passed 100%.*

---

### 4. Opsi Alternatif: Menjalankan Menggunakan Docker

Jika komputer Anda memiliki Docker Desktop terpasang:

```bash
# Menjalankan container di latar belakang
docker compose up -d --build
```
- Akses aplikasi di: [http://localhost:8000](http://localhost:8000)
- Untuk melihat log: `docker compose logs -f`
- Untuk menghentikan: `docker compose down`

---

### 5. Troubleshooting (Masalah Umum)

1. **Port 8000 Bentrok / Sudah Digunakan Aplikasi Lain**:
   Ganti port dengan perintah:
   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
   ```
   Lalu buka di browser: `http://127.0.0.1:8080`.

2. **Menghentikan Server**:
   Tekan kombinasi tombol `Ctrl + C` pada terminal tempat Uvicorn berjalan.
