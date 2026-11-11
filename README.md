# Face Recognition (Zero-Photo)

Aplikasi pengenalan wajah berbasis web. Foto **tidak disimpan**: wajah diubah menjadi vektor 128 angka (embedding, 512 bytes) di browser, lalu hanya vektor itu yang dikirim dan disimpan di database.

## Fitur

- Pendaftaran wajah lewat kamera langsung, scan KTP lewat bingkai kamera, atau upload file KTP.
- Pengenalan wajah real-time dari kamera, dengan kotak deteksi dan persentase kemiripan.
- Mode **Wajah Terdepan** (satu wajah di frame) dan **Deteksi Banyak Orang** (mencari target di foto grup).
- Peak lock: wajah cocok (>= 70%) dengan persentase tertinggi dikunci pada kotak hijau. Wajah tidak cocok ditandai merah.
- Upload foto untuk dicocokkan, dengan hasil beranotasi.
- Simulasi kedatangan, log pengenalan, dan metrik penyimpanan.
- Import model MC dari folder `mc/`.

## Struktur Proyek

```
FaceRecognition/
├── backEnd/    Laravel (REST API + MySQL/SQLite)
├── frontEnd/   Vue 3 + Vite (UI + ekstraksi wajah di browser)
└── mc/         Gambar referensi model (Michael Jackson)
```

### Backend (`backEnd/`)

- `app/Http/Controllers/FaceRecognitionController.php`: endpoint API.
- `app/Services/FaceRecognitionService.php`: perhitungan jarak dan kemiripan vektor.
- `app/Models/FaceProfile.php`, `FaceRecognitionLog.php`: model data.
- `routes/api.php`: daftar route.

### Frontend (`frontEnd/`)

- `src/App.vue`: halaman utama (kamera, form daftar, hasil, log).
- `src/utils/faceExtractor.js`: deteksi dan ekstraksi vektor wajah, dan pencocokan lokal.
- `src/utils/mcModelTemplates.js`: template model referensi.
- Library `face-api` dimuat lewat CDN di `index.html`, jadi perlu internet saat pertama kali jalan.

## Kebutuhan

- PHP 8.2+ dengan Composer (misal XAMPP)
- MySQL (default) atau SQLite
- Node.js `^22.18.0` atau `>=24.12.0`
- Browser dengan akses kamera

## Cara Menjalankan

### 1. Backend

```bash
cd backEnd
composer install
copy .env.example .env      # lalu atur koneksi database
php artisan key:generate
```

Konfigurasi default di `.env`:

```
DB_CONNECTION=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_DATABASE=facerecognetion
DB_USERNAME=root
DB_PASSWORD=
```

Buat database `facerecognetion` di MySQL, lalu:

```bash
php artisan migrate
php artisan db:seed --class=FaceProfileSeeder
php artisan serve           # http://127.0.0.1:8000
```

Di Windows bisa memakai `migrate.bat` dan `serve.bat`. `clear-data.bat` dipakai untuk menghapus data.

### 2. Frontend

```bash
cd frontEnd
npm install
npm run dev                 # http://localhost:5173
```

Vite mem-proxy `/api` ke `http://127.0.0.1:8000`, jadi backend harus jalan lebih dulu.

Build produksi:

```bash
npm run build
```

## Endpoint API

Semua endpoint berawalan `/api`.

| Method | Endpoint | Fungsi |
|--------|----------|--------|
| GET | `/ping` | Cek status server |
| GET | `/face/profiles` | Daftar profil terdaftar |
| POST | `/face/register` | Daftarkan nama, role, dan embedding 128 angka |
| POST | `/face/import-mc-model` | Import model MC |
| DELETE | `/face/profiles/{id}` | Hapus satu profil |
| DELETE | `/face/clear` | Hapus semua data |
| POST | `/face/recognize` | Cocokkan satu atau banyak embedding |
| POST | `/face/simulate-seed` | Isi data simulasi |
| POST | `/face/simulate-visit` | Simulasi kedatangan |
| GET | `/face/logs` | Log pengenalan |
| GET | `/face/storage-metrics` | Metrik penyimpanan |

## Cara Kerja Singkat

1. Browser mendeteksi wajah dan mengubahnya jadi vektor 128 angka.
2. Pendaftaran: nama, role, dan vektor dikirim ke `/api/face/register`. Foto dibuang.
3. Pengenalan: vektor dicocokkan secara lokal (respons instan) dan dicatat ke backend sebagai log.
4. Kemiripan dihitung dari jarak antar vektor, ditampilkan sebagai persentase. Batas lulus adalah 70%.

## Catatan

- Akses kamera butuh `localhost` atau HTTPS.
- Akurasi dipengaruhi pencahayaan, sudut wajah, dan kualitas foto.
- Jika PHP tidak punya ekstensi GD, fitur yang butuh pemrosesan gambar di server tidak akan jalan. Ekstraksi wajah utama berjalan di browser.
