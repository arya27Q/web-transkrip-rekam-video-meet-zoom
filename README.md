# VoxScribe v3 — YouTube, Live Meeting & Local Video Transcriber

Aplikasi web fullstack lokal berarsitektur modular untuk mengekstrak transkrip dari **video YouTube**, **rekaman Live Meeting** (Google Meet, Zoom, Microsoft Teams, Discord — baik versi Web maupun Aplikasi Desktop), serta **file video/audio lokal** secara gratis dan privat menggunakan **FastAPI**, **WebSocket**, **SQLite**, dan **faster-whisper**.

---

## Fitur Utama

- **Workspace & Riwayat Transkrip (SQLite CRUD)**
  - Semua hasil transkrip (YouTube, Live Meeting, Import Video) tersimpan otomatis di database SQLite (`data/voxscribe.db`) beserta rekaman audionya (`data/audio/`).
  - **Sidebar Riwayat** dengan pencarian full-text lintas semua transkrip, filter kategori (*YouTube / Meeting / Video*), rename judul, serta tombol hapus.
  - **Inline Edit Mode** langsung di halaman (baik di tab *Paragraf Bersih* maupun per baris *Timestamp*) untuk memperbaiki typo sebelum di-export.
- **Pemisah Pembicara Otomatis (*Dual-Channel Speaker Separation*)**
  - Saat merekam Live Meeting dengan Mikrofon aktif, suara **Mikrofon Anda** dan **Audio Tab/Sistem Meeting** direkam di 2 kanal stereo terpisah.
  - Hasil transkrip otomatis diberi label dialog **`Anda`** (biru) dan **`Peserta`** (hijau) secara kronologis tanpa perlu model diarization tambahan.
- **Live Meeting Recorder + Real-Time WebSocket**
  - Menangkap suara semua peserta meeting (2 orang atau lebih) dari tab browser maupun audio sistem Windows + mikrofon Anda.
  - Audio dibersihkan di browser (*Echo Cancellation*, *Noise Suppression*, *Auto Gain Control*, *High-Pass/Low-Pass Filter*, *Dynamics Compressor*) dengan **dual audio level meter** & slider **Gain Tab/Mic**.
  - Pilihan **Rekam Langsung (transkrip saat Stop)** atau **Live Streaming via WebSocket (`/ws/transcribe`)** tiap ~8 / ~15 detik (dengan fallback HTTP otomatis).
- **Interactive Transcript & Sync Audio Player**
  - **Search & Highlight** kata kunci di dalam transkrip aktif lengkap dengan penghitung kecocokan.
  - Klik timestamp mana saja untuk melompatkan audio player langsung ke detik tersebut (plus highlight baris aktif saat audio diputar).
- **Import Video / Audio Lokal (Non-YouTube)**
  - Mendukung `.mp4`, `.webm`, `.mov`, `.mkv`, `.m4v`, `.mp3`, `.m4a`, `.wav`, `.ogg` tanpa FFmpeg eksternal (didekode langsung di browser lalu diproses per 30 detik).
- **Export Multi-Format**
  - **`.PDF`** (A4 rapi), **`.DOCX`** (Microsoft Word), **`.TXT`**, **`.SRT`** (subtitle CapCut/Premiere), **`.MD`** (Notion/Obsidian), dan **`.WAV`** (audio rekaman bersih).

---

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn, WebSockets, SQLite (`sqlite3`), Pydantic, `youtube-transcript-api`, `faster-whisper`, `numpy`
- **Frontend:** HTML5, Tailwind CSS (CDN), Vanilla JavaScript, Web Audio API (`AudioWorklet`, `ChannelMergerNode`), jsPDF (CDN)
- **Testing & Container:** `pytest`, `httpx` (`TestClient`), Docker & Docker Compose

---

## Cara Instalasi & Menjalankan

### Opsi 1 — Jalankan Lokal (Python `venv`)

1. **Clone repository & masuk folder:**
   ```bash
   git clone https://github.com/arya27Q/web-transkrip-rekam-video-meet-zoom.git
   cd web-transkrip-rekam-video-meet-zoom
   ```

2. **Buat & aktifkan virtual environment:**
   ```powershell
   # Windows PowerShell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   ```bash
   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   - **Python 3.10 – 3.13:**
     ```bash
     pip install -r requirements.txt
     ```
   - **Khusus Python 3.15 (Windows):**
     ```powershell
     pip install fastapi uvicorn websockets youtube-transcript-api pydantic python-multipart ctranslate2 huggingface-hub tokenizers numpy tqdm pytest httpx
     pip install --no-deps faster-whisper
     ```

4. **Jalankan server:**
   ```powershell
   uvicorn main:app --reload
   ```
   *(Atau tanpa aktivasi `venv`: `.\venv\Scripts\uvicorn.exe main:app --reload`)*

5. **Buka di Chrome / Edge:**
   - Aplikasi Web: `http://127.0.0.1:8000`
   - Dokumentasi API (Swagger): `http://127.0.0.1:8000/docs`

### Opsi 2 — Jalankan dengan Docker Compose

```bash
docker compose up --build
```
Database SQLite, rekaman audio, dan cache model Whisper tersimpan persisten di volume Docker (`voxscribe_data` dan `voxscribe_hf_cache`).

---

## Menjalankan Automated Tests (`pytest`)

```powershell
.\venv\Scripts\pytest.exe -v
```

---

## Panduan Rekam Meeting

- **Meeting via Browser (Web)** *(Google Meet, Zoom Web, MS Teams Web, Discord Web)*:
  Klik **Mulai Rekam Meeting** → pilih tab meeting → **centang `"Share tab audio"`**.
- **Aplikasi Desktop Windows** *(Zoom App, MS Teams App, Discord App, Meet App)*:
  Klik **Mulai Rekam Meeting** → pindah ke tab **Entire Screen (Seluruh Layar)** → **centang `"Also share system audio"`** *(jangan pilih tab "Window")*.
- **Tips Suara Bersih:** Mute notifikasi/tab lain saat mode *Entire Screen*, dan samakan perangkat Speaker di aplikasi meeting dengan output default Windows.

---

## Endpoint API

| Method | Endpoint | Deskripsi |
| --- | --- | --- |
| `GET` | `/` | Halaman web **VoxScribe** (`index.html`) |
| `GET` | `/health` | Cek status server backend |
| `GET` | `/whisper-status` | Status model `faster-whisper` & device (`cpu`/`cuda`) |
| `POST` | `/transcribe` | Transkrip video YouTube + simpan otomatis ke SQLite |
| `POST` | `/transcribe-audio` | Transkrip WAV 16 kHz (mono / stereo dual-speaker) |
| `WS` | `/ws/transcribe` | Streaming transkrip audio real-time via WebSocket |
| `GET` | `/sessions` | Daftar & cari riwayat transkrip (`?q=...&source=...`) |
| `POST` | `/sessions` | Simpan sesi transkrip baru ke SQLite |
| `GET` | `/sessions/{id}` | Detail lengkap sesi transkrip |
| `PATCH` | `/sessions/{id}` | Ubah judul atau simpan hasil edit transkrip |
| `DELETE` | `/sessions/{id}` | Hapus sesi transkrip beserta audionya |
| `POST/GET` | `/sessions/{id}/audio` | Simpan & putar/unduh file audio `.WAV` sesi |
| `POST` | `/export/docx` | Export dokumen Microsoft Word (`.docx`) |
| `POST` | `/export/srt` | Export file subtitle (`.srt`) |
| `POST` | `/export/md` | Export dokumen Markdown (`.md`) |

---

## Struktur Project

```text
yt-transcribe-web/
├── main.py              # FastAPI app, REST & WebSocket routes
├── database.py          # Layer database SQLite & penyimpanan audio
├── audio_service.py     # Decoder WAV dual-channel, VAD, & faster-whisper engine
├── exporters.py         # Generator export .DOCX, .SRT, dan .MD
├── index.html           # Frontend SPA (Sidebar Riwayat, Editor, Dual-Speaker, Export)
├── tests/
│   └── test_app.py      # Automated test suite (pytest + FastAPI TestClient)
├── Dockerfile           # Container image siap deploy
├── docker-compose.yml   # Orkestrasi container + persistent volumes
├── requirements.txt     # Daftar library Python
└── README.md            # Dokumentasi project
```
