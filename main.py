import importlib.util
import io
import os
from pathlib import Path
import re
import sys
import threading
import types
from typing import Annotated, Any
import wave

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    InvalidVideoId,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
    YouTubeTranscriptApiException,
)

# Kompatibilitas Python 3.15: faster-whisper mengimpor `av` di top-level audio.py,
# padahal kita langsung meneruskan numpy float32 array 16 kHz dari decoder WAV internal.
if importlib.util.find_spec("av") is None and "av" not in sys.modules:
    sys.modules["av"] = types.ModuleType("av")

try:
    import ctranslate2
    from faster_whisper import WhisperModel

    WHISPER_AVAILABLE = True
    WHISPER_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover
    ctranslate2 = None  # type: ignore[assignment]
    WhisperModel = None  # type: ignore[assignment,misc]
    WHISPER_AVAILABLE = False
    WHISPER_IMPORT_ERROR = str(exc)

app = FastAPI(
    title="VoxScribe API",
    description="API lokal untuk mengekstrak transkrip video YouTube, live meeting (Zoom/Meet/Teams/Discord), dan file video/audio.",
    version="2.0.0",
)

# Izinkan semua origin agar bisa diakses langsung dari file HTML lokal (file://) maupun localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

# Regex untuk mengekstrak video_id (11 karakter) dari berbagai format URL YouTube
YOUTUBE_URL_REGEX = re.compile(
    r"(?:https?://)?(?:www\.|m\.|music\.)?"
    r"(?:youtube\.com/(?:watch\?(?:[^#]*&)?v=|shorts/|embed/|live/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})(?:[?&/#].*|$)"
)

ALLOWED_WHISPER_MODELS = {"tiny", "base", "small", "medium"}
ALLOWED_LANGUAGES = {"id", "en", "auto"}
LANGUAGE_NAMES = {
    "id": "Indonesian",
    "en": "English",
    "ms": "Malay",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
}

TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

_whisper_lock = threading.Lock()
_whisper_state: dict[str, Any] = {
    "model": None,
    "model_size": None,
    "device": None,
    "compute_type": None,
}


class TranscribeRequest(BaseModel):
    url: str = Field(..., min_length=1, description="URL video YouTube")


class TimestampItem(BaseModel):
    timestamp: str
    start: float
    duration: float
    text: str


class TranscribeResponse(BaseModel):
    video_id: str
    language: str
    language_code: str
    is_generated: bool
    transcript: str
    timestamps: list[TimestampItem]


class AudioTranscribeResponse(BaseModel):
    model_size: str
    device: str
    language: str
    language_code: str
    duration: float
    skipped_silence: bool
    transcript: str
    timestamps: list[TimestampItem]


def extract_video_id(url: str) -> str:
    """Mengekstrak video_id 11 karakter dari link YouTube."""
    cleaned_url = url.strip()
    match = YOUTUBE_URL_REGEX.search(cleaned_url)
    if not match:
        raise HTTPException(
            status_code=400,
            detail=(
                "Link YouTube tidak valid. Gunakan format "
                "youtube.com/watch?v=..., youtu.be/..., atau youtube.com/shorts/..."
            ),
        )
    return match.group(1)


def format_timestamp(seconds: float) -> str:
    """Mengubah detik menjadi format MM:SS atau HH:MM:SS."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def clean_snippet_text(text: str) -> str:
    """Membersihkan newline dan spasi berlebih dari potongan teks transkrip."""
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def fetch_youtube_transcript(video_id: str, languages: list[str]) -> dict[str, Any]:
    """
    Mengambil transkrip YouTube dengan prioritas bahasa.
    Mendukung youtube-transcript-api v1.x (fetch) serta fallback untuk v0.6.x (get_transcript).
    """
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        raw_snippets = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        return {
            "snippets": raw_snippets,
            "language": languages[0],
            "language_code": languages[0],
            "is_generated": False,
        }

    api = YouTubeTranscriptApi()
    fetched = api.fetch(video_id, languages=languages)
    return {
        "snippets": [
            {
                "text": snippet.text,
                "start": snippet.start,
                "duration": snippet.duration,
            }
            for snippet in fetched
        ],
        "language": getattr(fetched, "language", "Unknown"),
        "language_code": getattr(fetched, "language_code", "unknown"),
        "is_generated": getattr(fetched, "is_generated", False),
    }


def decode_wav_to_float32_16k(wav_bytes: bytes) -> tuple[np.ndarray, float]:
    """
    Mendekode file WAV PCM menjadi array float32 mono 16 kHz untuk faster-whisper
    tanpa memerlukan FFmpeg eksternal.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frame_count = wf.getnframes()
            raw_frames = wf.readframes(frame_count)
    except wave.Error as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Format audio WAV tidak valid: {exc}",
        ) from exc

    if frame_count == 0 or not raw_frames:
        return np.zeros(0, dtype=np.float32), 0.0

    if sample_width == 2:
        samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw_frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        samples = (np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Bit depth WAV ({sample_width * 8}-bit) tidak didukung. Gunakan PCM 16-bit.",
        )

    if channels > 1:
        usable_len = (len(samples) // channels) * channels
        samples = samples[:usable_len].reshape(-1, channels).mean(axis=1)

    duration_sec = len(samples) / float(sample_rate) if sample_rate > 0 else 0.0

    if sample_rate != TARGET_SAMPLE_RATE and len(samples) > 1:
        target_len = max(1, int(round(len(samples) * TARGET_SAMPLE_RATE / sample_rate)))
        old_idx = np.linspace(0.0, len(samples) - 1, num=len(samples), dtype=np.float64)
        new_idx = np.linspace(0.0, len(samples) - 1, num=target_len, dtype=np.float64)
        samples = np.interp(new_idx, old_idx, samples).astype(np.float32)

    return np.ascontiguousarray(samples, dtype=np.float32), duration_sec


def preprocess_speech_audio(audio: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Membersihkan audio sebelum masuk ke Whisper:
    - Hapus DC offset
    - Deteksi & pangkas bagian hening awal/akhir (energy-based VAD agar tidak halusinasi)
    - Normalisasi volume lembut agar suara kecil tetap terdengar jelas tanpa mengangkat noise
    Mengembalikan (audio_bersih, leading_trim_seconds).
    """
    if audio.size < int(TARGET_SAMPLE_RATE * 0.2):
        return np.zeros(0, dtype=np.float32), 0.0

    audio = audio - float(np.mean(audio))

    overall_rms = float(np.sqrt(np.mean(audio * audio)))
    if overall_rms < 0.0025:
        return np.zeros(0, dtype=np.float32), 0.0

    # Analisis energi per frame 30ms untuk memangkas hening awal/akhir
    frame_size = int(TARGET_SAMPLE_RATE * 0.03)
    num_frames = len(audio) // frame_size
    leading_trim_sec = 0.0

    if num_frames > 0:
        framed = audio[: num_frames * frame_size].reshape(num_frames, frame_size)
        frame_rms = np.sqrt(np.mean(framed * framed, axis=1))
        noise_floor = float(np.percentile(frame_rms, 15))
        speech_threshold = max(0.0045, noise_floor * 2.2)

        active = np.where(frame_rms >= speech_threshold)[0]
        if active.size == 0:
            return np.zeros(0, dtype=np.float32), 0.0

        pad_frames = 8  # ~240ms padding di awal & akhir ucapan
        start_frame = max(0, int(active[0]) - pad_frames)
        end_frame = min(num_frames, int(active[-1]) + pad_frames + 1)

        start_sample = start_frame * frame_size
        end_sample = min(len(audio), end_frame * frame_size)
        audio = audio[start_sample:end_sample]
        leading_trim_sec = start_sample / float(TARGET_SAMPLE_RATE)

    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if peak > 1e-4:
        # Target peak 0.75 dengan batas gain maksimal 5x agar background hiss tidak ikut naik
        gain = min(0.75 / peak, 5.0)
        audio = np.clip(audio * gain, -1.0, 1.0)

    return np.ascontiguousarray(audio, dtype=np.float32), leading_trim_sec


def _load_whisper_model_unlocked(model_size: str, force_cpu: bool = False) -> tuple[Any, str, str]:
    """Memuat model faster-whisper (mencoba CUDA lebih dulu, otomatis fallback ke CPU int8)."""
    if not WHISPER_AVAILABLE or WhisperModel is None:
        raise HTTPException(
            status_code=500,
            detail=f"faster-whisper belum siap di server: {WHISPER_IMPORT_ERROR}",
        )

    if (
        not force_cpu
        and _whisper_state["model"] is not None
        and _whisper_state["model_size"] == model_size
    ):
        return (
            _whisper_state["model"],
            _whisper_state["device"],
            _whisper_state["compute_type"],
        )

    if (
        force_cpu
        and _whisper_state["model"] is not None
        and _whisper_state["model_size"] == model_size
        and _whisper_state["device"] == "cpu"
    ):
        return (
            _whisper_state["model"],
            _whisper_state["device"],
            _whisper_state["compute_type"],
        )

    cpu_threads = max(2, min(8, os.cpu_count() or 4))

    if not force_cpu and ctranslate2 is not None and ctranslate2.get_cuda_device_count() > 0:
        try:
            model = WhisperModel(model_size, device="cuda", compute_type="float16")
            _whisper_state.update(
                {
                    "model": model,
                    "model_size": model_size,
                    "device": "cuda",
                    "compute_type": "float16",
                }
            )
            return model, "cuda", "float16"
        except Exception:
            pass

    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=cpu_threads,
    )
    _whisper_state.update(
        {
            "model": model,
            "model_size": model_size,
            "device": "cpu",
            "compute_type": "int8",
        }
    )
    return model, "cpu", "int8"


def run_whisper_transcription(
    audio_np: np.ndarray,
    language: str,
    model_size: str,
    leading_trim_sec: float,
    offset_seconds: float,
) -> tuple[list[TimestampItem], str, str, str]:
    """Menjalankan inferensi faster-whisper secara thread-safe dengan fallback otomatis ke CPU."""
    lang_arg = None if language == "auto" else language

    with _whisper_lock:
        model, device, _ = _load_whisper_model_unlocked(model_size, force_cpu=False)

        def _do_transcribe(active_model: Any) -> tuple[list[Any], Any]:
            segments_gen, info = active_model.transcribe(
                audio_np,
                language=lang_arg,
                beam_size=5,
                vad_filter=False,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )
            return list(segments_gen), info

        try:
            raw_segments, info = _do_transcribe(model)
        except Exception as exc:
            # Jika CUDA gagal saat runtime (misalnya cublas/cudnn DLL tidak ditemukan), fallback ke CPU
            if device == "cuda":
                model, device, _ = _load_whisper_model_unlocked(model_size, force_cpu=True)
                raw_segments, info = _do_transcribe(model)
            else:
                raise exc

    timestamps: list[TimestampItem] = []
    for seg in raw_segments:
        if getattr(seg, "no_speech_prob", 0.0) > 0.78:
            continue
        cleaned = clean_snippet_text(str(seg.text))
        if not cleaned:
            continue

        abs_start = max(0.0, offset_seconds + leading_trim_sec + float(seg.start))
        duration = max(0.0, float(seg.end) - float(seg.start))
        timestamps.append(
            TimestampItem(
                timestamp=format_timestamp(abs_start),
                start=round(abs_start, 2),
                duration=round(duration, 2),
                text=cleaned,
            )
        )

    detected_code = getattr(info, "language", None) or (language if language != "auto" else "id")
    detected_name = LANGUAGE_NAMES.get(detected_code, detected_code.upper())
    return timestamps, detected_code, detected_name, device


@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html", media_type="text/html")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "message": "YouTube & Live Meeting Transcribe API siap digunakan."}


@app.get("/whisper-status")
def whisper_status() -> dict[str, Any]:
    return {
        "available": WHISPER_AVAILABLE,
        "error": WHISPER_IMPORT_ERROR,
        "loaded_model": _whisper_state["model_size"],
        "device": _whisper_state["device"],
        "compute_type": _whisper_state["compute_type"],
        "supported_models": sorted(ALLOWED_WHISPER_MODELS),
    }


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe_video(payload: TranscribeRequest) -> TranscribeResponse:
    video_id = extract_video_id(payload.url)

    try:
        result = fetch_youtube_transcript(video_id, languages=["id", "en"])
    except InvalidVideoId as exc:
        raise HTTPException(
            status_code=400,
            detail="ID video YouTube tidak valid.",
        ) from exc
    except TranscriptsDisabled as exc:
        raise HTTPException(
            status_code=404,
            detail="Subtitle/transkrip dinonaktifkan oleh pemilik video ini.",
        ) from exc
    except NoTranscriptFound as exc:
        raise HTTPException(
            status_code=404,
            detail="Transkrip dalam bahasa Indonesia (id) atau Inggris (en) tidak ditemukan untuk video ini.",
        ) from exc
    except (VideoUnavailable, VideoUnplayable) as exc:
        raise HTTPException(
            status_code=404,
            detail="Video tidak ditemukan, bersifat privat, atau tidak dapat diputar.",
        ) from exc
    except (RequestBlocked, IpBlocked) as exc:
        raise HTTPException(
            status_code=429,
            detail="Permintaan sementara dibatasi oleh YouTube. Silakan coba beberapa saat lagi.",
        ) from exc
    except (CouldNotRetrieveTranscript, YouTubeTranscriptApiException) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gagal mengambil transkrip dari YouTube: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan pada server: {exc}",
        ) from exc

    timestamps: list[TimestampItem] = []
    text_parts: list[str] = []

    for item in result["snippets"]:
        cleaned_text = clean_snippet_text(str(item.get("text", "")))
        if not cleaned_text:
            continue

        start = float(item.get("start", 0.0))
        duration = float(item.get("duration", 0.0))

        text_parts.append(cleaned_text)
        timestamps.append(
            TimestampItem(
                timestamp=format_timestamp(start),
                start=round(start, 2),
                duration=round(duration, 2),
                text=cleaned_text,
            )
        )

    if not text_parts:
        raise HTTPException(
            status_code=404,
            detail="Transkrip ditemukan tetapi tidak berisi teks.",
        )

    full_paragraph = re.sub(r"\s+", " ", " ".join(text_parts)).strip()

    return TranscribeResponse(
        video_id=video_id,
        language=result["language"],
        language_code=result["language_code"],
        is_generated=result["is_generated"],
        transcript=full_paragraph,
        timestamps=timestamps,
    )


@app.post("/transcribe-audio", response_model=AudioTranscribeResponse)
def transcribe_audio(
    file: Annotated[UploadFile, File(description="File audio WAV PCM (16 kHz mono)")],
    language: Annotated[str, Form()] = "id",
    model_size: Annotated[str, Form()] = "small",
    offset_seconds: Annotated[float, Form()] = 0.0,
) -> AudioTranscribeResponse:
    if not WHISPER_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail=f"Library faster-whisper belum tersedia: {WHISPER_IMPORT_ERROR}",
        )

    language = language.strip().lower()
    model_size = model_size.strip().lower()

    if language not in ALLOWED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Bahasa tidak didukung. Pilih salah satu: {', '.join(sorted(ALLOWED_LANGUAGES))}.",
        )
    if model_size not in ALLOWED_WHISPER_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model Whisper tidak valid. Pilih salah satu: {', '.join(sorted(ALLOWED_WHISPER_MODELS))}.",
        )

    wav_bytes = file.file.read(MAX_AUDIO_UPLOAD_BYTES + 1)
    if not wav_bytes:
        raise HTTPException(status_code=400, detail="File audio kosong.")
    if len(wav_bytes) > MAX_AUDIO_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Ukuran file audio melebihi batas 100 MB.")

    audio_np, duration_sec = decode_wav_to_float32_16k(wav_bytes)
    clean_audio, leading_trim_sec = preprocess_speech_audio(audio_np)

    if clean_audio.size == 0:
        lang_code = language if language != "auto" else "id"
        return AudioTranscribeResponse(
            model_size=model_size,
            device=_whisper_state["device"] or "cpu",
            language=LANGUAGE_NAMES.get(lang_code, lang_code.upper()),
            language_code=lang_code,
            duration=round(duration_sec, 2),
            skipped_silence=True,
            transcript="",
            timestamps=[],
        )

    try:
        timestamps, detected_code, detected_name, device = run_whisper_transcription(
            audio_np=clean_audio,
            language=language,
            model_size=model_size,
            leading_trim_sec=leading_trim_sec,
            offset_seconds=max(0.0, offset_seconds),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mentranskrip audio dengan faster-whisper: {exc}",
        ) from exc

    full_paragraph = re.sub(r"\s+", " ", " ".join(item.text for item in timestamps)).strip()

    return AudioTranscribeResponse(
        model_size=model_size,
        device=device,
        language=detected_name,
        language_code=detected_code,
        duration=round(duration_sec, 2),
        skipped_silence=len(timestamps) == 0,
        transcript=full_paragraph,
        timestamps=timestamps,
    )
