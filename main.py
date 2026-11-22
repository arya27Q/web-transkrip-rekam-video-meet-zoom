from pathlib import Path
import re
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

app = FastAPI(
    title="YouTube Transcribe API",
    description="API untuk mengekstrak transkrip dari link video YouTube.",
    version="1.0.0",
)

# Izinkan semua origin agar bisa diakses langsung dari file HTML lokal (file://)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Regex untuk mengekstrak video_id (11 karakter) dari berbagai format URL YouTube
YOUTUBE_URL_REGEX = re.compile(
    r"(?:https?://)?(?:www\.|m\.|music\.)?"
    r"(?:youtube\.com/(?:watch\?(?:[^#]*&)?v=|shorts/|embed/|live/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})(?:[?&/#].*|$)"
)


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


BASE_DIR = Path(__file__).resolve().parent


@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html", media_type="text/html")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "message": "YouTube Transcribe API siap digunakan."}


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
