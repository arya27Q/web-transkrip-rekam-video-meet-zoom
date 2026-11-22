import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import re
from typing import Annotated, Any

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
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

from audio_service import (
    ALLOWED_LANGUAGES,
    ALLOWED_WHISPER_MODELS,
    MAX_AUDIO_UPLOAD_BYTES,
    WHISPER_AVAILABLE,
    WHISPER_IMPORT_ERROR,
    clean_snippet_text,
    format_timestamp,
    get_whisper_status,
    transcribe_wav_bytes,
)
import database
from exporters import build_docx_bytes, build_markdown, build_srt

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(_app: FastAPI):
    database.init_db()
    yield


app = FastAPI(
    title="VoxScribe API",
    description=(
        "API lokal untuk mengekstrak transkrip video YouTube, live meeting "
        "(Zoom/Meet/Teams/Discord) dengan pemisah pembicara, file video/audio, "
        "riwayat SQLite, WebSocket real-time, serta export DOCX/SRT/MD."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# Izinkan semua origin agar bisa diakses dari localhost maupun file HTML lokal
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

YOUTUBE_URL_REGEX = re.compile(
    r"(?:https?://)?(?:www\.|m\.|music\.)?"
    r"(?:youtube\.com/(?:watch\?(?:[^#]*&)?v=|shorts/|embed/|live/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})(?:[?&/#].*|$)"
)


class TranscribeRequest(BaseModel):
    url: str = Field(..., min_length=1, description="URL video YouTube")
    save: bool = Field(default=True, description="Simpan otomatis ke riwayat SQLite")


class TimestampItem(BaseModel):
    timestamp: str
    start: float
    duration: float
    text: str
    speaker: str | None = None


class TranscribeResponse(BaseModel):
    session_id: str | None = None
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
    has_speakers: bool
    transcript: str
    timestamps: list[TimestampItem]


class SessionCreateRequest(BaseModel):
    title: str = Field(..., min_length=1)
    source: str = Field(..., pattern="^(youtube|meeting|file)$")
    source_ref: str = ""
    language: str = "Indonesian"
    language_code: str = "id"
    model_size: str | None = None
    device: str | None = None
    is_generated: bool = False
    has_speakers: bool = False
    transcript: str = ""
    timestamps: list[TimestampItem] = Field(default_factory=list)


class SessionUpdateRequest(BaseModel):
    title: str | None = None
    transcript: str | None = None
    timestamps: list[TimestampItem] | None = None


class ExportRequest(BaseModel):
    title: str = "Transkrip VoxScribe"
    source: str = "meeting"
    source_ref: str = ""
    language: str = "Indonesian"
    language_code: str = "id"
    model_size: str | None = None
    mode: str = Field(default="timestamps", pattern="^(paragraph|timestamps)$")
    transcript: str = ""
    timestamps: list[TimestampItem] = Field(default_factory=list)


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


def _safe_filename(name: str, fallback: str = "transkrip") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    return cleaned[:80] or fallback


# =====================================================================
# Frontend & Health Routes
# =====================================================================
@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html", media_type="text/html")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "message": "VoxScribe API v3 siap digunakan."}


@app.get("/whisper-status")
def whisper_status() -> dict[str, Any]:
    return get_whisper_status()


# =====================================================================
# YouTube Transcription Endpoint
# =====================================================================
@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe_video(payload: TranscribeRequest) -> TranscribeResponse:
    video_id = extract_video_id(payload.url)

    try:
        result = fetch_youtube_transcript(video_id, languages=["id", "en"])
    except InvalidVideoId as exc:
        raise HTTPException(status_code=400, detail="ID video YouTube tidak valid.") from exc
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

    session_id: str | None = None
    if payload.save:
        database.init_db()
        saved = database.create_session(
            title=f"YouTube: {video_id}",
            source="youtube",
            source_ref=video_id,
            language=result["language"],
            language_code=result["language_code"],
            is_generated=result["is_generated"],
            transcript=full_paragraph,
            timestamps=[t.model_dump() for t in timestamps],
        )
        session_id = saved["id"]

    return TranscribeResponse(
        session_id=session_id,
        video_id=video_id,
        language=result["language"],
        language_code=result["language_code"],
        is_generated=result["is_generated"],
        transcript=full_paragraph,
        timestamps=timestamps,
    )


# =====================================================================
# Audio Transcription (HTTP + Dual-Channel Speaker Separation)
# =====================================================================
@app.post("/transcribe-audio", response_model=AudioTranscribeResponse)
def transcribe_audio(
    file: Annotated[UploadFile, File(description="File audio WAV PCM 16 kHz (mono atau stereo dual-speaker)")],
    language: Annotated[str, Form()] = "id",
    model_size: Annotated[str, Form()] = "small",
    offset_seconds: Annotated[float, Form()] = 0.0,
    speaker_mode: Annotated[str, Form()] = "auto",
) -> AudioTranscribeResponse:
    if not WHISPER_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail=f"Library faster-whisper belum tersedia: {WHISPER_IMPORT_ERROR}",
        )

    language = language.strip().lower()
    model_size = model_size.strip().lower()
    speaker_mode = speaker_mode.strip().lower()

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
    if speaker_mode not in {"auto", "mono", "dual", "remote_only"}:
        raise HTTPException(status_code=400, detail="speaker_mode tidak valid.")

    wav_bytes = file.file.read(MAX_AUDIO_UPLOAD_BYTES + 1)
    if not wav_bytes:
        raise HTTPException(status_code=400, detail="File audio kosong.")
    if len(wav_bytes) > MAX_AUDIO_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Ukuran file audio melebihi batas 100 MB.")

    try:
        result = transcribe_wav_bytes(
            wav_bytes,
            language=language,
            model_size=model_size,
            offset_seconds=offset_seconds,
            speaker_mode=speaker_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mentranskrip audio dengan faster-whisper: {exc}",
        ) from exc

    return AudioTranscribeResponse(**result)


# =====================================================================
# Real-Time WebSocket Streaming Transcription (/ws/transcribe)
# =====================================================================
@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket) -> None:
    """
    Protokol streaming audio Live Meeting:
    1. Client kirim JSON metadata:
       {"type": "chunk", "language": "id", "model_size": "small", "offset_seconds": 8.0, "speaker_mode": "dual"}
    2. Client kirim binary frame WAV PCM 16 kHz.
    3. Server balas JSON {"type": "result", ...} atau {"type": "error", "detail": "..."}.
    """
    await websocket.accept()
    config: dict[str, Any] = {
        "language": "id",
        "model_size": "small",
        "offset_seconds": 0.0,
        "speaker_mode": "auto",
    }

    await websocket.send_json(
        {
            "type": "ready",
            "whisper_available": WHISPER_AVAILABLE,
        }
    )

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if "text" in message and message["text"] is not None:
                import json

                try:
                    meta = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "detail": "Payload JSON tidak valid."})
                    continue

                if meta.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue

                if "language" in meta and str(meta["language"]).lower() in ALLOWED_LANGUAGES:
                    config["language"] = str(meta["language"]).lower()
                if "model_size" in meta and str(meta["model_size"]).lower() in ALLOWED_WHISPER_MODELS:
                    config["model_size"] = str(meta["model_size"]).lower()
                if "offset_seconds" in meta:
                    config["offset_seconds"] = max(0.0, float(meta["offset_seconds"]))
                if "speaker_mode" in meta and str(meta["speaker_mode"]).lower() in {
                    "auto",
                    "mono",
                    "dual",
                    "remote_only",
                }:
                    config["speaker_mode"] = str(meta["speaker_mode"]).lower()

            elif "bytes" in message and message["bytes"] is not None:
                wav_bytes = message["bytes"]
                if not wav_bytes:
                    continue
                if len(wav_bytes) > MAX_AUDIO_UPLOAD_BYTES:
                    await websocket.send_json(
                        {"type": "error", "detail": "Ukuran segmen audio melebihi batas."}
                    )
                    continue

                try:
                    result = await asyncio.to_thread(
                        transcribe_wav_bytes,
                        wav_bytes,
                        language=config["language"],
                        model_size=config["model_size"],
                        offset_seconds=config["offset_seconds"],
                        speaker_mode=config["speaker_mode"],
                    )
                    await websocket.send_json({"type": "result", **result})
                except Exception as exc:
                    await websocket.send_json(
                        {"type": "error", "detail": f"Gagal memproses segmen audio: {exc}"}
                    )
    except WebSocketDisconnect:
        return


# =====================================================================
# SQLite Session Workspace CRUD (/sessions)
# =====================================================================
@app.get("/sessions")
def list_saved_sessions(
    q: Annotated[str | None, Query(description="Kata kunci pencarian lintas transkrip")] = None,
    source: Annotated[str | None, Query(description="Filter: youtube, meeting, file")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict[str, Any]:
    database.init_db()
    items = database.list_sessions(q=q, source=source, limit=limit)
    return {"items": items, "total": len(items)}


@app.post("/sessions", status_code=201)
def create_saved_session(payload: SessionCreateRequest) -> dict[str, Any]:
    database.init_db()
    return database.create_session(
        title=payload.title,
        source=payload.source,
        source_ref=payload.source_ref,
        language=payload.language,
        language_code=payload.language_code,
        model_size=payload.model_size,
        device=payload.device,
        is_generated=payload.is_generated,
        has_speakers=payload.has_speakers,
        transcript=payload.transcript,
        timestamps=[t.model_dump() for t in payload.timestamps],
    )


@app.get("/sessions/{session_id}")
def get_saved_session(session_id: str) -> dict[str, Any]:
    database.init_db()
    session = database.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Riwayat transkrip tidak ditemukan.")
    return session


@app.patch("/sessions/{session_id}")
def update_saved_session(session_id: str, payload: SessionUpdateRequest) -> dict[str, Any]:
    database.init_db()
    updated = database.update_session(
        session_id,
        title=payload.title,
        transcript=payload.transcript,
        timestamps=[t.model_dump() for t in payload.timestamps]
        if payload.timestamps is not None
        else None,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Riwayat transkrip tidak ditemukan.")
    return updated


@app.delete("/sessions/{session_id}")
def delete_saved_session(session_id: str) -> dict[str, Any]:
    database.init_db()
    deleted = database.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Riwayat transkrip tidak ditemukan.")
    return {"deleted": True, "id": session_id}


@app.post("/sessions/{session_id}/audio")
def upload_session_audio(
    session_id: str,
    file: Annotated[UploadFile, File(description="Rekaman audio WAV sesi")],
) -> dict[str, Any]:
    database.init_db()
    wav_bytes = file.file.read(MAX_AUDIO_UPLOAD_BYTES + 1)
    if not wav_bytes:
        raise HTTPException(status_code=400, detail="File audio kosong.")
    if len(wav_bytes) > MAX_AUDIO_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Ukuran audio melebihi batas 100 MB.")

    updated = database.save_session_audio(session_id, wav_bytes)
    if updated is None:
        raise HTTPException(status_code=404, detail="Riwayat transkrip tidak ditemukan.")
    return updated


@app.get("/sessions/{session_id}/audio")
def get_session_audio(session_id: str) -> FileResponse:
    database.init_db()
    audio_path = database.get_session_audio_path(session_id)
    if audio_path is None:
        raise HTTPException(status_code=404, detail="Audio untuk sesi ini tidak tersedia.")
    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=f"voxscribe-{session_id}.wav",
    )


# =====================================================================
# Multi-Format Export Endpoints (.DOCX, .SRT, .MD)
# =====================================================================
@app.post("/export/docx")
def export_docx(payload: ExportRequest) -> Response:
    subtitle = (
        f"Link: https://www.youtube.com/watch?v={payload.source_ref}"
        if payload.source == "youtube" and payload.source_ref
        else f"Sumber: {payload.source.upper()} ({payload.source_ref or '-'})"
        + (f"  |  Model: faster-whisper ({payload.model_size})" if payload.model_size else "")
    )
    mode_label = "Dengan Timestamp" if payload.mode == "timestamps" else "Paragraf Bersih"
    meta_line = f"Bahasa: {payload.language} ({payload.language_code})  |  Mode: {mode_label}"

    docx_bytes = build_docx_bytes(
        title=payload.title,
        subtitle=subtitle,
        meta_line=meta_line,
        transcript=payload.transcript,
        timestamps=[t.model_dump() for t in payload.timestamps],
        mode=payload.mode,
    )
    fname = _safe_filename(payload.title, "voxscribe")
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{fname}.docx"'},
    )


@app.post("/export/srt")
def export_srt(payload: ExportRequest) -> Response:
    srt_text = build_srt([t.model_dump() for t in payload.timestamps])
    fname = _safe_filename(payload.title, "voxscribe")
    return Response(
        content=srt_text.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}.srt"'},
    )


@app.post("/export/md")
def export_markdown(payload: ExportRequest) -> Response:
    md_text = build_markdown(
        title=payload.title,
        source=payload.source,
        source_ref=payload.source_ref,
        language=payload.language,
        language_code=payload.language_code,
        transcript=payload.transcript,
        timestamps=[t.model_dump() for t in payload.timestamps],
    )
    fname = _safe_filename(payload.title, "voxscribe")
    return Response(
        content=md_text.encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}.md"'},
    )
