from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any
import uuid

BASE_DIR = Path(__file__).resolve().parent


def get_data_dir() -> Path:
    custom = os.environ.get("VOXSCRIBE_DATA_DIR")
    return Path(custom) if custom else BASE_DIR / "data"


def get_db_path() -> Path:
    return get_data_dir() / "voxscribe.db"


def get_audio_dir() -> Path:
    return get_data_dir() / "audio"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    get_audio_dir().mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(get_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                source_ref TEXT NOT NULL DEFAULT '',
                language TEXT NOT NULL DEFAULT 'Indonesian',
                language_code TEXT NOT NULL DEFAULT 'id',
                model_size TEXT,
                device TEXT,
                is_generated INTEGER NOT NULL DEFAULT 0,
                has_speakers INTEGER NOT NULL DEFAULT 0,
                transcript TEXT NOT NULL DEFAULT '',
                timestamps_json TEXT NOT NULL DEFAULT '[]',
                word_count INTEGER NOT NULL DEFAULT 0,
                audio_filename TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source)"
        )
        conn.commit()


def _count_words(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w])


def _row_to_dict(row: sqlite3.Row, include_body: bool = True) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": row["id"],
        "title": row["title"],
        "source": row["source"],
        "source_ref": row["source_ref"],
        "language": row["language"],
        "language_code": row["language_code"],
        "model_size": row["model_size"],
        "device": row["device"],
        "is_generated": bool(row["is_generated"]),
        "has_speakers": bool(row["has_speakers"]),
        "word_count": int(row["word_count"]),
        "has_audio": bool(row["audio_filename"]),
        "audio_url": f"/sessions/{row['id']}/audio" if row["audio_filename"] else None,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
    if include_body:
        data["transcript"] = row["transcript"]
        try:
            data["timestamps"] = json.loads(row["timestamps_json"])
        except json.JSONDecodeError:
            data["timestamps"] = []
    else:
        snippet = (row["transcript"] or "").strip()
        data["preview"] = snippet[:140] + ("..." if len(snippet) > 140 else "")
    return data


def create_session(
    *,
    title: str,
    source: str,
    source_ref: str = "",
    language: str = "Indonesian",
    language_code: str = "id",
    model_size: str | None = None,
    device: str | None = None,
    is_generated: bool = False,
    has_speakers: bool = False,
    transcript: str = "",
    timestamps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    session_id = uuid.uuid4().hex[:12]
    now = _utc_now_iso()
    ts_list = timestamps or []
    clean_transcript = transcript.strip()
    if not clean_transcript and ts_list:
        clean_transcript = re.sub(
            r"\s+", " ", " ".join(str(t.get("text", "")).strip() for t in ts_list)
        ).strip()

    detected_speakers = has_speakers or any(bool(t.get("speaker")) for t in ts_list)
    word_count = _count_words(clean_transcript)

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                id, title, source, source_ref, language, language_code,
                model_size, device, is_generated, has_speakers,
                transcript, timestamps_json, word_count, audio_filename,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                title.strip() or "Transkrip Tanpa Judul",
                source,
                source_ref,
                language,
                language_code,
                model_size,
                device,
                1 if is_generated else 0,
                1 if detected_speakers else 0,
                clean_transcript,
                json.dumps(ts_list, ensure_ascii=False),
                word_count,
                None,
                now,
                now,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    assert row is not None
    return _row_to_dict(row, include_body=True)


def list_sessions(
    q: str | None = None,
    source: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []

    if source and source in {"youtube", "meeting", "file"}:
        clauses.append("source = ?")
        params.append(source)

    if q and q.strip():
        like = f"%{q.strip()}%"
        clauses.append("(title LIKE ? OR transcript LIKE ? OR source_ref LIKE ?)")
        params.extend([like, like, like])

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM sessions {where_sql} ORDER BY updated_at DESC LIMIT ?"
    params.append(max(1, min(limit, 500)))

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r, include_body=False) for r in rows]


def get_session(session_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return _row_to_dict(row, include_body=True) if row else None


def update_session(
    session_id: str,
    *,
    title: str | None = None,
    transcript: str | None = None,
    timestamps: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    existing = get_session(session_id)
    if existing is None:
        return None

    new_title = title.strip() if title is not None and title.strip() else existing["title"]
    new_timestamps = timestamps if timestamps is not None else existing["timestamps"]

    if transcript is not None:
        new_transcript = transcript.strip()
    elif timestamps is not None:
        new_transcript = re.sub(
            r"\s+", " ", " ".join(str(t.get("text", "")).strip() for t in new_timestamps)
        ).strip()
    else:
        new_transcript = existing["transcript"]

    has_speakers = any(bool(t.get("speaker")) for t in new_timestamps)
    word_count = _count_words(new_transcript)
    now = _utc_now_iso()

    with _connect() as conn:
        conn.execute(
            """
            UPDATE sessions
            SET title = ?, transcript = ?, timestamps_json = ?,
                has_speakers = ?, word_count = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                new_title,
                new_transcript,
                json.dumps(new_timestamps, ensure_ascii=False),
                1 if has_speakers else 0,
                word_count,
                now,
                session_id,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return _row_to_dict(row, include_body=True) if row else None


def save_session_audio(session_id: str, wav_bytes: bytes) -> dict[str, Any] | None:
    existing = get_session(session_id)
    if existing is None:
        return None

    audio_dir = get_audio_dir()
    audio_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{session_id}.wav"
    (audio_dir / filename).write_bytes(wav_bytes)
    now = _utc_now_iso()

    with _connect() as conn:
        conn.execute(
            "UPDATE sessions SET audio_filename = ?, updated_at = ? WHERE id = ?",
            (filename, now, session_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return _row_to_dict(row, include_body=True) if row else None


def get_session_audio_path(session_id: str) -> Path | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT audio_filename FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    if not row or not row["audio_filename"]:
        return None
    audio_path = get_audio_dir() / row["audio_filename"]
    return audio_path if audio_path.exists() else None


def delete_session(session_id: str) -> bool:
    audio_path = get_session_audio_path(session_id)
    with _connect() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        deleted = cur.rowcount > 0
    if deleted and audio_path is not None:
        audio_path.unlink(missing_ok=True)
    return deleted
