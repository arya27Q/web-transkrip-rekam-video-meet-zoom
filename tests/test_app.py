import io
from types import SimpleNamespace
from unittest.mock import patch
import wave
import zipfile

from fastapi.testclient import TestClient
import numpy as np
import pytest

import audio_service
import database
import main


def _make_wav_bytes(
    channels_f32: list[np.ndarray],
    sample_rate: int = 16000,
) -> bytes:
    num_channels = len(channels_f32)
    length = min(len(ch) for ch in channels_f32)
    if num_channels == 1:
        pcm = np.clip(channels_f32[0][:length] * 32767.0, -32768, 32767).astype(np.int16)
    else:
        stacked = np.stack([ch[:length] for ch in channels_f32], axis=1)
        pcm = np.clip(stacked * 32767.0, -32768, 32767).astype(np.int16).reshape(-1)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXSCRIBE_DATA_DIR", str(tmp_path))
    database.init_db()
    with TestClient(main.app) as c:
        yield c


def test_health_and_frontend(client: TestClient):
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_home = client.get("/")
    assert r_home.status_code == 200
    assert "VoxScribe" in r_home.text


def test_youtube_regex_and_transcribe_autosave(client: TestClient):
    for url in [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?si=abc",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
    ]:
        assert main.extract_video_id(url) == "dQw4w9WgXcQ"

    bad = client.post("/transcribe", json={"url": "https://example.com/not-yt"})
    assert bad.status_code == 400

    class FakeFetched(list):
        language = "Indonesian"
        language_code = "id"
        is_generated = False

    fake_data = FakeFetched(
        [
            SimpleNamespace(text="Halo semuanya", start=0.0, duration=2.0),
            SimpleNamespace(text="selamat datang di VoxScribe", start=2.0, duration=3.0),
        ]
    )
    with patch.object(main.YouTubeTranscriptApi, "fetch", return_value=fake_data):
        res = client.post(
            "/transcribe",
            json={"url": "https://youtu.be/dQw4w9WgXcQ", "save": True},
        )
    assert res.status_code == 200
    body = res.json()
    assert body["session_id"] is not None
    assert body["transcript"] == "Halo semuanya selamat datang di VoxScribe"

    # Pastikan tersimpan otomatis di SQLite
    listed = client.get("/sessions?source=youtube")
    assert listed.status_code == 200
    assert listed.json()["total"] == 1
    assert listed.json()["items"][0]["id"] == body["session_id"]


def test_dual_channel_speaker_separation(client: TestClient):
    t = np.linspace(0, 1.0, 16000, dtype=np.float32)
    mic_wave = 0.25 * np.sin(2 * np.pi * 440 * t)  # Ch 0 = Anda
    tab_wave = 0.25 * np.sin(2 * np.pi * 660 * t)  # Ch 1 = Peserta
    stereo_wav = _make_wav_bytes([mic_wave, tab_wave], sample_rate=16000)

    call_idx = {"n": 0}

    class FakeWhisper:
        def transcribe(self, audio, **kwargs):
            call_idx["n"] += 1
            if call_idx["n"] == 1:
                seg = SimpleNamespace(start=0.2, end=1.0, text="Halo ini peserta meeting", no_speech_prob=0.02)
            else:
                seg = SimpleNamespace(start=0.5, end=1.0, text="Siap saya dengar jelas", no_speech_prob=0.02)
            return iter([seg]), SimpleNamespace(language="id", language_probability=0.99)

    with patch.object(
        audio_service, "_load_whisper_model_unlocked", return_value=(FakeWhisper(), "cpu", "int8")
    ):
        res = client.post(
            "/transcribe-audio",
            files={"file": ("dual.wav", stereo_wav, "audio/wav")},
            data={
                "language": "id",
                "model_size": "small",
                "offset_seconds": "10",
                "speaker_mode": "dual",
            },
        )

    assert res.status_code == 200
    data = res.json()
    assert data["has_speakers"] is True
    assert len(data["timestamps"]) == 2
    assert data["timestamps"][0]["speaker"] == "Peserta"
    assert data["timestamps"][1]["speaker"] == "Anda"
    assert data["timestamps"][0]["start"] < data["timestamps"][1]["start"]


def test_silent_wav_skips_whisper(client: TestClient):
    silent_wav = _make_wav_bytes([np.zeros(16000, dtype=np.float32)], sample_rate=16000)
    res = client.post(
        "/transcribe-audio",
        files={"file": ("silent.wav", silent_wav, "audio/wav")},
        data={"language": "id", "model_size": "small", "speaker_mode": "mono"},
    )
    assert res.status_code == 200
    assert res.json()["skipped_silence"] is True
    assert res.json()["timestamps"] == []


def test_websocket_streaming_transcribe(client: TestClient):
    t = np.linspace(0, 1.0, 16000, dtype=np.float32)
    wav_bytes = _make_wav_bytes([0.2 * np.sin(2 * np.pi * 500 * t)], sample_rate=16000)

    class FakeWhisper:
        def transcribe(self, audio, **kwargs):
            seg = SimpleNamespace(start=0.1, end=0.9, text="Streaming websocket lancar", no_speech_prob=0.01)
            return iter([seg]), SimpleNamespace(language="id", language_probability=0.95)

    with patch.object(
        audio_service, "_load_whisper_model_unlocked", return_value=(FakeWhisper(), "cpu", "int8")
    ):
        with client.websocket_connect("/ws/transcribe") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "ready"
            ws.send_json(
                {
                    "type": "chunk",
                    "language": "id",
                    "model_size": "small",
                    "offset_seconds": 5.0,
                    "speaker_mode": "remote_only",
                }
            )
            ws.send_bytes(wav_bytes)
            result = ws.receive_json()
            assert result["type"] == "result"
            assert result["transcript"] == "Streaming websocket lancar"
            assert result["timestamps"][0]["speaker"] == "Peserta"


def test_session_crud_audio_and_exports(client: TestClient):
    # Create session
    create_res = client.post(
        "/sessions",
        json={
            "title": "Rapat Sprint Mingguan",
            "source": "meeting",
            "source_ref": "20261122-1800",
            "language": "Indonesian",
            "language_code": "id",
            "model_size": "small",
            "has_speakers": True,
            "transcript": "Kita rilis versi baru minggu depan.",
            "timestamps": [
                {
                    "timestamp": "00:02",
                    "start": 2.0,
                    "duration": 3.5,
                    "text": "Kita rilis versi baru minggu depan.",
                    "speaker": "Anda",
                }
            ],
        },
    )
    assert create_res.status_code == 201
    sid = create_res.json()["id"]

    # Search lintas transkrip
    search_res = client.get("/sessions?q=Sprint")
    assert search_res.status_code == 200
    assert search_res.json()["total"] == 1

    # Update (inline edit + rename)
    patch_res = client.patch(
        f"/sessions/{sid}",
        json={
            "title": "Rapat Sprint Final",
            "transcript": "Kita rilis versi v3 besok pagi.",
        },
    )
    assert patch_res.status_code == 200
    assert patch_res.json()["title"] == "Rapat Sprint Final"
    assert patch_res.json()["word_count"] == 6

    # Upload & retrieve session audio WAV
    wav_bytes = _make_wav_bytes([np.zeros(8000, dtype=np.float32)], sample_rate=16000)
    up_res = client.post(
        f"/sessions/{sid}/audio",
        files={"file": ("meeting.wav", wav_bytes, "audio/wav")},
    )
    assert up_res.status_code == 200
    assert up_res.json()["has_audio"] is True

    dl_audio = client.get(f"/sessions/{sid}/audio")
    assert dl_audio.status_code == 200
    assert dl_audio.content[:4] == b"RIFF"

    # Multi-format exports (.DOCX, .SRT, .MD)
    export_payload = {
        "title": "Rapat Sprint Final",
        "source": "meeting",
        "source_ref": "20261122-1800",
        "language": "Indonesian",
        "language_code": "id",
        "model_size": "small",
        "mode": "timestamps",
        "transcript": "Kita rilis versi v3 besok pagi.",
        "timestamps": [
            {
                "timestamp": "00:02",
                "start": 2.0,
                "duration": 3.5,
                "text": "Kita rilis versi v3 besok pagi.",
                "speaker": "Anda",
            }
        ],
    }

    r_docx = client.post("/export/docx", json=export_payload)
    assert r_docx.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r_docx.content)) as zf:
        assert "word/document.xml" in zf.namelist()
        xml_content = zf.read("word/document.xml").decode("utf-8")
        assert "Rapat Sprint Final" in xml_content
        assert "Anda:" in xml_content

    r_srt = client.post("/export/srt", json=export_payload)
    assert r_srt.status_code == 200
    assert "00:00:02,000 --> 00:00:05,500" in r_srt.text
    assert "Anda: Kita rilis versi v3 besok pagi." in r_srt.text

    r_md = client.post("/export/md", json=export_payload)
    assert r_md.status_code == 200
    assert "# Rapat Sprint Final" in r_md.text

    # Delete session + audio
    del_res = client.delete(f"/sessions/{sid}")
    assert del_res.status_code == 200
    assert client.get(f"/sessions/{sid}").status_code == 404
    assert client.get(f"/sessions/{sid}/audio").status_code == 404
