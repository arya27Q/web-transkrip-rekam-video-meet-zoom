import importlib.util
import io
import os
import re
import sys
import threading
import types
from typing import Any
import wave

import numpy as np

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


def get_whisper_status() -> dict[str, Any]:
    return {
        "available": WHISPER_AVAILABLE,
        "error": WHISPER_IMPORT_ERROR,
        "loaded_model": _whisper_state["model_size"],
        "device": _whisper_state["device"],
        "compute_type": _whisper_state["compute_type"],
        "supported_models": sorted(ALLOWED_WHISPER_MODELS),
    }


def _resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or len(samples) <= 1:
        return np.ascontiguousarray(samples, dtype=np.float32)
    target_len = max(1, int(round(len(samples) * dst_rate / src_rate)))
    old_idx = np.linspace(0.0, len(samples) - 1, num=len(samples), dtype=np.float64)
    new_idx = np.linspace(0.0, len(samples) - 1, num=target_len, dtype=np.float64)
    return np.ascontiguousarray(np.interp(new_idx, old_idx, samples), dtype=np.float32)


def decode_wav_channels_16k(wav_bytes: bytes) -> tuple[list[np.ndarray], float]:
    """
    Mendekode WAV PCM menjadi daftar channel float32 16 kHz tanpa FFmpeg eksternal.
    - 1 channel -> [mono]
    - 2 channel -> [left_channel, right_channel] (mendukung Dual-Channel Speaker Separation)
    - >2 channel -> [mono_mix]
    """
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frame_count = wf.getnframes()
            raw_frames = wf.readframes(frame_count)
    except wave.Error as exc:
        raise ValueError(f"Format audio WAV tidak valid: {exc}") from exc

    if frame_count == 0 or not raw_frames:
        return [np.zeros(0, dtype=np.float32)], 0.0

    if sample_width == 2:
        samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw_frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        samples = (np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(
            f"Bit depth WAV ({sample_width * 8}-bit) tidak didukung. Gunakan PCM 16-bit."
        )

    usable_len = (len(samples) // channels) * channels
    samples = samples[:usable_len]
    duration_sec = (usable_len / channels) / float(sample_rate) if sample_rate > 0 else 0.0

    if channels == 1:
        return [_resample_linear(samples, sample_rate, TARGET_SAMPLE_RATE)], duration_sec

    reshaped = samples.reshape(-1, channels)
    if channels == 2:
        left = _resample_linear(reshaped[:, 0], sample_rate, TARGET_SAMPLE_RATE)
        right = _resample_linear(reshaped[:, 1], sample_rate, TARGET_SAMPLE_RATE)
        return [left, right], duration_sec

    mono = reshaped.mean(axis=1)
    return [_resample_linear(mono, sample_rate, TARGET_SAMPLE_RATE)], duration_sec


def decode_wav_to_float32_16k(wav_bytes: bytes) -> tuple[np.ndarray, float]:
    """Helper mono 16 kHz float32 (menggabungkan stereo bila ada)."""
    channels, duration_sec = decode_wav_channels_16k(wav_bytes)
    if len(channels) == 1:
        return channels[0], duration_sec
    mono = np.mean(np.stack(channels, axis=0), axis=0, dtype=np.float32)
    return np.ascontiguousarray(mono, dtype=np.float32), duration_sec


def preprocess_speech_audio(audio: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Membersihkan audio sebelum masuk ke Whisper:
    - Hapus DC offset
    - Energy-based VAD untuk melewati bagian hening & memangkas hening awal/akhir
    - Normalisasi volume lembut (target peak 0.75, max gain 5x)
    """
    if audio.size < int(TARGET_SAMPLE_RATE * 0.2):
        return np.zeros(0, dtype=np.float32), 0.0

    audio = audio - float(np.mean(audio))
    overall_rms = float(np.sqrt(np.mean(audio * audio)))
    if overall_rms < 0.0025:
        return np.zeros(0, dtype=np.float32), 0.0

    frame_size = int(TARGET_SAMPLE_RATE * 0.03)
    num_frames = len(audio) // frame_size
    leading_trim_sec = 0.0

    if num_frames > 0:
        framed = audio[: num_frames * frame_size].reshape(num_frames, frame_size)
        frame_rms = np.sqrt(np.mean(framed * framed, axis=1))
        # Batasi noise_floor agar segmen yang 100% berisi ucapan kontinu tidak dianggap noise
        noise_floor = min(float(np.percentile(frame_rms, 15)), 0.015)
        speech_threshold = max(0.0045, noise_floor * 2.0)

        active = np.where(frame_rms >= speech_threshold)[0]
        if active.size == 0:
            return np.zeros(0, dtype=np.float32), 0.0

        pad_frames = 8  # ~240ms padding
        start_frame = max(0, int(active[0]) - pad_frames)
        end_frame = min(num_frames, int(active[-1]) + pad_frames + 1)

        start_sample = start_frame * frame_size
        end_sample = min(len(audio), end_frame * frame_size)
        audio = audio[start_sample:end_sample]
        leading_trim_sec = start_sample / float(TARGET_SAMPLE_RATE)

    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if peak > 1e-4:
        gain = min(0.75 / peak, 5.0)
        audio = np.clip(audio * gain, -1.0, 1.0)

    return np.ascontiguousarray(audio, dtype=np.float32), leading_trim_sec


def _load_whisper_model_unlocked(model_size: str, force_cpu: bool = False) -> tuple[Any, str, str]:
    if not WHISPER_AVAILABLE or WhisperModel is None:
        raise RuntimeError(f"faster-whisper belum siap di server: {WHISPER_IMPORT_ERROR}")

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


def _transcribe_single_waveform(
    audio_np: np.ndarray,
    language: str,
    model_size: str,
    leading_trim_sec: float,
    offset_seconds: float,
    speaker: str | None = None,
) -> tuple[list[dict[str, Any]], str, str, str]:
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
            if device == "cuda":
                model, device, _ = _load_whisper_model_unlocked(model_size, force_cpu=True)
                raw_segments, info = _do_transcribe(model)
            else:
                raise exc

    items: list[dict[str, Any]] = []
    for seg in raw_segments:
        if getattr(seg, "no_speech_prob", 0.0) > 0.78:
            continue
        cleaned = clean_snippet_text(str(seg.text))
        if not cleaned:
            continue

        abs_start = max(0.0, offset_seconds + leading_trim_sec + float(seg.start))
        duration = max(0.0, float(seg.end) - float(seg.start))
        items.append(
            {
                "timestamp": format_timestamp(abs_start),
                "start": round(abs_start, 2),
                "duration": round(duration, 2),
                "text": cleaned,
                "speaker": speaker,
            }
        )

    detected_code = getattr(info, "language", None) or (language if language != "auto" else "id")
    detected_name = LANGUAGE_NAMES.get(detected_code, detected_code.upper())
    return items, detected_code, detected_name, device


def transcribe_wav_bytes(
    wav_bytes: bytes,
    *,
    language: str = "id",
    model_size: str = "small",
    offset_seconds: float = 0.0,
    speaker_mode: str = "auto",
) -> dict[str, Any]:
    """
    Transkrip WAV bytes dengan dukungan:
    - speaker_mode='dual' + stereo WAV: Ch0 (Kiri) = 'Anda' (Mic), Ch1 (Kanan) = 'Peserta' (Tab/Sistem)
    - speaker_mode='remote_only': mono -> dilabeli 'Peserta'
    - speaker_mode='mono' / 'auto' (mono WAV): transkrip standar tanpa label pembicara
    """
    channels, duration_sec = decode_wav_channels_16k(wav_bytes)
    safe_offset = max(0.0, float(offset_seconds))

    # Dual-channel speaker separation: kiri = Anda (Mic), kanan = Peserta (Tab/Sistem)
    if speaker_mode == "dual" and len(channels) == 2:
        mic_audio, mic_trim = preprocess_speech_audio(channels[0])
        remote_audio, remote_trim = preprocess_speech_audio(channels[1])

        merged_items: list[dict[str, Any]] = []
        detected_code = language if language != "auto" else "id"
        detected_name = LANGUAGE_NAMES.get(detected_code, detected_code.upper())
        device = _whisper_state["device"] or "cpu"

        if remote_audio.size > 0:
            r_items, detected_code, detected_name, device = _transcribe_single_waveform(
                remote_audio,
                language,
                model_size,
                remote_trim,
                safe_offset,
                speaker="Peserta",
            )
            merged_items.extend(r_items)

        if mic_audio.size > 0:
            m_items, m_code, m_name, device = _transcribe_single_waveform(
                mic_audio,
                language,
                model_size,
                mic_trim,
                safe_offset,
                speaker="Anda",
            )
            if not merged_items:
                detected_code, detected_name = m_code, m_name
            merged_items.extend(m_items)

        merged_items.sort(key=lambda item: item["start"])
        transcript = re.sub(
            r"\s+", " ", " ".join(item["text"] for item in merged_items)
        ).strip()

        return {
            "model_size": model_size,
            "device": device,
            "language": detected_name,
            "language_code": detected_code,
            "duration": round(duration_sec, 2),
            "skipped_silence": len(merged_items) == 0,
            "has_speakers": True,
            "transcript": transcript,
            "timestamps": merged_items,
        }

    # Mode mono standar (atau remote_only)
    if len(channels) > 1:
        mono = np.mean(np.stack(channels, axis=0), axis=0, dtype=np.float32)
    else:
        mono = channels[0]

    clean_audio, leading_trim_sec = preprocess_speech_audio(mono)
    if clean_audio.size == 0:
        lang_code = language if language != "auto" else "id"
        return {
            "model_size": model_size,
            "device": _whisper_state["device"] or "cpu",
            "language": LANGUAGE_NAMES.get(lang_code, lang_code.upper()),
            "language_code": lang_code,
            "duration": round(duration_sec, 2),
            "skipped_silence": True,
            "has_speakers": speaker_mode == "remote_only",
            "transcript": "",
            "timestamps": [],
        }

    speaker_label = "Peserta" if speaker_mode == "remote_only" else None
    items, detected_code, detected_name, device = _transcribe_single_waveform(
        clean_audio,
        language,
        model_size,
        leading_trim_sec,
        safe_offset,
        speaker=speaker_label,
    )
    transcript = re.sub(r"\s+", " ", " ".join(item["text"] for item in items)).strip()

    return {
        "model_size": model_size,
        "device": device,
        "language": detected_name,
        "language_code": detected_code,
        "duration": round(duration_sec, 2),
        "skipped_silence": len(items) == 0,
        "has_speakers": speaker_label is not None,
        "transcript": transcript,
        "timestamps": items,
    }
