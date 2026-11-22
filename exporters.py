from datetime import datetime
import io
from typing import Any
from xml.sax.saxutils import escape
import zipfile


def format_srt_timestamp(seconds: float) -> str:
    """Format detik menjadi timestamp SubRip standar: HH:MM:SS,mmm."""
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def build_srt(timestamps: list[dict[str, Any]]) -> str:
    """Membangun file subtitle .SRT siap pakai untuk CapCut, Premiere, atau YouTube."""
    blocks: list[str] = []
    for idx, item in enumerate(timestamps, start=1):
        start = float(item.get("start", 0.0))
        duration = float(item.get("duration", 0.0))
        if duration <= 0.0:
            next_start = (
                float(timestamps[idx].get("start", start + 3.0))
                if idx < len(timestamps)
                else start + 3.0
            )
            end = max(start + 0.5, min(next_start, start + 6.0))
        else:
            end = start + duration

        speaker = item.get("speaker")
        text = str(item.get("text", "")).strip()
        line = f"{speaker}: {text}" if speaker else text

        blocks.append(
            f"{idx}\n"
            f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n"
            f"{line}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def build_markdown(
    *,
    title: str,
    source: str,
    source_ref: str,
    language: str,
    language_code: str,
    transcript: str,
    timestamps: list[dict[str, Any]],
) -> str:
    """Membangun dokumen Markdown rapi untuk Notion / Obsidian."""
    exported_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# {title}",
        "",
        f"- **Sumber:** `{source}` ({source_ref or '-'})",
        f"- **Bahasa:** {language} (`{language_code}`)",
        f"- **Diekspor:** {exported_at}",
        "",
        "## Paragraf Bersih",
        "",
        transcript.strip() or "_(Kosong)_",
        "",
        "## Transkrip Dengan Timestamp",
        "",
    ]

    for item in timestamps:
        ts = item.get("timestamp", "00:00")
        speaker = item.get("speaker")
        text = str(item.get("text", "")).strip()
        if speaker:
            lines.append(f"- **`[{ts}]` {speaker}:** {text}")
        else:
            lines.append(f"- **`[{ts}]`** {text}")

    lines.append("")
    return "\n".join(lines)


def _docx_paragraph(runs_xml: str, space_after: int = 120, border_bottom: bool = False) -> str:
    border_xml = (
        '<w:pBdr><w:bottom w:val="single" w:sz="6" w:space="4" w:color="CBD5E1"/></w:pBdr>'
        if border_bottom
        else ""
    )
    return (
        f"<w:p>"
        f'<w:pPr><w:spacing w:after="{space_after}" w:line="276" w:lineRule="auto"/>{border_xml}</w:pPr>'
        f"{runs_xml}"
        f"</w:p>"
    )


def _docx_run(
    text: str,
    *,
    bold: bool = False,
    size_half_pt: int = 22,
    color: str = "1E293B",
) -> str:
    b_tag = "<w:b/>" if bold else ""
    safe_text = escape(text)
    return (
        "<w:r>"
        f'<w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>{b_tag}'
        f'<w:color w:val="{color}"/><w:sz w:val="{size_half_pt}"/></w:rPr>'
        f'<w:t xml:space="preserve">{safe_text}</w:t>'
        "</w:r>"
    )


def build_docx_bytes(
    *,
    title: str,
    subtitle: str,
    meta_line: str,
    transcript: str,
    timestamps: list[dict[str, Any]],
    mode: str = "timestamps",
) -> bytes:
    """
    Menghasilkan file Microsoft Word (.docx) standar OpenXML tanpa dependency eksternal.
    """
    body_parts: list[str] = [
        _docx_paragraph(_docx_run(title, bold=True, size_half_pt=32, color="0F172A"), space_after=80),
        _docx_paragraph(_docx_run(subtitle, size_half_pt=19, color="64748B"), space_after=40),
        _docx_paragraph(
            _docx_run(meta_line, size_half_pt=19, color="64748B"),
            space_after=200,
            border_bottom=True,
        ),
    ]

    if mode == "paragraph":
        for para in (transcript or "").splitlines():
            clean = para.strip()
            if clean:
                body_parts.append(_docx_paragraph(_docx_run(clean, size_half_pt=22), space_after=140))
    else:
        for item in timestamps:
            ts = str(item.get("timestamp", "00:00"))
            speaker = item.get("speaker")
            text = str(item.get("text", "")).strip()
            runs = _docx_run(f"[{ts}] ", bold=True, size_half_pt=20, color="DC2626")
            if speaker:
                spk_color = "2563EB" if speaker == "Anda" else "059669"
                runs += _docx_run(f"{speaker}: ", bold=True, size_half_pt=21, color=spk_color)
            runs += _docx_run(text, size_half_pt=22, color="1E293B")
            body_parts.append(_docx_paragraph(runs, space_after=100))

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + "".join(body_parts)
        + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/></w:sectPr>'
        "</w:body></w:document>"
    )

    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )

    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("word/document.xml", document_xml)
    return buf.getvalue()
