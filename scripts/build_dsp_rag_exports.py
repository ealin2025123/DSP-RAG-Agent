"""Build RAG-ready exports from the sanitized DSP Markdown documents.

This script never reads the original internship files. It only processes the
reviewed, sanitized Markdown files under data/private_knowledge_base/documents.
"""

from __future__ import print_function

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "private_knowledge_base"
DOCUMENTS_DIR = DATASET_ROOT / "documents"
EXPORTS_DIR = DATASET_ROOT / "exports"
PRIVATE_TERMS_PATH = PROJECT_ROOT / "config" / "private_terms.txt"

MAX_CHARS = 1000

FORBIDDEN_PATTERNS = {
    "url": re.compile(r"https?://|www\.", re.IGNORECASE),
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "asin": re.compile(r"\bB0[A-Z0-9]{8}\b", re.IGNORECASE),
    "phone": re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"),
}


def load_private_terms():
    if not PRIVATE_TERMS_PATH.exists():
        return []
    return [
        line.strip()
        for line in PRIVATE_TERMS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def parse_front_matter(text):
    metadata = {}
    body = text
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            raw = text[4:end]
            body = text[end + 5 :].strip()
            for line in raw.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                value = value.strip()
                if value.startswith("[") and value.endswith("]"):
                    value = [item.strip() for item in value[1:-1].split(",") if item.strip()]
                metadata[key.strip()] = value
    return metadata, body


def split_sections(body):
    """Split on Markdown headings while retaining the heading in each section."""
    sections = []
    current_heading = ""
    current_lines = []
    for line in body.splitlines():
        if re.match(r"^#{1,3}\s+", line):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = re.sub(r"^#{1,3}\s+", "", line).strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))
    return [(heading, text) for heading, text in sections if text]


def split_long_section(text, max_chars=MAX_CHARS):
    """Split long prose on paragraph boundaries; keep Markdown tables intact."""
    if len(text) <= max_chars:
        return [text]

    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    chunks = []
    current = []
    current_len = 0
    for block in blocks:
        extra = len(block) + (2 if current else 0)
        if current and current_len + extra > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(block)
        current_len += len(block) + (2 if current_len else 0)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def safety_scan(text):
    findings = []
    for name, pattern in FORBIDDEN_PATTERNS.items():
        if pattern.search(text):
            findings.append(name)
    private_terms = load_private_terms()
    if private_terms and re.search(
        "|".join(re.escape(term) for term in private_terms), text, re.IGNORECASE
    ):
        findings.append("private_term")
    return findings


def build():
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    documents = []

    for path in sorted(DOCUMENTS_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        findings = safety_scan(raw)
        if findings:
            raise ValueError("Safety scan failed for {}: {}".format(path.name, ", ".join(findings)))

        metadata, body = parse_front_matter(raw)
        doc_id = metadata.get("doc_id", path.stem)
        doc_record = {
            "doc_id": doc_id,
            "title": metadata.get("title", path.stem),
            "category": metadata.get("category", "未分类"),
            "topic": metadata.get("topic", "未分类"),
            "source_refs": metadata.get("source_refs", []),
            "path": str(path.relative_to(DATASET_ROOT)).replace("\\", "/"),
            "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        }
        documents.append(doc_record)

        chunk_index = 0
        for heading, section in split_sections(body):
            # The document H1 is already present in metadata. Avoid creating a
            # tiny title-only vector that adds no retrieval value.
            if re.match(r"^#{1,3}\s+[^\n]+$", section.strip()):
                continue
            for chunk_text in split_long_section(section):
                chunk_index += 1
                record = {
                    "id": "{}-{:03d}".format(doc_id, chunk_index),
                    "text": chunk_text,
                    "metadata": {
                        "doc_id": doc_id,
                        "title": doc_record["title"],
                        "heading": heading or doc_record["title"],
                        "category": doc_record["category"],
                        "topic": doc_record["topic"],
                        "source_refs": doc_record["source_refs"],
                        "sensitivity": "sanitized",
                        "status": metadata.get("status", "reviewed"),
                        "source_path": doc_record["path"],
                    },
                }
                records.append(record)

    jsonl_path = EXPORTS_DIR / "documents.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    corpus_path = EXPORTS_DIR / "rag_corpus.txt"
    with corpus_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            meta = record["metadata"]
            handle.write(
                "[CHUNK {}]\n标题：{}\n章节：{}\n分类：{}\n主题：{}\n{}\n[/CHUNK]\n\n".format(
                    record["id"],
                    meta["title"],
                    meta["heading"],
                    meta["category"],
                    meta["topic"],
                    record["text"],
                )
            )

    manifest = {
        "dataset": "sanitized_dsp_configuration_knowledge_base",
        "version": "1.0.0",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "language": "zh-CN",
        "sensitivity": "sanitized",
        "document_count": len(documents),
        "chunk_count": len(records),
        "recommended_chunking": {
            "strategy": "markdown_heading_then_paragraph",
            "max_chars": MAX_CHARS,
            "overlap_chars": 0,
            "table_policy": "keep_table_with_header",
        },
        "documents": documents,
        "exports": [
            "exports/documents.jsonl",
            "exports/rag_corpus.txt",
        ],
    }
    manifest_path = DATASET_ROOT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    exported_text = jsonl_path.read_text(encoding="utf-8") + corpus_path.read_text(encoding="utf-8")
    findings = safety_scan(exported_text)
    if findings:
        raise ValueError("Export safety scan failed: {}".format(", ".join(findings)))

    print(
        "Built {} documents and {} chunks in {}".format(
            len(documents), len(records), EXPORTS_DIR
        )
    )


if __name__ == "__main__":
    build()
