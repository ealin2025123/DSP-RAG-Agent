import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from .loaders import LoaderFactory
from .security import SecurityAgent


class IngestionPipeline:
    def __init__(self, project_root, security_agent):
        self.root = Path(project_root)
        self.security = security_agent
        self.pending_dir = self.root / "data" / "imports" / "pending"
        self.pending_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, source_path, pages=None):
        source_path = Path(source_path).resolve()
        parsed = LoaderFactory.load(source_path, pages=pages)
        raw_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        doc_id = "import-{}".format(raw_hash[:16])
        sanitized, findings = self.security.sanitize(parsed.text)
        metadata = {
            "doc_id": doc_id,
            "title": "待审核导入文档 {}".format(raw_hash[:8]),
            "category": "待分类",
            "topic": "待分类",
            "source_type": parsed.source_type,
            "source_refs": ["IMPORT-{}".format(raw_hash[:12].upper())],
            "sensitivity": "sanitized_pending_review",
            "status": "pending_review",
            "security_findings": findings,
            "source_sha256": raw_hash,
            "extracted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "loader_metadata": parsed.metadata,
        }
        front = ["---"]
        for key in ("doc_id", "title", "category", "topic", "source_type", "sensitivity", "status"):
            front.append("{}: {}".format(key, metadata[key]))
        front.append("source_refs: [{}]".format(", ".join(metadata["source_refs"])))
        front.extend(["---", "", "# {}".format(metadata["title"]), "", sanitized.strip(), ""])
        output = self.pending_dir / (doc_id + ".md")
        output.write_text("\n".join(front), encoding="utf-8")
        audit = self.pending_dir / (doc_id + ".audit.json")
        audit.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return output, audit, metadata

    def list_pending(self):
        return sorted(self.pending_dir.glob("*.md"))

    def approve(self, pending_path, title, category, topic, confirmed=False):
        if not confirmed:
            raise ValueError("批准入库必须显式确认已人工检查客户名、账号数据和内容准确性")
        pending_path = Path(pending_path).resolve()
        pending_root = self.pending_dir.resolve()
        if pending_path.parent != pending_root or pending_path.suffix.lower() != ".md":
            raise ValueError("只能批准 data/imports/pending 中的 Markdown")
        if not pending_path.exists():
            raise ValueError("待审核文档不存在")
        for label, value in (("title", title), ("category", category), ("topic", topic)):
            if not str(value).strip() or "\n" in str(value) or "\r" in str(value):
                raise ValueError("{} 不能为空或包含换行".format(label))

        raw = pending_path.read_text(encoding="utf-8")
        if not raw.startswith("---\n") or "\n---\n" not in raw[4:]:
            raise ValueError("待审核文档缺少有效 front matter")
        front_end = raw.find("\n---\n", 4)
        metadata = {}
        for line in raw[4:front_end].splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()
        doc_id = metadata.get("doc_id", pending_path.stem)
        if not re.match(r"^import-[a-f0-9]{16}$", doc_id):
            raise ValueError("导入文档 ID 不合法")
        body = raw[front_end + 5:].strip()
        old_title = "# " + metadata.get("title", "")
        if body.startswith(old_title):
            body = body[len(old_title):].lstrip()
        body, findings = self.security.sanitize(body)
        if findings:
            raise ValueError("批准前仍检测到敏感字段: {}".format(", ".join(findings)))

        documents_dir = self.root / "data" / "private_knowledge_base" / "documents"
        documents_dir.mkdir(parents=True, exist_ok=True)
        target = documents_dir / (doc_id.replace("-", "_") + ".md")
        if target.exists():
            raise ValueError("正式知识库中已存在同源文档")
        source_ref = metadata.get("source_refs", "[IMPORT-REVIEWED]")
        reviewed = "\n".join([
            "---", "doc_id: {}".format(doc_id), "title: {}".format(title.strip()),
            "category: {}".format(category.strip()), "topic: {}".format(topic.strip()),
            "source_refs: {}".format(source_ref), "sensitivity: sanitized", "status: reviewed",
            "---", "", "# {}".format(title.strip()), "", body, "",
        ])
        target.write_text(reviewed, encoding="utf-8")
        return target

    def finalize_approval(self, pending_path, target):
        pending_path = Path(pending_path).resolve()
        audit = pending_path.with_suffix(".audit.json")
        approved_dir = self.root / "data" / "imports" / "approved"
        approved_dir.mkdir(parents=True, exist_ok=True)
        if audit.exists():
            record = json.loads(audit.read_text(encoding="utf-8"))
            record["status"] = "approved"
            record["approved_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            record["knowledge_document"] = str(Path(target).relative_to(self.root)).replace("\\", "/")
            approved_audit = approved_dir / audit.name
            approved_audit.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            audit.unlink()
        pending_path.unlink()
