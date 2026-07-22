import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT / "vendor"))
from lyy_rag_agent import DSPRAGAgent  # noqa: E402
from lyy_rag_agent.ingestion import IngestionPipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Manage DSP RAG knowledge indexes")
    sub = parser.add_subparsers(dest="command")
    build = sub.add_parser("build-index", help="Build missing text-embedding-v4 document vectors")
    build.add_argument("--force", action="store_true", help="Rebuild every vector")
    sub.add_parser("status", help="Show index and provider readiness without exposing keys")
    extract = sub.add_parser("extract", help="Extract and sanitize a file into the pending-review area")
    extract.add_argument("source", help="TXT, MD, PDF, DOCX, XLSX, or PPTX file")
    extract.add_argument("--pages", help="PDF pages, e.g. 2-9 or 2,4,6-9")
    sub.add_parser("list-pending", help="List sanitized documents awaiting human review")
    approve = sub.add_parser("approve", help="Approve a reviewed pending Markdown and rebuild indexes")
    approve.add_argument("pending", help="Path returned by the extract command")
    approve.add_argument("--title", required=True)
    approve.add_argument("--category", required=True)
    approve.add_argument("--topic", required=True)
    approve.add_argument("--confirmed", action="store_true", help="Confirm manual privacy and accuracy review")
    args = parser.parse_args()

    agent = DSPRAGAgent()
    pipeline = IngestionPipeline(ROOT, agent.security)
    if args.command == "build-index":
        count = agent.embedding_index.build(force=args.force)
        print("Embedded {} chunks; index ready={}".format(count, agent.embedding_index.ready))
        return
    if args.command == "status":
        status = {
            "document_vectors": len(agent.embedding_index.vectors),
            "document_chunks": len(agent.embedding_index.chunks),
            "embedding_ready": agent.embedding_index.ready,
            "embedding_endpoint_ready": agent.embedding_index.client.available,
            "rerank_endpoint_ready": agent.reranker.available,
            "qwen_ready": agent.providers["qwen"].available,
            "deepseek_ready": agent.providers["deepseek"].available,
            "offline_forced": os.getenv("LYY_OFFLINE", "").lower() in ("1", "true", "yes", "on"),
            "knowledge_mode": agent.knowledge_mode,
            "knowledge_path": str(agent.knowledge_path),
        }
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return
    if args.command == "extract":
        output, audit, metadata = pipeline.extract(args.source, pages=args.pages)
        print("Pending review: {}".format(output))
        print("Audit record: {}".format(audit))
        print("Security findings: {}".format(", ".join(metadata["security_findings"]) or "none"))
        print("This document has NOT been added to the live knowledge base.")
        return
    if args.command == "list-pending":
        pending = pipeline.list_pending()
        print("\n".join(str(path) for path in pending) if pending else "No pending documents")
        return
    if args.command == "approve":
        target = pipeline.approve(
            args.pending, args.title, args.category, args.topic, confirmed=args.confirmed
        )
        try:
            subprocess.run([sys.executable, str(ROOT / "scripts" / "build_dsp_rag_exports.py")], check=True)
            fresh_agent = DSPRAGAgent()
            count = fresh_agent.embedding_index.build()
        except Exception:
            if target.exists():
                target.unlink()
            raise
        pipeline.finalize_approval(args.pending, target)
        print("Approved knowledge document: {}".format(target))
        print("Embedded {} new chunks. Restart a running Web service to reload the corpus.".format(count))
        return
    parser.print_help()


if __name__ == "__main__":
    main()
