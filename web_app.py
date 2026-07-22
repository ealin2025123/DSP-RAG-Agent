import argparse
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))
from lyy_rag_agent import DSPRAGAgent  # noqa: E402


class AgentHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler):
        super().__init__(address, handler)
        self.agent = DSPRAGAgent()
        self.index_html = (ROOT / "web" / "index.html").read_bytes()


class Handler(BaseHTTPRequestHandler):
    server_version = "LYY-RAG-Agent/0.2"

    def _send(self, status, body, content_type="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, self.server.index_html, "text/html; charset=utf-8")
            return
        if path == "/api/health":
            agent = self.server.agent
            self._send(200, {
                "status": "ok",
                "chunks": len(agent.embedding_index.chunks),
                "vectors": len(agent.embedding_index.vectors),
                "embedding": agent.embedding_index.ready,
                "rerank": agent.reranker.available,
                "qwen": agent.providers["qwen"].available,
                "deepseek": agent.providers["deepseek"].available,
            })
            return
        self._send(404, {"error": "not_found"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/ask":
            self._send(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 2 or length > 32768:
                self._send(413, {"error": "请求体大小不合法"})
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            question = str(payload.get("question", "")).strip()
            session = str(payload.get("session", "web-local"))[:80]
            if not question or len(question) > 4000:
                self._send(400, {"error": "问题不能为空且不能超过 4000 字符"})
                return
            response = self.server.agent.ask(question, session)
            self._send(200, {
                "answer": response.answer,
                "provider": response.provider,
                "intent": response.route.intent,
                "complexity": response.route.complexity,
                "review_passed": response.review_passed,
                "trace": response.trace,
            })
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
        except Exception:
            traceback.print_exc()
            self._send(500, {"error": "Agent 处理失败，请查看本地服务日志"})

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


def main():
    parser = argparse.ArgumentParser(description="Run the local DSP RAG web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = AgentHTTPServer((args.host, args.port), Handler)
    print("DSP RAG Agent: http://{}:{}".format(args.host, args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
