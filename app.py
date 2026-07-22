import argparse
import json
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


def print_response(response, show_trace=False):
    print(response.answer)
    print("\n---")
    print("路由: {} / {} / {}".format(response.route.intent, response.route.complexity, response.provider))
    if show_trace:
        print("Trace: " + json.dumps(response.trace, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="LYY Amazon DSP Agentic RAG")
    parser.add_argument("--question", "-q", help="Ask one question and exit")
    parser.add_argument("--session", default="local-cli", help="Conversation session id")
    parser.add_argument("--trace", action="store_true", help="Show workflow trace")
    args = parser.parse_args()
    agent = DSPRAGAgent()
    if args.question:
        print_response(agent.ask(args.question, args.session), args.trace)
        return
    print("DSP RAG Agent 已启动。无 API Key 时自动使用离线检索模式。输入 exit 退出。")
    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "退出"):
            break
        print("\nAgent：")
        print_response(agent.ask(question, args.session), args.trace)


if __name__ == "__main__":
    main()
