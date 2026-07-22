import json
import os
from pathlib import Path


def project_root():
    return Path(__file__).resolve().parents[2]


def load_dotenv(path=None):
    path = Path(path or (project_root() / ".env"))
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_settings(path=None):
    load_dotenv()
    path = Path(path or (project_root() / "config" / "settings.json"))
    settings = json.loads(path.read_text(encoding="utf-8"))
    if os.getenv("QWEN_MODEL"):
        settings["models"]["qwen"]["model"] = os.environ["QWEN_MODEL"]
    if os.getenv("DEEPSEEK_MODEL"):
        settings["models"]["deepseek"]["model"] = os.environ["DEEPSEEK_MODEL"]
    return settings


def dashscope_workspace_id():
    """Accept common workspace variable names without exposing the value."""
    for name in ("DASHSCOPE_WORKSPACE_ID", "workspaceId", "WORKSPACE_ID", "BAILIAN_WORKSPACE_ID"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""
