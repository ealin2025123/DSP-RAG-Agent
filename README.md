# DSP-RAG-Agent

面向 DSP 配置学习、操作规范和异常排查场景的开源 Agentic RAG 项目。项目使用
LangChain 1.x 与 LangGraph 构建带反馈循环的状态工作流，并提供混合检索、
可选 Tavily 联网工具、模型路由、回答核验、SQLite 多轮会话和 Streamlit 页面。

> 本仓库不包含作者的学习资料、客户名称、账户数据、原始 PDF/Excel/Word、私有知识库、向量数据库、聊天记录或 API Key。仓库自带的语料完全是虚构 Demo，
> 仅用于验证程序能够运行，不代表任何广告平台的正式规范。

## 功能

- BM25、本地字符向量、DashScope Embedding 和 RRF 混合检索
- 可选 `qwen3-rerank` 语义重排
- 自动、快速和深度三种回答模式
- `qwen-plus`、`qwen3-max` 及低成本回退模型
- LangGraph 检索质量评估、问题改写和有界重试循环
- 知识库不足时可选择 Tavily 搜索 Amazon Ads 官方域名
- Reviewer 不通过时携带反馈重新生成，达到上限后安全结束
- 输入/输出脱敏与回答 Reviewer
- LangGraph StateGraph 条件分支、循环状态和可选 LangSmith Trace
- SQLite 多轮对话持久化
- TXT、Markdown、PDF、DOCX、XLSX、PPTX 提取与人工批准入库
- Streamlit Web UI

## 工作流

```text
问题
  -> 安全检查
  -> 意图与复杂度路由
  -> 混合检索和可选重排
  -> 检索质量评估
       |-- 证据不足且未达上限 -> 改写问题 -> 重新检索
       |-- 要求最新信息或重试后仍不足 -> 可选 Tavily 搜索
       `-- 证据充分 -> 继续
  -> Qwen 生成或离线证据摘录
  -> Reviewer
       |-- 不通过且未达上限 -> 携带审核反馈重新生成
       `-- 通过或达到上限 -> 继续
  -> 输出脱敏
  -> SQLite 会话持久化
```
## 成品展示

- 可打开B站链接看成品展示【广告配置智能问答助手-哔哩哔哩】 https://b23.tv/7yBOTuc

循环次数由 `config/settings.json` 限制，避免无限检索或无限生成。快速模式跳过
远程语义检索和 Reviewer；自动/深度模式才启用完整 Agentic 决策。

## 快速开始

要求 Python 3.10 及以上版本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
.\run_streamlit.ps1
```

打开 <http://127.0.0.1:8501>。不填写 API Key 时，系统使用 Demo 数据和离线
证据摘录；填写百炼 Key 后可启用生成、Embedding 和 Rerank。

如果 Python 环境不在项目 `.venv` 中，可以新建本地文件
`config/local_python_path.txt`，写入 Python 可执行文件的绝对路径。该文件已被
`.gitignore` 排除。

## 环境变量

复制 `.env.example` 为 `.env` 后按需填写：

- `DASHSCOPE_API_KEY`：通义千问、Embedding 和 Rerank
- `DASHSCOPE_WORKSPACE_ID`：百炼 Workspace
- `DEEPSEEK_API_KEY`：可选 DeepSeek 接口
- `TAVILY_API_KEY`：可选 Tavily Search；未配置时自动禁用联网工具
- `LYY_KB_PATH`：可选自定义 JSONL 知识库路径
- `LANGSMITH_TRACING`、`LANGSMITH_API_KEY`：可选链路追踪

`.env` 不得提交到 Git。本项目只提交空值的 `.env.example`。

## Tavily 联网工具

页面中的“允许 Tavily 联网补充”默认关闭。开启后，Agent 仍不会对每个问题都
联网，只会在以下情况调用：

- 本地知识库经过限定次数的检索和问题改写后仍缺少证据；
- 问题明确包含“最新”“当前”“官方政策”等时效性要求。

默认配置使用 Tavily `basic` 搜索、最多5条结果，并限制为
`advertising.amazon.com`。联网内容会标记为补充证据，不能静默覆盖本地规范；
两者冲突时必须提示人工核对后台和最新官方政策。

Tavily 配置位于 `config/settings.json`：

```json
{
  "tavily": {
    "enabled": true,
    "search_depth": "basic",
    "max_results": 5,
    "include_domains": ["advertising.amazon.com"]
  }
}
```

## 使用私有知识库

公开 Demo 位于 `data/demo/documents.jsonl`。自己的资料应放在被 Git 忽略的目录：

```text
data/private_knowledge_base/
├─ documents/       # 人工审核后的 Markdown
└─ exports/         # 自动生成的 RAG JSONL
```

将敏感客户名或账户名称逐行写入 `config/private_terms.txt`，然后运行：

```powershell
python scripts/build_dsp_rag_exports.py
python manage_kb.py build-index
```

存在私有导出文件时，Agent 会优先使用私有知识库；否则自动使用公开 Demo。

也可以先提取原始文件，人工检查后再批准：

```powershell
python manage_kb.py extract "inbox/example.pdf" --pages 2-9
python manage_kb.py list-pending
python manage_kb.py approve "data/imports/pending/import-xxx.md" `
  --title "标题" --category "分类" --topic "主题" --confirmed
```

原始文件、待审核内容、批准后的私有文档和索引全部被 `.gitignore` 排除。

## 测试

```powershell
$env:LYY_OFFLINE="1"
python -m unittest discover -s tests -v
```

## 发布前安全检查

发布前至少确认：

```powershell
git status --short
git check-ignore .env data/private_knowledge_base runtime
```

不要提交原始办公文件、截图、SQLite 数据库或任何真实 Key。如果 Key 曾经进入
Git 历史，应立即撤销并重新生成，而不仅是删除文件。

## License

[MIT](LICENSE)
