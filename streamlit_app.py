import sys
import uuid
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT / "vendor"))
from lyy_rag_agent.langchain_runtime import LangChainDSPRAGAgent  # noqa: E402


st.set_page_config(
    page_title="DSP RAG Agent",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container {max-width: 980px; padding-top: 2rem;}
  [data-testid="stChatMessage"] {border: 1px solid #e7eaf1; border-radius: 14px; padding: .35rem .7rem;}
  [data-testid="stSidebar"] {background: #f7f8fc;}
  .status-ok {color: #087a55; font-weight: 650;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="正在加载 RAG 知识库与 LangChain 工作流…")
def get_agent():
    return LangChainDSPRAGAgent()


agent = get_agent()

query_session = st.query_params.get("session")
if "session_id" not in st.session_state:
    st.session_state.session_id = str(query_session or uuid.uuid4().hex[:12])
    st.query_params["session"] = st.session_state.session_id
if "messages" not in st.session_state:
    st.session_state.messages = [
        item for item in agent.base.memory.get(st.session_state.session_id, 100)
        if item["content"].strip()
    ]

with st.sidebar:
    st.title("DSP 知识助手")
    st.caption("LangChain 1.x · RAG · Qwen/DeepSeek · SQLite")
    st.markdown('<p class="status-ok">● Agent 已就绪</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("知识块", len(agent.base.embedding_index.chunks))
    col2.metric("向量", len(agent.base.embedding_index.vectors))
    st.write("Embedding：", "已就绪" if agent.base.embedding_index.ready else "离线")
    st.write("Rerank：", "已就绪" if agent.base.reranker.available else "离线")
    st.write("会话：", st.session_state.session_id)
    st.write("知识库：", "本地私有数据" if agent.base.knowledge_mode == "private" else "公开 Demo 数据")
    show_trace = st.toggle("显示 Agent 工作流", value=False)
    mode_label = st.radio(
        "回答模式",
        ["自动", "快速", "深度"],
        horizontal=True,
        help="自动：简单问题快速回答，复杂问题深度核验；快速：本地检索且跳过 Reviewer；深度：Qwen3-Max 并完整核验。",
    )
    response_mode = {"自动": "auto", "快速": "fast", "深度": "deep"}[mode_label]
    if st.button("新建对话", use_container_width=True):
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.session_state.messages = []
        st.query_params["session"] = st.session_state.session_id
        st.rerun()
    if st.button("清空当前对话", use_container_width=True):
        agent.base.memory.clear(st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("敏感字段会在进入模型前脱敏；知识块编号仅供内部核验，不显示在回答中。")

st.title("DSP RAG Agent")
st.caption("面向 DSP 新手的配置规范、素材要求与异常排查知识库")

if not st.session_state.messages:
    st.info("可以询问：Line item 类型、Order 配置、素材尺寸、受众定向、指标公式或无曝光排查。")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("请输入 DSP 配置问题…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库并核验回答…"):
            try:
                response = agent.invoke(question, st.session_state.session_id, response_mode)
            except Exception as exc:
                st.error("Agent 处理失败：{}".format(exc))
            else:
                st.markdown(response.answer)
                st.caption("{} · {} · Reviewer {}".format(
                    response.provider,
                    response.route.intent,
                    "通过" if response.review_passed else "需关注",
                ))
                total_ms = response.trace[-1].get("total_ms", 0)
                st.caption("总耗时：{:.2f} 秒 · {}模式".format(total_ms / 1000, mode_label))
                if show_trace:
                    with st.expander("Agent 工作流"):
                        timings = [
                            {
                                "环节": item.get("node"),
                                "耗时（秒）": round(item.get("elapsed_ms", 0) / 1000, 3),
                                "模型": item.get("model", ""),
                            }
                            for item in response.trace if "elapsed_ms" in item
                        ]
                        st.dataframe(timings, use_container_width=True, hide_index=True)
                        st.json(response.trace)
                st.session_state.messages.append({"role": "assistant", "content": response.answer})
