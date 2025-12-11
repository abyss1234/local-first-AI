import os
import json
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Local-first AI Skeleton", layout="centered")
st.title("🧠 Local-first AI")
st.caption("Chat / RAG / Agent (local-first stack)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("Settings")

    mode = st.radio(
        "Mode",
        ["Chat", "RAG Chat", "Agent"],
        index=0,
        help="Choose how to talk to the backend.",
    )

    model = st.text_input("Model (optional, for Chat/Agent)", value="")
    system = st.text_area("System prompt (Chat only, optional)", value="")

    # RAG / Agent 用的 top_k
    top_k = st.slider("RAG / Agent top_k", min_value=1, max_value=10, value=3, step=1)

    # streaming 只对 Chat 有意义
    if mode == "Chat":
        use_stream = st.toggle("Stream output", value=False)
    else:
        use_stream = False

    st.divider()
    if st.button("Health check"):
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            st.json(r.json())
        except Exception as e:
            st.error(f"Health check failed: {e}")

# ---------------- History ----------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------------- Input ----------------
prompt = st.chat_input("Say something...")

if prompt:
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 不同模式调用不同的后端
    if mode == "Chat":
        # ---------- 普通聊天 ----------
        body = {"message": prompt}
        if model.strip():
            body["model"] = model.strip()
        if system.strip():
            body["system"] = system.strip()

        with st.chat_message("assistant"):
            if not use_stream:
                try:
                    r = requests.post(f"{API_URL}/chat", json=body, timeout=120)
                    if r.status_code != 200:
                        st.error(r.text)
                    else:
                        reply = r.json().get("reply", "")
                        st.markdown(reply)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": reply}
                        )
                except Exception as e:
                    st.error(f"Request failed: {e}")
            else:
                # Stream plain text from /chat?stream=true
                try:
                    r = requests.post(
                        f"{API_URL}/chat?stream=true",
                        json=body,
                        stream=True,
                        timeout=120,
                    )
                    if r.status_code != 200:
                        st.error(r.text)
                    else:
                        placeholder = st.empty()
                        acc = ""
                        for chunk in r.iter_content(chunk_size=None):
                            if not chunk:
                                continue
                            acc += chunk.decode("utf-8", errors="ignore")
                            placeholder.markdown(acc)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": acc}
                        )
                except Exception as e:
                    st.error(f"Stream failed: {e}")

    elif mode == "RAG Chat":
        # ---------- RAG 聊天 (/chat_rag) ----------
        body = {"message": prompt, "top_k": top_k}
        # model 可选（如果你的 /chat_rag 支持自定义模型，可以加上）
        if model.strip():
            body["model"] = model.strip()

        with st.chat_message("assistant"):
            try:
                r = requests.post(f"{API_URL}/chat_rag", json=body, timeout=180)
                if r.status_code != 200:
                    st.error(r.text)
                else:
                    data = r.json()
                    answer = data.get("answer", "")
                    citations = data.get("citations", [])

                    # 主回答
                    st.markdown(answer or "_(empty answer)_")
                    # 存入历史，带个前缀让你分辨
                    history_text = f"**[RAG]**\n\n{answer}"
                    st.session_state.messages.append(
                        {"role": "assistant", "content": history_text}
                    )

                    # 引用来源展开
                    if citations:
                        with st.expander("📚 Citations"):
                            for c in citations:
                                file = c.get("file", "unknown")
                                chunk_id = c.get("chunk_id", -1)
                                snippet = c.get("snippet", "")
                                st.markdown(
                                    f"- **{file}** `#{chunk_id}`\n\n  > {snippet}"
                                )
                    else:
                        st.info("No citations returned.")
            except Exception as e:
                st.error(f"RAG request failed: {e}")

    else:
        # ---------- Agent 模式 (/agent) ----------
        body = {"task": prompt, "top_k": top_k}
        if model.strip():
            body["model"] = model.strip()

        with st.chat_message("assistant"):
            try:
                r = requests.post(f"{API_URL}/agent", json=body, timeout=240)
                if r.status_code != 200:
                    st.error(r.text)
                else:
                    data = r.json()
                    answer = data.get("answer", "")
                    plan = data.get("plan", [])
                    tool_calls = data.get("tool_calls", [])
                    citations = data.get("citations", [])
                    note_path = data.get("note_path")
                    todos = data.get("todos")

                    # 主回答
                    st.markdown(answer or "_(empty answer)_")
                    history_text = f"**[Agent]**\n\n{answer}"
                    st.session_state.messages.append(
                        {"role": "assistant", "content": history_text}
                    )

                    # 计划
                    if plan:
                        with st.expander("📝 Plan"):
                            for i, step in enumerate(plan, start=1):
                                st.markdown(f"{i}. {step}")

                    # 工具调用
                    if tool_calls:
                        with st.expander("🛠 Tool calls"):
                            st.json(tool_calls)

                    # TODO 列表
                    if todos:
                        with st.expander("✅ TODOs"):
                            for t in todos:
                                st.markdown(f"- [{t.get('id')}] {t.get('task')}")

                    # 引用来源
                    if citations:
                        with st.expander("📚 Citations"):
                            for c in citations:
                                file = c.get("file", "unknown")
                                chunk_id = c.get("chunk_id", -1)
                                snippet = c.get("snippet", "")
                                st.markdown(
                                    f"- **{file}** `#{chunk_id}`\n\n  > {snippet}"
                                )

                    # 写 note 的路径
                    if note_path:
                        st.success(f"🗒 Note saved at: `{note_path}`")

            except Exception as e:
                st.error(f"Agent request failed: {e}")




# import os
# import json
# import requests
# import streamlit as st

# API_URL = os.getenv("API_URL", "http://localhost:8000")

# st.set_page_config(page_title="Local-first AI Skeleton", layout="centered")
# st.title("🧠 Local-first AI")
# st.caption("Chat / RAG / Agent.")

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# with st.sidebar:
#     st.subheader("Settings")
#     model = st.text_input("Model (optional)", value="")
#     system = st.text_area("System prompt (optional)", value="")
#     use_stream = st.toggle("Stream output", value=False)
#     st.divider()
#     if st.button("Health check"):
#         r = requests.get(f"{API_URL}/health", timeout=5)
#         st.json(r.json())

# # Show history
# for m in st.session_state.messages:
#     with st.chat_message(m["role"]):
#         st.markdown(m["content"])

# prompt = st.chat_input("Say something...")
# if prompt:
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     body = {"message": prompt}
#     if model.strip():
#         body["model"] = model.strip()
#     if system.strip():
#         body["system"] = system.strip()

#     with st.chat_message("assistant"):
#         if not use_stream:
#             r = requests.post(f"{API_URL}/chat", json=body, timeout=120)
#             if r.status_code != 200:
#                 st.error(r.text)
#             else:
#                 reply = r.json()["reply"]
#                 st.markdown(reply)
#                 st.session_state.messages.append({"role": "assistant", "content": reply})
#         else:
#             # Stream plain text from /chat?stream=true
#             r = requests.post(f"{API_URL}/chat?stream=true", json=body, stream=True, timeout=120)
#             if r.status_code != 200:
#                 st.error(r.text)
#             else:
#                 placeholder = st.empty()
#                 acc = ""
#                 for chunk in r.iter_content(chunk_size=None):
#                     if not chunk:
#                         continue
#                     acc += chunk.decode("utf-8", errors="ignore")
#                     placeholder.markdown(acc)
#                 st.session_state.messages.append({"role": "assistant", "content": acc})
