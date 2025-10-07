import streamlit as st
from src.rag.pipeline import answer_rag 

st.set_page_config(page_title="Support RAG", page_icon="🤖", layout="wide")
st.title("🤖 Support RAG — LM Studio + Qwen")




with st.sidebar:
    st.subheader("Настройки запроса")
    persist_dir = st.text_input("Persist dir (Chroma)", value="vectorstore/chroma")
    collection_name = st.text_input("Collection", value="support")
    k = st.slider("k (top-k)", 1, 12, 5)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("Max tokens (вывод)", 128, 2048, 700, step=32)
    max_total_chars = st.slider("Лимит символов в контексте", 600, 4000, 2200, step=100)

    st.divider()
    if st.button("Очистить чат"):
        st.session_state["history"] = []

# ---- История диалога ----
if "history" not in st.session_state:
    st.session_state["history"] = []

# показать историю
for m in st.session_state["history"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---- Ввод пользователя ----
query = st.chat_input("Опишите проблему...")

if query:
    # 1) показать сообщение пользователя
    st.session_state["history"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 2) получить ответ от RAG
    with st.chat_message("assistant"):
        with st.spinner("Готовлю ответ..."):
            try:
                answer = answer_rag(
                    query=query,
                    persist_dir=persist_dir,
                    collection_name=collection_name,
                    k=k,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_total_chars=max_total_chars,
                )
            except Exception as e:
                answer = f"⚠️ Ошибка при вызове модели: {e}"
        st.markdown(answer)

    # 3) сохранить ответ в историю
    st.session_state["history"].append({"role": "assistant", "content": answer})
