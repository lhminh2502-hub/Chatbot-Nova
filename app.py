import streamlit as st
import time
from chat_inference import chatbot_response

st.set_page_config(page_title=" Chatbot Nova ", page_icon="🤖", layout="wide")

# Khởi tạo session_state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "just_sent" not in st.session_state:
    st.session_state.just_sent = False

st.title("🤖 Chatbot Nova ")

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style="text-align: right; margin: 10px;">
                <span style="background-color: #007bff; color: white; padding: 10px; border-radius: 15px;">
                    {msg["content"]}
                </span> 🧑
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="text-align: left; margin: 10px;">
                🤖 <span style="background-color: #222; color: #00ffcc; padding: 10px; border-radius: 15px;">
                    {msg["content"]}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Ô nhập tin nhắn
user_input = st.text_input("Nhập tin nhắn:", value=st.session_state.user_input, key="chat_input")

# Xử lý khi người dùng gửi tin
if user_input and not st.session_state.just_sent:
    st.session_state.just_sent = True

    # Lưu tin nhắn user
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Bot trả lời
    with st.spinner("🤖 Nova đang trả lời..."):
        time.sleep(0.8)
        bot_reply = chatbot_response(user_input)

    st.session_state.messages.append({"role": "bot", "content": bot_reply})

    # Reset input
    st.session_state.user_input = ""
    st.rerun()
else:
    st.session_state.just_sent = False
