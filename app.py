import streamlit as st
import requests

# URL của FastAPI backend
BACKEND_URL = "http://127.0.0.1:8000/query/"

# Tạo tiêu đề cho ứng dụng
st.title("Chatbot QA System - Uchiha Itachi")
st.write("Nhập câu hỏi của bạn về nhân vật Uchiha Itachi")

# Nhập liệu từ người dùng
query = st.text_input("Nhập câu hỏi của bạn:", "")

# Gửi yêu cầu đến backend khi người dùng nhấn nút
if st.button("Gửi câu hỏi"):
    if query.strip():
        try:
            # Gửi yêu cầu POST đến backend
            response = requests.post(
                BACKEND_URL,
                json={"query_text": query},
            )
            # Xử lý phản hồi từ backend
            if response.status_code == 200:
                data = response.json()
                st.success("Câu trả lời:")
                st.write(data["response"])
                st.write("Nguồn tài liệu liên quan:")
                for source in data["sources"]:
                    st.write(f"- {source}")
            else:
                st.error(f"Lỗi từ backend: {response.json().get('detail', 'Không rõ lỗi')}")
        except Exception as e:
            st.error(f"Lỗi: {str(e)}")
    else:
        st.warning("Vui lòng nhập câu hỏi!")

# Ghi chú ở cuối giao diện
st.sidebar.title("Thông tin ứng dụng")
st.sidebar.info(
    """
    Ứng dụng QA Chatbot sử dụng:
    - Mô hình Ollama để sinh câu trả lời
    - Vector Database (Chroma) để tìm tài liệu liên quan
    """
)
