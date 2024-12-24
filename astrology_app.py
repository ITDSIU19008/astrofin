import streamlit as st


# Create a select box for language selection in the sidebar
language = st.sidebar.selectbox("Select Language:", ["Tiếng Việt", "English"], index=0)

# Check the selected language and display the respective link and quote
if language == "English":
    st.title("Astrotomi has finally have a new home")
    st.markdown("<h2>Click here to see what the stars say about your financial journey</h2>", unsafe_allow_html=True)
    st.markdown("<h1><a href='https://timo.vn/en/astrotomi/' target='_blank'>Visit AstroTomi</a></h1>", unsafe_allow_html=True)
elif language == "Tiếng Việt":
    st.title("Astrotomi nay đã có nhà mới")
    st.markdown("<h2>Truy cập ngay link này để xem những chòm sao nói gì về hành trình tài chính của bạn nhé</h2>", unsafe_allow_html=True)
    st.markdown("<h1><a href='https://timo.vn/astrotomi/' target='_blank'>Truy cập AstroTomi</a></h1>", unsafe_allow_html=True)
