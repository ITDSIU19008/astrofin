import streamlit as st

# Title for the app
st.title("Language-Based Redirect")

# Create a select box for language selection
language = st.selectbox("Select Language:", ["English", "Tiếng Việt"])

# Check the selected language and display the respective link
if language == "English":
    st.markdown("[Visit AstroTomi at](https://timo.vn/en/astrotomi/)", unsafe_allow_html=True)
elif language == "Tiếng Việt":
    st.markdown("[Truy cập AstroTomi tại nhà mới tại link này nhaaa](https://timo.vn/astrotomi/)", unsafe_allow_html=True)
