import streamlit as st

st.title("Medical Bot")
st.write(
    "Hello World!"
)
st.image("sunrise.jpg", caption="Sunrise by the mountains")
st.button("Reset", type="primary")
if st.button("Say hello"):
    st.write("Why hello there")
else:
    st.write("Goodbye")
