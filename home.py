import streamlit as st

st.set_page_config(page_title='First app', initial_sidebar_state="expanded")

st.header("Car - Dekho  Used Car Prediction")

st.image("C:/Users/Welcome/Desktop/usedcar/car.jpg")





x= st.button(" CLICK TO START")
st.balloons()
if x==True:
    st.switch_page("pages/MainPage.py")
