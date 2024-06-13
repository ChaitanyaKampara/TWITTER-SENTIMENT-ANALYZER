import streamlit as st
import requests

# FastAPI endpoint
FASTAPI_URL = "http://127.0.0.1:8000/predict"

st.title("Tweet Sentiment Analysis")
st.write("Enter a tweet to predict its sentiment:")

tweet_text = st.text_input("Tweet")
st.caption("boss ra lucha")

if st.button("Predict"):
    if tweet_text:
        response = requests.post(FASTAPI_URL, json={"text": tweet_text})
        if response.status_code == 200:
            sentiment = response.json().get("sentiment")
            st.write(f"Predicted Sentiment: {sentiment}")
        else:
            st.write("Error: Could not get a response from the API.")
    else:
        st.write("Please enter a tweet.")
