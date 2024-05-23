from dotenv import load_dotenv
load_dotenv() ## load all the environment variables from .env

import streamlit as st
import os
from PIL import Image
import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
recognizer = sr.Recognizer()
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


## Load Gemini pro vision model
model=genai.GenerativeModel(model_name='gemini-pro-vision',safety_settings=safety_settings)

def get_gemini_response(input,image,user_prompt):
    response=model.generate_content([input,image[0],user_prompt])
    return response.text
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio_data = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            st.error("Could not understand the audio. Please try again.")
            return None
        except sr.RequestError as e:
            st.error(f"Speech recognition error: {e}")
            return None

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

def input_image_details(uploaded_file):
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")


##initialize our streamlit app
def show():
# st.set_page_config(page_title="MultiLanguage Invoice Extractor")
    st.title("Invoice Query")
    st.markdown("""
        <style>
            .st-emotion-cache-cnbvxy {
                color: rgb(27, 156, 113); 
        </style>
    """, unsafe_allow_html=True)
    input=st.text_input("Input Prompt: ",key="input")
    if st.button("Use Microphone"):
        input = recognize_speech()
        if input:
            st.text_input("Type your question here:", value=input, key="question_field")

    uploaded_file = st.file_uploader("Choose an image of the invoice...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image=Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

    submit=st.button("Tell me about the invoice")

    input_prompt="""
    You are an expert in understanding invoices. We will upload a a image as invoice
    and you will have to answer any questions based on the uploaded invoice image
    """

## if submit button is clicked

    if submit:
        image_data=input_image_details(uploaded_file)
        response=get_gemini_response(input_prompt,image_data,input)
        st.subheader("Response:")
        st.write(response)
        audio_file = text_to_speech(response)
        st.audio(audio_file) 