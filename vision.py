from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os 
import google.generativeai as genai
from PIL import Image
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from pymongo import MongoClient

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
model=genai.GenerativeModel(model_name='gemini-pro-vision',safety_settings=safety_settings)
def get_database():
    CONNECTION_STRING = "mongodb+srv://madhogariaraksha27:HdAlSZloseoSMdNZ@llama.siihara.mongodb.net/user_questions?retryWrites=true&w=majority"
    client = MongoClient(CONNECTION_STRING)
    return client['user_questions']['vision']

def insert_question_answer(question, answer,user_id):
    db = get_database()
    db.insert_one({'question': question, 'answer': answer,"user_id": user_id})

def fetch_all_questions(user_id):
    db = get_database()
    questions = list(db.find({"user_id": user_id}, {'_id': 0, 'question': 1}))
    return [q['question'] for q in questions]  

def fetch_answer(question,user_id):
    db = get_database()
    answer = db.find_one({"user_id": user_id,'question': question}, {'_id': 0, 'answer': 1})
    return answer['answer'] if answer else "Answer not found."


def get_gemini_responses(input,image):
    if input!="":
        response=model.generate_content([input,image])
    else:
        response=model.generate_content(image)
    return response.text

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

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


def show():
    st.markdown("""
        <style>
            .st-emotion-cache-cnbvxy {
                color: rgb(27, 156, 113); 
        </style>
    """, unsafe_allow_html=True)
    if 'user_id' not in st.session_state:
        st.error("Please log in to view and interact with questions.")
        return
    user_id = st.session_state['user_id']
# st.set_page_config(page_title="text_to_video demo")
    with st.sidebar:
        st.title("Previous Questions")
        all_questions = fetch_all_questions(user_id)
        selected_question = st.selectbox("Select a question:", all_questions, key='previous_questions')

        if st.button("Get Answer", key='get_answer'):
            answer = fetch_answer(selected_question,user_id)
            st.text_area("Answer:", value=answer, height=100)


    st.title("PictoResponse")
    input=st.text_input("Input prompt: ",key="input")
  
    if st.button("Use Microphone"):
        input = recognize_speech()
        if input:
            st.text_input("Type your question here:", value=input, key="question_field")

    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image=""   
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)


    submit=st.button("Tell me about the image")

    if submit:
    
        response=get_gemini_responses(input,image)
        st.subheader("Response:")
        st.write(response)
        insert_question_answer(input, response,user_id) 
        audio_file = text_to_speech(response)
        st.audio(audio_file) 