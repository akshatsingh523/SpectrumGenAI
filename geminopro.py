import streamlit as st
import os
import requests
import speech_recognition as sr
from gtts import gTTS
import google.generativeai as genai
from io import BytesIO
from pymongo import MongoClient

def get_database():
    CONNECTION_STRING = os.getenv("mongodb+srv://madhogariaraksha27:HdAlSZloseoSMdNZ@llama.siihara.mongodb.net/user_questions?retryWrites=true&w=majority")
    client = MongoClient(CONNECTION_STRING)
    return client['user_questions']['responses']

from dotenv import load_dotenv
load_dotenv()

# Set up Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-pro")
recognizer = sr.Recognizer()

def insert_question_response(question, response,user_id):
    db = get_database()
    db.insert_one({'question': question, 'response': response,"user_id": user_id})

def fetch_all_questions(user_id):
    db = get_database()
    questions = list(db.find({"user_id": user_id}, {'_id': 0, 'question': 1}))
    return [q['question'] for q in questions]

def fetch_response(question,user_id):
    db = get_database()
    response = db.find_one({'question': question,"user_id": user_id}, {'_id': 0, 'response': 1})
    return response['response'] if response else "Response not found."


def get_gemini_responses(question):
    response = model.generate_content(question)
    return response.text

def truncate_snippet(snippet, word_limit=15):
    words = snippet.split()
    if len(words) > word_limit:
        truncated = ' '.join(words[:word_limit]) + '...'
    else:
        truncated = snippet
    return truncated

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

# Bing API Key
BING_API_KEY = 'fdc44c7aa1014ad8853d6c9cc8813253'

#
# Streamlit page config
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
    if 'history' not in st.session_state:
        st.session_state['history'] = ""

    with st.sidebar:
        st.title("Previous Questions")
        all_questions = fetch_all_questions(user_id)
        selected_question = st.selectbox("Select a question:", all_questions, key='previous_questions')

        if st.button("Get Answer", key='get_answer'):
            answer = fetch_response(selected_question,user_id)
            st.text_area("Answer:", value=answer, height=100)


    st.title("TextLink Gemini")
    input_question = st.text_input("Input your question:")

    use_microphone = st.checkbox('Use Microphone', key='use_mic')    
    if use_microphone:
        with sr.Microphone() as source:
            st.write("Listening...")
            audio_data = recognizer.listen(source)
            recognizer.adjust_for_ambient_noise(source)
            try:
                input_question = recognizer.recognize_google(audio_data)
                st.text_input("Input your question:",value=input_question , key="input")
            except sr.UnknownValueError:
                st.error("Google Speech Recognition could not understand audio.")
            except sr.RequestError as e:
                st.error(f"Could not request results from Google Speech Recognition service; {e}")
            
# Creating two columns
    col1, col2 = st.columns([3,2])

# Column 1 for Gemini LLM Application
    with col1:
        if input_question:
            response = get_gemini_responses(input_question)
            st.subheader("Response:")
            st.write(response)
        

            audio_response = text_to_speech(response)
            st.audio(audio_response)
            insert_question_response(input_question, response,user_id)
            st.session_state['history'] += f"\nUser: {input_question}\nAI: {response}"
# Column 2 for Bing search links
    with col2:
    # st.header("Search for Related Links")
        if input_question:  # Ensures search term is based on the input question
            url = f"https://api.bing.microsoft.com/v7.0/search?q={input_question}"
            headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
            bing_response = requests.get(url, headers=headers)
        # links = [result['url'] for result in bing_response.json().get('webPages', {}).get('value', [])]
            results = bing_response.json().get('webPages', {}).get('value', [])
            if results:
                st.subheader(f"Top {len(results)} Links:")
                for result in results:
                    title = result['name']
                    snippet = result.get('snippet', 'No description available.')
                    display_url = result['displayUrl']
                    truncated_snippet = truncate_snippet(snippet)
                # Using Markdown to display the link with the title and URL
                    st.markdown(f"**[{title}]({display_url})**")
                    st.caption(truncated_snippet)  # Using caption for the snippet
            else:
                st.write("No results found.")

    # st.text_area("Chat History", value=st.session_state['history'], height=200)

# Optional: Add a button to clear the conversation history
    if st.button("Clear History"):
        st.session_state['history'] = ""
        st.experimental_rerun()   