import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from pymongo import MongoClient

# Load environment variables and configure API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    raise ValueError("GOOGLE_API_KEY not found. Please check your .env file.")


def get_database():
    CONNECTION_STRING = "mongodb+srv://madhogariaraksha27:HdAlSZloseoSMdNZ@llama.siihara.mongodb.net/user_questions?retryWrites=true&w=majority"
    client = MongoClient(CONNECTION_STRING)
    return client['user_questions']

def fetch_all_questions(user_id):
    db = get_database()
    questions_collection = db.questions
    return list(questions_collection.find({"user_id": user_id}, {'_id': 0, 'question': 1}))

def fetch_answer(question,user_id):
    db = get_database()
    questions_collection = db.questions
    result = questions_collection.find_one({"question": question,"user_id": user_id}, {'_id': 0, 'answer': 1})
    return result['answer'] if result else "No answer found for this question."

# Insert the new question and its response into the database
def insert_data(question, answer,user_id):
    try:
        db = get_database()
        questions_collection = db.questions
        question_data = {
            "user_id": user_id,
            "question": question,
            "answer": answer
        }
        questions_collection.insert_one(question_data)
        print("Data inserted successfully:", question_data)
    except Exception as e:
        print("An error occurred:", e)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ''
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, 'answer is not available in the context', don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

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

def show():
    
    # st.set_page_config(page_title="Chat with PDFs")
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
    with st.sidebar:
        st.title("Previously Asked Questions")
        all_questions = fetch_all_questions(user_id)
        selected_question = st.selectbox("Select a question to see the answer:", [q['question'] for q in all_questions])

        if st.button("Show Answer"):
            answer = fetch_answer(selected_question, user_id)
            st.write("Answer: " + answer)

    st.title("Doc Assist Bot")
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
    if st.button("Process PDFs"):
        if pdf_docs:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDFs processed and indexed.")

    # Question Interaction section
    # st.header("Question Interaction")
    # st.write("Ask a question by typing or using the microphone:")

    user_question = st.text_input("Type your question here:")
    if st.button("Use Microphone"):
        user_question = recognize_speech()
        if user_question:
            st.text_input("Type your question here:", value=user_question, key="question_field")

    if user_question:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        insert_data(user_question, response["output_text"], user_id)
        st.subheader("Response:")
        st.write(response["output_text"])

        if response["output_text"]:  # Ensures that there's a response to convert to speech
            audio_response = text_to_speech(response["output_text"])
            st.audio(audio_response, format='audio/mp3')

    # Previously Asked Questions section
    # st.header("Previously Asked Questions")
    # all_questions = fetch_all_questions()
    # selected_question = st.selectbox("Select a question to see the answer:", [q['question'] for q in all_questions])

    # if st.button("Show Answer"):
    #     answer = fetch_answer(selected_question)
    #     st.write("Answer: " + answer)


