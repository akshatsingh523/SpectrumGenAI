import os
import fitz
import requests
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from PIL import Image
from PyPDF2 import PdfReader
from gtts import gTTS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from io import BytesIO
import speech_recognition as sr
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
from io import BytesIO
import google.generativeai as genai
from pymongo import MongoClient

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
recognizer = sr.Recognizer()

def get_database():
    CONNECTION_STRING = "mongodb+srv://madhogariaraksha27:HdAlSZloseoSMdNZ@llama.siihara.mongodb.net/user_questions?retryWrites=true&w=majority"
    client = MongoClient(CONNECTION_STRING)
    return client['user_questions']['allinone']

def insert_question_answer(question, answer,user_id):
    db = get_database()
    db.insert_one({'question': question, 'answer': answer,'user_id': user_id})

def fetch_all_questions(user_id):
    db = get_database()
    questions = list(db.find({"user_id": user_id}, {'_id': 0, 'question': 1}))
    return [q['question'] for q in questions]  

def fetch_answer(question,user_id):
    db = get_database()
    answer = db.find_one({'question': question,"user_id": user_id}, {'_id': 0, 'answer': 1})
    return answer['answer'] if answer else "Answer not found."

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

def get_gemini_responses(question):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(question)
    return response.text

def truncate_snippet(snippet, word_limit=15):
    words = snippet.split()
    if len(words) > word_limit:
        truncated = ' '.join(words[:word_limit]) + '...'
    else:
        truncated = snippet
    return truncated


BING_API_KEY = 'fdc44c7aa1014ad8853d6c9cc8813253'
def process_video(video_file):
    # Temporary save uploaded video
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(video_file.getvalue())

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_directory = 'Selected_frames'
    os.makedirs(output_directory, exist_ok=True)

    selected_frames = []
    previous_frame = None
    threshold = 0.40  # Change threshold according to need

    for frame_idx in tqdm(range(n_frames), desc="Processing Frames"):
        ret, img = cap.read()
        if not ret:
            break
        if previous_frame is not None:
            b, g, r = cv2.split(img)
            ssim_b, _ = ssim(previous_frame[0], b, full=True)
            ssim_g, _ = ssim(previous_frame[1], g, full=True)
            ssim_r, _ = ssim(previous_frame[2], r, full=True)
            similarity_index = (ssim_b + ssim_g + ssim_r) / 3
            if similarity_index < threshold:
                selected_frames.append(img)
                frame_filename = os.path.join(output_directory, f"frame_{frame_idx:04d}.png")
                cv2.imwrite(frame_filename, img)
        previous_frame = cv2.split(img)

    cap.release()
    return output_directory

def generate_summary(output_directory):
    model = genai.GenerativeModel('gemini-pro-vision')
    model1 = genai.GenerativeModel('gemini-pro')
    descriptions = []

    for frame_filename in os.listdir(output_directory):
        frame_path = os.path.join(output_directory, frame_filename)
        with Image.open(frame_path) as img:
            response = model.generate_content(["Provide a short description about the image.", img])
            descriptions.append(response.text)

    summary_prompt = "Summarize the Content"
    combined_text = " ".join(descriptions)
    summary_response = model1.generate_content([summary_prompt, combined_text])
    return summary_response.text

# docassist
def get_pdf_text(pdf_docs):
    text = ""
    # for pdf in pdf_docs:
    pdf_reader = PdfReader(pdf_docs)
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
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # prompt_template = """
    # Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    # provided context just say, 'answer is not available in the context', don't provide the wrong answer\n\n
    # Context:\n {context}?\n
    # Question: \n{question}\n
    # Answer:
    # """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
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

# vision
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
def get_gemini_responses2(question):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(question)
    return response.text

def get_gemini_responses(input,image):
    model=genai.GenerativeModel(model_name='gemini-pro-vision',safety_settings=safety_settings)
    if input!="":
        response=model.generate_content([input,image])
    else:
        response=model.generate_content(image)
    return response.text

# for contentsum
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]
        return transcript
    except Exception as e:
        raise e

def extract_article_content(article_url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(article_url,headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extracts all text within paragraph tags, you can adjust the tags based on your needs
        paragraphs = soup.find_all('p')
        article_text = ' '.join([para.get_text() for para in paragraphs])
        return article_text
    except Exception as e:
        raise e

def generate_gemini_content(content_text):
    prompt = '''You are a summarizer. You will be taking the content text and summarizing the entire content covering each and every important points discussed in the entire content. The content text will be appended here: '''
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + content_text)
    return response.text

def is_youtube_link(link):
    return 'youtube.com/watch?' in link or 'youtu.be/' in link


# functions call for streamlit



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
    with st.sidebar:
        st.title("Previous Questions")
        all_questions = fetch_all_questions(user_id)
        selected_question = st.selectbox("Select a question:", all_questions, key='previous_questions')

        if st.button("Get Answer", key='get_answer'):
            answer = fetch_answer(selected_question,user_id)
            st.text_area("Answer:", value=answer, height=100)

    st.title("Spectrum AI")
    st.markdown("""
        <style>
            .st-emotion-cache-cnbvxy {
                color: rgb(27, 156, 113); 
        </style>
    """, unsafe_allow_html=True)
    input_text = st.text_area("Enter Text or URL:")
    if st.button("Use Microphone"):
        input_text = recognize_speech()
        if input_text:
            st.text_input("Type your question here:", value=input_text, key="question_field")

    uploaded_file = st.file_uploader("Upload a document or image",accept_multiple_files=True, type=['pdf', 'jpg', 'jpeg', 'png','mp4'])

    # submit = st.button("Process Input")
    
    if input_text and not uploaded_file :
        if 'http' in input_text :
            if input_text:
            # prompt = '''You are a summarizer. You will be taking the content text and summarizing the entire content covering each and every important points discussed in the entire content. The content text will be appended here: '''
                if is_youtube_link(input_text):
                    if 'youtube.com/watch?' in input_text:
                        video_id = input_text.split("=")[1]
                    else: # shortened YouTube URL
                        video_id = input_text.split("/")[-1]
                    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
                else:
                    st.markdown("### Article Content Detected")

            if st.button("Get Detailed Notes"):
                try:
                    if is_youtube_link(input_text):
                        content_text = extract_transcript_details(input_text)
                    else:
                        content_text = extract_article_content(input_text)
    
                    if content_text:
                        summary = generate_gemini_content(content_text)
                        st.markdown("## Detailed Notes:")
                        st.write(summary)
                        audio_file = text_to_speech(summary)
                        st.audio(audio_file)
                except requests.exceptions.HTTPError as err:
                    st.error(f"Failed to fetch content due to an HTTP error: {err}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")


        elif 'http' not in input_text:
            col1, col2 = st.columns([3,2])

# Column 1 for Gemini LLM Application
            with col1:
                if input_text:
                    response = get_gemini_responses2(input_text)
                    st.subheader("Response:")
                    st.write(response)
                    audio_file = text_to_speech(response)
                    st.audio(audio_file)
                    insert_question_answer(input_text, response,user_id)
            
            
# Column 2 for Bing search links
            with col2:
    # st.header("Search for Related Links")
                if input_text:  # Ensures search term is based on the input question
                    url = f"https://api.bing.microsoft.com/v7.0/search?q={input_text}"
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

    if uploaded_file:
            if st.button("process"):
                for uploade_file in uploaded_file:
                    if uploade_file.type.endswith('pdf'):
                    
        
                        with st.spinner("Processing PDFs..."):
                            raw_text = get_pdf_text(uploade_file)
                            text_chunks = get_text_chunks(raw_text)
                            get_vector_store(text_chunks)
                            st.success("PDFs processed and indexed.")
                        if input_text:
                            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                            docs = new_db.similarity_search(input_text)
                            chain = get_conversational_chain()
                            response = chain({"input_documents": docs, "question": input_text}, return_only_outputs=True)
        
                            st.subheader("Response:")
                            st.write(response["output_text"])

                            if response["output_text"]:  # Ensures that there's a response to convert to speech
                                audio_response = text_to_speech(response["output_text"])
                                st.audio(audio_response, format='audio/mp3')
                                insert_question_answer(input_text, response["output_text"],user_id)

                    elif uploade_file.type.endswith(('png','jpg','jpeg')):
                        image=""   
                        if uploade_file is not None:
                            image = Image.open(uploade_file)
                            st.image(image, caption="Uploaded Image.", use_column_width=True)
                        
    
                            response=get_gemini_responses(input_text,image)
                            st.subheader("Response:")
                            st.write(response)
                            insert_question_answer(input_text, response, user_id)
        
                            audio_file = text_to_speech(response)
                            st.audio(audio_file) 
                    elif uploade_file.type.endswith('mp4'):
                        output_directory = process_video(uploade_file)
                        summary_text = generate_summary(output_directory)
                        st.subheader("Video Summary")
                        st.write(summary_text)

    
      

               

