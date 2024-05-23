import streamlit as st
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import requests
from bs4 import BeautifulSoup
from gtts import gTTS
from io import BytesIO

load_dotenv()   # Load env variables

import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

prompt = '''You are a summarizer. You will be taking the content text and summarizing the entire content covering each and every important points discussed in the entire content. The content text will be appended here: '''

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

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

def generate_gemini_content(content_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + content_text)
    return response.text

def is_youtube_link(link):
    return 'youtube.com/watch?' in link or 'youtu.be/' in link
def show():
    st.title("LinkSynth")
    st.markdown("""
        <style>
            .st-emotion-cache-cnbvxy {
                color: rgb(27, 156, 113); 
        </style>
    """, unsafe_allow_html=True)
    content_link = st.text_input("Enter YouTube Video Link or Article URL:")

    if content_link:
        if is_youtube_link(content_link):
            if 'youtube.com/watch?' in content_link:
                video_id = content_link.split("=")[1]
            else: # shortened YouTube URL
                video_id = content_link.split("/")[-1]
            st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
        else:
            st.markdown("### Article Content Detected")

    if st.button("Get Detailed Notes"):
        try:
            if is_youtube_link(content_link):
                content_text = extract_transcript_details(content_link)
            else:
                content_text = extract_article_content(content_link)
    
            if content_text:
                summary = generate_gemini_content(content_text, prompt)
                st.markdown("## Detailed Notes:")
                st.write(summary)
                audio_file = text_to_speech(summary)
                st.audio(audio_file) 
        except requests.exceptions.HTTPError as err:
            st.error(f"Failed to fetch content due to an HTTP error: {err}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

