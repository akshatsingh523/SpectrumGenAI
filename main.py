import streamlit as st
import login  # Ensure this module has the necessary authentication functions
import docasssistbot
import vision
import content_summarizer
import invoice
import geminopro
from streamlit_lottie import st_lottie
import requests
import allinone
import Video_to_text_using_gemini

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def main():
    # This must be the very first Streamlit command used.
    st.set_page_config(layout="centered",page_title="Spectrum AI")
    # st.title("Welcome to Our Application")
    # Now you can proceed with other parts of your Streamlit app
    st.markdown("""
        <style>
            .stButton>button {
                width: auto;
                border-radius: 0.3rem;
                background-color: transparent;
                background: linear-gradient(to right, #32cd32, #4169e1);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            p, h1 {
                color: rgb(27, 156, 113);
            }
            h2,h3 {
                color: rgb(116 194 255);
            }
            .st-bs {
                color: rgb(116 194 255);
            }
        </style>
    """, unsafe_allow_html=True)
    
    
    run_app()

def run_app():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if st.session_state['logged_in']:
        navigation()
    else:
        authentication()


def authentication():
    # st.title("Welcome to Our Application")
    # show_lottie_animation()
    st.title("Welcome to Our Application")
    # st.sidebar.title("Authentication")
    page = st.sidebar.radio("Authentication", [ "Home","Register", "Login"], index=0)

    if page == "Home":
        show_lottie_animation()
    if page == "Login":
        login_form()
    elif page == "Register":
        registration_form()

def show_lottie_animation():
    """Function to display Lottie animation"""
    lottie_url = "https://lottie.host/33fd6eb2-0ed8-4309-8179-9ecf129a2faf/QU4TTKI3NH.json"
    lottie_animation = load_lottieurl(lottie_url)
    if lottie_animation:
        st_lottie(lottie_animation, width=300, height=300, key="lottie")
    else:
        st.error("Failed to load animation.")

def login_form():
    with st.form("Login Form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type='password', key="login_password")
        
        submitted = st.form_submit_button("Login")
        user = login.check_user(username, password)
        if submitted and user:
            st.session_state['logged_in'] = True
            st.session_state['username'] = user['username']  # Store the username in session_state
            st.session_state['user_id'] = user.get('_id', None)
            st.success("Logged in Successfully!")
            st.experimental_rerun()
        elif submitted:
            st.error("Invalid username or password")
# def logout():
#     keys_to_clear = ['logged_in', 'username', 'user_id']
#     for key in keys_to_clear:
#         if key in st.session_state:
#             del st.session_state[key]
#     st.experimental_rerun()

def registration_form():
    with st.form("Registration Form"):
        new_username = st.text_input("New Username", key="register_username")
        new_password = st.text_input("New Password", type='password', key="register_password")
        confirm_password = st.text_input("Confirm Password", type='password', key="confirm_password")
        registered = st.form_submit_button("Register")
        if registered:
            if new_password != confirm_password:
                st.error("Passwords do not match")
            elif login.create_user(new_username, new_password):
                st.success("Registration successful. You can now login.")
            else:
                st.error("Username already exists")

def navigation():
    """Display the navigation sidebar after a successful login."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["Home", "Doc Assist Bot", "LinkSynth", "Invoice Query", "PictoResponse","TextLink Gemini","Spectrum AI","NarratoVision" ], index=0)
    display_page(page)
    if st.sidebar.button("Logout"):
        keys_to_clear = ['logged_in', 'username', 'user_id']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()

        # st.session_state['logged_in'] = False
        # st.experimental_rerun()

def display_page(page):
    if page == "Home":
        # st.title(f"Welcome back, {st.session_state['username']}!")
        if 'username' in st.session_state:
            st.title(f"Welcome back, {st.session_state['username']}!")
        else:
            st.write("Welcome to the Home Page")
    elif page == "Doc Assist Bot":
        docasssistbot.show()
    elif page == "LinkSynth":
        content_summarizer.show()
    elif page == "Invoice Query":
        invoice.show()
    elif page == "PictoResponse":
        vision.show()
    elif page == "TextLink Gemini":
        geminopro.show()
    elif page == "NarratoVision":
        Video_to_text_using_gemini.show()
    elif page =="Spectrum AI":
        allinone.show()
    

if __name__ == '__main__':
    main()
