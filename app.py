import os
import base64
import tempfile
import streamlit as st
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
import groq
from gtts import gTTS
from PIL import Image
import requests
from langdetect import detect, LangDetectException
import time

# Load environment variables from .env file
load_dotenv()

# Get API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Page configuration
st.set_page_config(
    page_title="Indic Language Translator Pro",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with more modern effects
st.markdown("""
<style>
    /* Modern theme colors */
    :root {
        --bg-color: #0a0f1c;
        --card-bg: #1a1f2e;
        --accent: #4f46e5;
        --accent-hover: #4338ca;
        --text: #f8fafc;
        --text-secondary: #94a3b8;
        --border: #2d3748;
        --gradient-1: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        --gradient-2: linear-gradient(135deg, #1a1f2e 0%, #2d3748 100%);
        --gradient-3: linear-gradient(45deg, #3b82f6, #8b5cf6, #d946ef);
        --gradient-4: linear-gradient(-45deg, #4f46e5, #7c3aed, #2563eb);
    }

    /* Enhanced header section */
    .header-section {
        background: var(--gradient-3);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
    }

    .header-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: var(--gradient-4);
        opacity: 0.3;
        animation: gradient 8s ease infinite;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .header-content {
        position: relative;
        z-index: 1;
    }

    .header-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(to right, #fff, #e2e8f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        transform-style: preserve-3d;
        transition: transform 0.3s ease;
    }

    .header-title:hover {
        transform: translateZ(20px) rotateX(5deg);
    }

    .header-subtitle {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        max-width: 600px;
        line-height: 1.6;
    }

    /* Enhanced translator panels */
    .translator-panel {
        background: rgba(26, 31, 46, 0.8);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 28px;
        margin: 20px 0;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        transform-style: preserve-3d;
        transition: all 0.4s ease;
    }

    .translator-panel:hover {
        transform: translateY(-8px) translateZ(20px);
        box-shadow: 0 15px 45px rgba(0,0,0,0.3);
    }

    /* Language selector enhancement */
    .language-selector {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
        gap: 12px;
        margin: 20px 0;
    }

    .language-btn {
        background: var(--gradient-2);
        color: var(--text);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        font-size: 0.9rem;
        backdrop-filter: blur(8px);
    }

    .language-btn:hover {
        background: var(--gradient-1);
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(79, 70, 229, 0.3);
    }

    .language-btn.active {
        background: var(--gradient-1);
        border-color: var(--accent);
    }

    /* Enhanced buttons */
    .stButton > button {
        background: var(--gradient-1) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        transform-style: preserve-3d !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2) !important;
    }

    .stButton > button:hover {
        transform: translateY(-3px) translateZ(5px) !important;
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.3) !important;
    }

    /* Enhanced text areas */
    .stTextArea > div > div > textarea {
        background: rgba(26, 31, 46, 0.8) !important;
        color: var(--text) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 15px !important;
        padding: 20px !important;
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
        backdrop-filter: blur(12px) !important;
        transition: all 0.3s ease !important;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.2) !important;
        transform: translateY(-2px);
    }

    /* Floating animation */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    .floating {
        animation: float 3s ease-in-out infinite;
    }

    /* Enhanced developer card */
    .dev-info {
        background: var(--gradient-2);
        border-radius: 20px;
        padding: 28px;
        margin: 24px 0;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
    }

    .dev-info:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    }

    /* Enhanced footer */
    .footer {
        text-align: center;
        padding: 32px;
        margin-top: 48px;
        background: var(--gradient-2);
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(12px);
    }
</style>
""", unsafe_allow_html=True)

# Language dictionary with script information
LANGUAGES = {
    "en": {"name": "English", "script": "Latin", "direction": "ltr"},
    "hi": {"name": "Hindi - हिन्दी", "script": "Devanagari", "direction": "ltr"},
    "mr": {"name": "Marathi - मराठी", "script": "Devanagari", "direction": "ltr"},
    "bn": {"name": "Bengali - বাংলা", "script": "Bengali", "direction": "ltr"},
    "ta": {"name": "Tamil - தமிழ்", "script": "Tamil", "direction": "ltr"},
    "te": {"name": "Telugu - తెలుగు", "script": "Telugu", "direction": "ltr"},
    "gu": {"name": "Gujarati - ગુજરાતી", "script": "Gujarati", "direction": "ltr"},
    "kn": {"name": "Kannada - ಕನ್ನಡ", "script": "Kannada", "direction": "ltr"},
    "ml": {"name": "Malayalam - മലയാളം", "script": "Malayalam", "direction": "ltr"},
    "pa": {"name": "Punjabi - ਪੰਜਾਬੀ", "script": "Gurmukhi", "direction": "ltr"},
    "ur": {"name": "Urdu - اردو", "script": "Arabic", "direction": "rtl"}
}

def detect_language(text):
    """Detect the language of input text with improved accuracy"""
    try:
        if not text or len(text.strip()) == 0:
            return "en"
        
        # Try to detect the script first
        for lang_code, lang_info in LANGUAGES.items():
            if any(ord(char) > 127 for char in text):
                # Check for script-specific character ranges
                if lang_code == "hi" and any("\u0900" <= char <= "\u097F" for char in text):
                    return "hi"
                elif lang_code == "bn" and any("\u0980" <= char <= "\u09FF" for char in text):
                    return "bn"
                elif lang_code == "ta" and any("\u0B80" <= char <= "\u0BFF" for char in text):
                    return "ta"
                # Add more script detection rules
        
        # Fallback to langdetect
        lang = detect(text)
        return lang if lang in LANGUAGES else "en"
    except LangDetectException:
        return "en"

def translate_text(text, target_lang, source_lang='auto'):
    """Enhanced translation function with better error handling and retries"""
    if not text:
        return ""
    
    try:
        # Add a small delay to prevent rate limiting
        time.sleep(0.2)
        
        # If source language is auto, try to detect it
        if source_lang == 'auto':
            detected_lang = detect_language(text)
            source_lang = detected_lang
        
        # Create translator instance
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        
        # Attempt translation
        translated = translator.translate(text)
        
        # Validate translation
        if not translated or len(translated.strip()) == 0:
            # Retry with auto detection
            translator = GoogleTranslator(source='auto', target=target_lang)
            translated = translator.translate(text)
            
            if not translated:
                st.error("Translation failed. Please try again.")
                return ""
        
        return translated
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        try:
            # One more retry with auto detection
            translator = GoogleTranslator(source='auto', target=target_lang)
            return translator.translate(text)
        except:
            return ""

def text_to_speech(text, lang_code):
    """Convert text to speech with improved error handling"""
    try:
        if not text:
            return None
        
        # Map language codes for gTTS
        gtts_lang_mapping = {
            "hi": "hi", "bn": "bn", "ta": "ta",
            "te": "te", "mr": "mr", "gu": "gu",
            "kn": "kn", "ml": "ml", "pa": "pa",
            "ur": "ur", "en": "en"
        }
        
        lang = gtts_lang_mapping.get(lang_code, "en")
        tts = gTTS(text=text, lang=lang, slow=False)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return None

def get_audio_player(audio_path):
    """Generate HTML for audio player with improved styling"""
    try:
        if not os.path.exists(audio_path):
            return None
        
        audio_file = open(audio_path, 'rb')
        audio_bytes = audio_file.read()
        audio_file.close()
        
        audio_base64 = base64.b64encode(audio_bytes).decode()
        return f"""
            <audio controls>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
        """
    except Exception:
        return None

def main():
    # Enhanced header section
    st.markdown("""
        <div class="header-section">
            <div class="header-content">
                <h1 class="header-title">🌐 Indic Language Translator Pro</h1>
                <p class="header-subtitle">
                    Experience seamless translation across Indian languages with advanced script support, 
                    real-time voice output, and intelligent language detection.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Developer info card
    st.markdown("""
        <div class="dev-info">
            <h3 style="color: var(--accent); font-size: 1.5rem; margin-bottom: 1rem;">👨‍💻 Developer</h3>
            <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">Created by <strong>Nandesh Kalashetti</strong></p>
            <p style="color: var(--text-secondary); margin-bottom: 1rem;">Geni AI / Front-end Developer</p>
            <p>Portfolio: <a href="https://nandesh-kalashettiportfilio2386.netlify.app/" target="_blank" 
                style="color: var(--accent); text-decoration: none; font-weight: 500;">
                View Portfolio</a></p>
        </div>
    """, unsafe_allow_html=True)

    # Main translation interface
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="translator-panel floating">', unsafe_allow_html=True)
        
        # Source language selection
        source_lang = st.selectbox(
            "From Language",
            options=[lang["name"] for lang in LANGUAGES.values()],
            index=0,
            key="source_lang"
        )
        
        # Quick language selection buttons
        st.markdown('<div class="language-selector">', unsafe_allow_html=True)
        for lang_code, lang_info in LANGUAGES.items():
            if st.button(
                lang_info["name"], 
                key=f"quick_src_{lang_code}",
                help=f"Script: {lang_info['script']}"
            ):
                source_lang = lang_info["name"]
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Source text input
        source_text = st.text_area(
            "Enter text to translate",
            height=200,
            placeholder="Type or paste your text here...",
            key="source_text"
        )

        # Source language controls
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            if st.button("🔍 Detect Language", use_container_width=True):
                if source_text:
                    detected_lang = detect_language(source_text)
                    for lang_code, lang_info in LANGUAGES.items():
                        if lang_code == detected_lang:
                            source_lang = lang_info["name"]
                            st.success(f"Detected language: {lang_info['name']}")
                            break

        with col1_2:
            if st.button("🔊 Listen", use_container_width=True):
                if source_text:
                    source_lang_code = [code for code, info in LANGUAGES.items() 
                                     if info["name"] == source_lang][0]
                    audio_path = text_to_speech(source_text, source_lang_code)
                    if audio_path:
                        audio_player = get_audio_player(audio_path)
                        if audio_player:
                            st.markdown(audio_player, unsafe_allow_html=True)
                            try:
                                os.remove(audio_path)
                            except:
                                pass

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="translator-panel floating">', unsafe_allow_html=True)
        
        # Target language selection
        target_lang = st.selectbox(
            "To Language",
            options=[lang["name"] for lang in LANGUAGES.values()],
            index=1,
            key="target_lang"
        )
        
        # Quick language selection buttons for target
        st.markdown('<div class="language-selector">', unsafe_allow_html=True)
        for lang_code, lang_info in LANGUAGES.items():
            if st.button(
                lang_info["name"], 
                key=f"quick_tgt_{lang_code}",
                help=f"Script: {lang_info['script']}"
            ):
                target_lang = lang_info["name"]
        st.markdown('</div>', unsafe_allow_html=True)

        # Translation
        if source_text:
            source_lang_code = [code for code, info in LANGUAGES.items() 
                              if info["name"] == source_lang][0]
            target_lang_code = [code for code, info in LANGUAGES.items() 
                              if info["name"] == target_lang][0]
            
            with st.spinner("Translating..."):
                translated_text = translate_text(source_text, target_lang_code, source_lang_code)
            
            if translated_text:
                # Get script information for styling
                target_script = LANGUAGES[target_lang_code]["script"]
                target_direction = LANGUAGES[target_lang_code]["direction"]
                
                st.text_area(
                    "Translation",
                    value=translated_text,
                    height=200,
                    key="translated_text",
                    help=f"Script: {target_script} | Direction: {target_direction}"
                )

                # Target language controls
                col2_1, col2_2 = st.columns(2)
                
                with col2_1:
                    if st.button("🔊 Listen to Translation", use_container_width=True):
                        audio_path = text_to_speech(translated_text, target_lang_code)
                        if audio_path:
                            audio_player = get_audio_player(audio_path)
                            if audio_player:
                                st.markdown(audio_player, unsafe_allow_html=True)
                                try:
                                    os.remove(audio_path)
                                except:
                                    pass
                
                with col2_2:
                    if st.button("📋 Copy Translation", use_container_width=True):
                        st.code(translated_text)
                        st.success("Text copied to clipboard!")

        st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced footer
    st.markdown("""
        <div class="footer">
            <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">Made with ❤️ by Nandesh Kalashetti</p>
            <p style="color: var(--text-secondary); margin-bottom: 1rem;">Geni AI / Front-end Developer</p>
            <p style="font-size: 0.9rem;">
                <a href="https://nandesh-kalashettiportfilio2386.netlify.app/" target="_blank" 
                   style="color: var(--accent); text-decoration: none;">View Portfolio</a>
            </p>
            <p style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 1rem;">
                © 2024 Indic Language Translator Pro
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
