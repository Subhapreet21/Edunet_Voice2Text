
import os
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from transformers import pipeline, AutoTokenizer
import tempfile
import nltk
from datetime import datetime
import json

# ---- Force NLTK download ----
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ---- Page Configuration ----
st.set_page_config(
    page_title="Lecture Voice-to-Notes Generator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #4a148c 0%, #1e3c72 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        color: #ffffff;
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .subtitle {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.85;
        color: #e0e0e0;
    }

    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #e0e0e0;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
    }

    .custom-card {
        background: #1e1e1e;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.4);
        border-left: 4px solid #3498db;
    }

    .card-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #f5f5f5;
        margin-bottom: 0.5rem;
    }

    .card-subtitle {
        color: #aaaaaa;
        margin-bottom: 1rem;
    }

    .processing-section {
        background: #2a2a2a;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
    }

    .transcription-text {
        background: #2a2a2a;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #17a2b8;
        font-family: 'Courier New', monospace;
        line-height: 1.6;
        max-height: 300px;
        overflow-y: auto;
        color: #e0e0e0;
    }

    .summary-text {
        background: #3a2f0b;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        line-height: 1.6;
        color: #f5f5f5;
    }

    .flashcard {
        background: #1e1e1e;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease;
    }

    .flashcard:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
    }

    .flashcard-number {
        font-size: 0.9rem;
        color: #bbbbbb;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }

    .flashcard-question {
        background: #102542;
        padding: 0.75rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
        border-left: 3px solid #3498db;
        color: #e0e0e0;
    }

    .flashcard-answer {
        background: #0f3d27;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 3px solid #28a745;
        color: #e0e0e0;
    }

    .footer {
        margin-top: 3rem;
        padding: 2rem 0;
        text-align: center;
        color: #aaaaaa;
        border-top: 1px solid #444;
    }

    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }

        .custom-card {
            margin: 0.5rem 0;
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# ---- State Management ----
for key in ["audio_data", "file_name", "transcription", "summary", "flashcards"]:
    if key not in st.session_state: st.session_state[key] = None

def reset_app():
    # Define the keys you want to clear
    keys_to_clear = ["audio_data", "file_name", "transcription", "summary", "flashcards"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# ---- AI Models (Cached) ----
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="t5-small")
    qg_pipe = pipeline("text2text-generation", model="valhalla/t5-small-qg-hl")
    return summarizer, qg_pipe

@st.cache_resource
def load_whisper():
    import whisper
    return whisper.load_model("base")

# ---- Functions ----
def transcribe_with_whisper(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name
    try:
        model = load_whisper()
        result = model.transcribe(temp_path, fp16=False)
        return result["text"]
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return None
    finally:
        try: os.remove(temp_path)
        except: pass

def generate_summary(text):
    summarizer, _ = load_models()
    try:
        return summarizer(text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
    except Exception as e:
        st.error(f"Summary failed: {e}")
        return None

def generate_flashcards(text):
    _, qg_pipe = load_models()
    sentences = nltk.sent_tokenize(text)
    good_sentences = [s for s in sentences if 10 < len(s.split()) < 50]
    flashcards = []
    for s in good_sentences[:10]:
        try:
            q = qg_pipe(s, max_length=64, do_sample=True)[0]['generated_text']
            flashcards.append({"question": q.strip(), "answer": s.strip()})
        except: continue
    return flashcards

# ---- Main UI ----
st.markdown("""
<div class="main-header">
    <h1>🎓 Lecture Voice-to-Notes Generator</h1>
    <p class="subtitle">Transcribe • Summarize • Study smarter with AI</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.audio_data is None:
    st.markdown('<div class="section-header">📥 Input Your Lecture</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="custom-card"><b>🎤 Live Recording</b><br>Record from your microphone</div>', unsafe_allow_html=True)
        audio = mic_recorder("🎙️ Start Recording", "⏹️ Stop", format="webm", key="recorder")
        if audio and audio['bytes']:
            st.session_state.file_name = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.webm"
            st.session_state.audio_data = audio['bytes']
            st.success("✅ Recording captured!")
            st.rerun()

    with col2:
        st.markdown('<div class="custom-card"><b>📁 File Upload</b><br>WAV, MP3, WEBM, M4A</div>', unsafe_allow_html=True)
        audio_file = st.file_uploader("Upload audio...", type=["wav", "mp3", "webm", "m4a"], key="uploader")
        if audio_file:
            st.session_state.file_name = audio_file.name
            st.session_state.audio_data = audio_file.read()
            st.success("✅ File uploaded!")
            st.rerun()

else:
    st.markdown(f'<div class="section-header">🔍 Analysis for: {st.session_state.file_name}</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1: st.audio(st.session_state.audio_data)
    with col2:
        st.download_button("💾 Download Audio", st.session_state.audio_data, st.session_state.file_name, mime="audio/webm")

    if st.session_state.transcription is None:
        with st.spinner("⏳ Processing audio..."):
            transcription = transcribe_with_whisper(st.session_state.audio_data)
            st.session_state.transcription = transcription
            if transcription:
                st.session_state.summary = generate_summary(transcription)
                st.session_state.flashcards = generate_flashcards(transcription)
                st.balloons()
                st.rerun()
    else:
        # --- Results ---
        with st.expander("📝 Full Transcription"):
            st.markdown(f'<div class="transcription-text">{st.session_state.transcription}</div>', unsafe_allow_html=True)
            st.download_button("📄 Download Transcription", st.session_state.transcription, f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        if st.session_state.summary:
            st.markdown('<div class="custom-card"><b>📋 Summary</b><br>Key lecture points</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-text">{st.session_state.summary}</div>', unsafe_allow_html=True)
            st.download_button("📄 Download Summary", st.session_state.summary, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        if st.session_state.flashcards:
            with st.expander("🎯 Study Flashcards", expanded=True):
                for i, card in enumerate(st.session_state.flashcards, 1):
                    st.markdown(f"""
                    <div class="flashcard">
                        <div><b>Card {i}</b></div>
                        <div class="flashcard-question"><strong>Q:</strong> {card['question']}</div>
                        <div class="flashcard-answer"><strong>A:</strong> {card['answer']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                flashcard_text = "\n\n".join([f"Q: {c['question']}\nA: {c['answer']}" for c in st.session_state.flashcards])
                flashcard_json = json.dumps(st.session_state.flashcards, indent=2)

                st.download_button("📄 Flashcards (TXT)", flashcard_text, f"flashcards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                st.download_button("📚 Flashcards (JSON)", flashcard_json, f"flashcards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        st.markdown("---")
        col1, col2 = st.columns([1,1])
        with col1: st.button("🔄 Start Over", on_click=reset_app)
        with col2:
            if st.button("📊 Export All"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                full_report = f"""LECTURE ANALYSIS REPORT
File: {st.session_state.file_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== TRANSCRIPTION ===
{st.session_state.transcription}

=== SUMMARY ===
{st.session_state.summary or 'No summary'}

=== FLASHCARDS ===
"""
                for i, card in enumerate(st.session_state.flashcards or []):
                    full_report += f"\nCard {i+1}\nQ: {card['question']}\nA: {card['answer']}\n"
                st.download_button("📥 Download Report", full_report, f"lecture_analysis_{timestamp}.txt")

# ---- Footer ----
st.markdown("""
<div class="footer">
    <p>🎓 Powered by Whisper + Hugging Face • Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)

