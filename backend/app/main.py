# backend/app/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from contextlib import asynccontextmanager
import os, logging
import tempfile
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from dotenv import load_dotenv
load_dotenv()

from app.graph import build_graph
from app.tools import setup_database


# Configure the logger to show timestamps and severity levels
logging.basicConfig(
    level=logging.INFO, # Change to DEBUG to see more detailed logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- 1. Startup Event ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists("data/hospitals.db"):
        logger.info("Initializing SQLite database...")
        setup_database("data/hospitals.csv") 
    yield

app = FastAPI(title="Loop AI Voice Backend", lifespan=lifespan)

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_graph = build_graph()

# --- 2. The Voice-to-Voice Endpoint ---
@app.post("/chat")
async def chat_endpoint(
    audio_file: UploadFile = File(...), 
    session_id: str = Form("default_user_1")
):
    try:
        # 1. Save incoming browser audio (usually .webm)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_in:
            tmp_in.write(await audio_file.read())
            input_audio_path = tmp_in.name

        # 2. Transcode to pure WAV using pydub
        wav_audio_path = input_audio_path + ".wav"
        audio = AudioSegment.from_file(input_audio_path)
        audio.export(wav_audio_path, format="wav")

        # 3. Speech-to-Text (Now using the clean WAV file)
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_audio_path) as source:
            audio_data = recognizer.record(source)
            user_text = recognizer.recognize_google(audio_data) 
            logger.info(f"User Transcribed: {user_text}")

        # 4. Process with LangGraph & Gemini
        inputs = {"messages": [HumanMessage(content=user_text)]}
        config = {"configurable": {"thread_id": session_id}} 
        result = agent_graph.invoke(inputs, config=config)

        # Extract the raw content
        raw_content = result["messages"][-1].content
        
        # NORMALIZATION: If Gemini returns a list, extract just the text string
        if isinstance(raw_content, list):
            final_message = "".join([block.get("text", "") for block in raw_content if isinstance(block, dict) and "text" in block])
        else:
            final_message = str(raw_content)
            
        logger.info(f"Loop AI: {final_message}")

        # 5. Text-to-Speech (TTS)
        tts = gTTS(text=final_message, lang='en', tld='co.in')
        output_audio_path = tempfile.mktemp(suffix=".mp3")
        tts.save(output_audio_path)

        # 6. Clean up temp files
        os.remove(input_audio_path)
        os.remove(wav_audio_path)
        
        return FileResponse(
            path=output_audio_path, 
            media_type="audio/mpeg", 
            filename="response.mp3"
        )
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))  
#njnf