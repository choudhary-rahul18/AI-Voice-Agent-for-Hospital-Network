# backend/app/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from contextlib import asynccontextmanager
from twilio.twiml.voice_response import VoiceResponse, Gather, Dial
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
    level=logging.INFO, 
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

# ==========================================
# 🌐 PIPELINE 1: BROWSER WEB UI ENDPOINT
# ==========================================
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

        # 3. Speech-to-Text
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_audio_path) as source:
            audio_data = recognizer.record(source)
            user_text = recognizer.recognize_google(audio_data) 
            logger.info(f"User Transcribed: {user_text}")

        # 4. Process with LangGraph & Gemini
        inputs = {"messages": [HumanMessage(content=user_text)]}
        config = {"configurable": {"thread_id": session_id}} 
        result = agent_graph.invoke(inputs, config=config)

        # Extract the raw content safely
        raw_content = result["messages"][-1].content
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


# ==========================================
# 📞 PIPELINE 2: TWILIO TELEPHONY ENDPOINTS
# ==========================================
@app.post("/voice")
async def handle_incoming_call():
    """Triggered by Twilio the moment someone calls the phone number."""
    logger.info("\n[TELEPHONY] 📞 Incoming call received!")
    response = VoiceResponse()
    
    # timeout=5: Waits 5 seconds for you to say your first word
    # speechTimeout=2: Waits 2 full seconds of silence before cutting you off
    gather = Gather(input='speech', action='/process_speech', timeout=5, speechTimeout=2)
    gather.say("Hello! I am Loop AI. How can I help you find a hospital today?")
    response.append(gather)
    
    # Fallback if they sit in silence and the Gather times out
    response.say("I didn't hear anything. Please call back when you're ready.")
    response.hangup()
    
    return HTMLResponse(content=str(response), media_type="application/xml")


@app.post("/process_speech")
async def process_speech(request: Request):
    """Triggered by Twilio every time the user finishes a sentence."""
    # Twilio sends data as form fields; we extract the Speech and the unique Call ID
    form_data = await request.form()
    speech_result = form_data.get("SpeechResult", "")
    call_sid = form_data.get("CallSid", "unknown_call")
    
    logger.info(f"\n[TELEPHONY] 🗣️ User Transcribed: {speech_result}")
    response = VoiceResponse()
    
    # 1. Handle Silence / Bad Transcriptions
    if not speech_result:
        gather = Gather(input='speech', action='/process_speech', timeout=5, speechTimeout=2)
        gather.say("I'm sorry, I didn't quite catch that. Could you repeat?")
        response.append(gather)
        return HTMLResponse(content=str(response), media_type="application/xml")
        
    # 2. Feed the text to your LangGraph backend!
    # Using CallSid ensures this specific phone call gets its own pagination memory!
    config = {"configurable": {"thread_id": call_sid}}
    inputs = {"messages": [HumanMessage(content=speech_result)]}
    
    try:
        result = agent_graph.invoke(inputs, config=config)
        
        raw_content = result["messages"][-1].content
        if isinstance(raw_content, list):
            ai_message = "".join([block.get("text", "") for block in raw_content if isinstance(block, dict) and "text" in block])
        else:
            ai_message = str(raw_content)
            
        logger.info(f"\n[TELEPHONY] 🤖 Loop AI Output: {ai_message}")
        
        # 3A. The Exact Assignment Requirement (Out of scope -> Hang up)
        if "I am forwarding this to a human agent" in ai_message:
            response.say(ai_message)
            response.hangup() # THIS completely ends the interaction per the assignment
            return HTMLResponse(content=str(response), media_type="application/xml")

        # 3B. The Escape Hatch (User asks to transfer -> Dial)
        elif "HANDOFF" in ai_message or "transfer" in ai_message.lower():
            response.say("I understand. Please hold while I transfer you to a human agent.")
            response.dial("+919876543210") # Your brownie points flex
            return HTMLResponse(content=str(response), media_type="application/xml")
            
        # 4. Speak the AI's response and loop back to listening!
        else:
            gather = Gather(input='speech', action='/process_speech', timeout=5, speechTimeout=2)
            gather.say(ai_message)
            response.append(gather)
            return HTMLResponse(content=str(response), media_type="application/xml")
        
    except Exception as e:
        logger.error(f"\n[TELEPHONY] 🛑 Critical Error: {str(e)}")
        response.say("I'm having trouble connecting to my database right now. Please try again later.")
        response.hangup()

    return HTMLResponse(content=str(response), media_type="application/xml")