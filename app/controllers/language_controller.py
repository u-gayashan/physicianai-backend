from fastapi import APIRouter,FastAPI, File, UploadFile,Form
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse, HTMLResponse
from gtts import gTTS
import googletrans
from googletrans import Translator
from services.language_service import LanguageService

router = APIRouter()

@router.post("/tts")
def text_to_speech(text: str = Form(...), language: str = Form(...)):
    audio_file = "../media/text_to_speech.wav"
    tts = gTTS(text, lang=language)
    tts.save(audio_file)
    return FileResponse(audio_file, media_type='audio/mpeg')

@router.post("/translate/")
async def translate(text: str, dest: str, src: str):
    try:
        translated_text = Translator.translate(text, dest=dest, src=src).text
        return {"translated_text": translated_text}
    except Exception as e:
        return {"error": str(e)}

@router.post("/stt/")
async def speech_to_text(file: UploadFile = File(...)):
    speech_to_text_service = LanguageService()
    return speech_to_text_service.process_audio(file)

@router.get("/supported_languages/")
async def speech_to_text():
    languageService = LanguageService()
    return languageService.getSupportedLanguages()