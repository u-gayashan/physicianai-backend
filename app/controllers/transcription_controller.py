from fastapi import APIRouter, File, UploadFile
from services import transcription_service
# , word_count_service

router = APIRouter()

@router.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    transcription = await transcription_service.transcribe(audio_file.file)
    # word_count = len(transcription.split())
    # await word_count_service.save_word_count(word_count)
    return {"transcription": transcription}
