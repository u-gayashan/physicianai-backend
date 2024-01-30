from faster_whisper import WhisperModel
from repositories.language_repository import TranslationRepository

class LanguageService:
    def __init__(self):
        self.transcribe_model = WhisperModel("large-v3")
        self.translationRepository = TranslationRepository()


    def process_audio(self, file):
        file_path = "../media/harvard.wav"

        with open(file_path, "wb") as f:
            f.write(file.file.read())

        segments, info = self.transcribe_model.transcribe(file_path, beam_size=5)
        result = "".join(segment.text for segment in segments)

        original_text = {"original_text": result}
        english_text = {"english_text": self.translationRepository.translate(result)}

        return {**original_text, **english_text}
    
    def getSupportedLanguages(self):
        return self.translationRepository.getSupportedLanguages()