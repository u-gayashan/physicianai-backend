from googletrans import LANGUAGES, Translator
import googletrans

class TranslationRepository:
    def __init__(self):
        self.translator = Translator()
        # lan = googletrans.LANGUAGES
        # self.keys = list(lan.keys())
        # self.vals = list(lan.values())

    def translate(self, text):
        translated_text = self.translator.translate(text, dest='en').text
        print(translated_text)
        return translated_text
        # return self.translator.translate(text, dest=language).text
        
    def getSupportedLanguages(self):
        language_codes = list(LANGUAGES.keys())
        return language_codes