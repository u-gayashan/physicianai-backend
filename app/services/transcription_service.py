# import numpy as np
# import whisper
# import openai

# model = openai.Model.from_path("models/tiny.en.pt")
# # from whisper import load_model

# # model = whisper.load_model("large")
# # model = openai.Model.from_path("models/tiny.en.pt")  # Load Whisper model

# async def transcribe(audio_data):
#     audio_bytes = audio_data.read()
#     response = await model.predict(
#         input=[{"data": audio_bytes.read()}],
#         return_full_text=True,  # Get full transcription
#     )
#     transcription = response.choices[0].text.strip()
#     return transcription
