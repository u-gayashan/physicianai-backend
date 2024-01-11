import openai

model = openai.Model.from_path("models/tiny.en.pt")  # Load Whisper model

async def transcribe(audio_data):
    response = await model.predict(
        input=[{"data": audio_data.read()}],
        return_full_text=True,  # Get full transcription
    )
    transcription = response.choices[0].text.strip()
    return transcription
