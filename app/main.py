import uvicorn
from fastapi import FastAPI
from pyngrok import ngrok
from controllers import conversation_controller, transcription_controller

app = FastAPI()
app.include_router(conversation_controller.router)
app.include_router(transcription_controller.router)

if __name__ == "__main__":
    # Start ngrok tunnel
    public_url = ngrok.connect(8000)
    print("Public URL:", public_url)

    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8000)
