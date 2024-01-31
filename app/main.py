import uvicorn
from fastapi import FastAPI
# from controllers import llm_response_controller
from controllers.llm_response_controller import router as llm_response_router
from controllers.language_controller import router as language_router

app = FastAPI()
app.include_router(llm_response_router)
app.include_router(language_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # Start ngrok tunnel
    # public_url = ngrok.connect(8000)
    # print("Public URL:", public_url)

    # # Run the API server
    # uvicorn.run(app, host="0.0.0.0", port=8000)
