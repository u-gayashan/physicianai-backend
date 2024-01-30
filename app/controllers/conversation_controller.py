from fastapi import APIRouter, Body
from sqlalchemy.orm import Session

# from services import ConversationService

router = APIRouter()


@router.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello, {name}"}

# @router.post("/conversation")
# async def handle_conversation(text_input: str = Body(...), language: str = Body(...)):
#     response = await ConversationService.respond(text_input, language)
#     return {"response": response}


@app.post("/save_chat/")
async def save_chat(user_email, chat_historie):
    with Session(engine) as session:
        chat_object = ChatHistory(user_email=user_email, chat_history=sanitized_chat_history)
        session.add(chat_object)
        session.commit()
        return "Chat history successfully saved!"
