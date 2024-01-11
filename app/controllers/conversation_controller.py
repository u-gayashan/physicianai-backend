from fastapi import APIRouter, Body
# from services import ConversationService

router = APIRouter()


@router.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello, {name}"}

# @router.post("/conversation")
# async def handle_conversation(text_input: str = Body(...), language: str = Body(...)):
#     response = await ConversationService.respond(text_input, language)
#     return {"response": response}
