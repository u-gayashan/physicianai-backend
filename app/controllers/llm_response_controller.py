from fastapi import APIRouter, Body

from services.llm_response_service import LLMService
from repositories.llm_response_repository import LLMRepository

router = APIRouter()
llm_repository = LLMRepository()
llm_service = LLMService(llm_repository)

@router.post("/llm_response/")
async def llm_response(chain: str = Body(...), id: int = Body(...), mode: str = Body(...)):
    response = llm_service.get_response(chain, id, mode)
    return {"response": response}
