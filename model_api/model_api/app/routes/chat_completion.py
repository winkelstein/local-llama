from fastapi import APIRouter
from ..llama import llama
from pydantic import BaseModel
from typing import Literal, List, Dict

router = APIRouter()


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Messages(BaseModel):
    messages: List[Message]


@router.post("/chat-completion/")
async def chat_completion_route(messages: Messages) -> Dict[str, str]:
    response: str = llama.chat_completion(messages.messages)
    return {"response": response}
