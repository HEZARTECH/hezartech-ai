from pydantic import BaseModel

class Request(BaseModel):
    text: str
    time: str

class Response(BaseModel):
    entities: list[str]

