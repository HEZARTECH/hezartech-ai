from pydantic import BaseModel

class Request(BaseModel):
    text: str

class Response(BaseModel):
    entity_list: list[str]
    results: list[dict[str, str]]
