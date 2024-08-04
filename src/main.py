from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.encoders import jsonable_encoder

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from util import Request, Response

from ai import sentiment_analysis

app = FastAPI()

app.mount('/static', StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.post('/predict')
async def predict(raw_input: Request):
    _input = raw_input.model_dump()
    if "text" not in _input.keys():

        return {
            "success": False,
            "message": "You don't set text key for your JSON POST Request."
        }


    result = sentiment_analysis(_input.text)
    response = Response(
        entity_list=result.entity_list,
        results=result.results
    )

    return JSONResponse(
        content = jsonable_encoder(response.model_dump())
    )

@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request}
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=1453)
