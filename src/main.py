from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from datetime import datetime, UTC
import certifi
import os
import uuid

import pymongo as pg

__import__('dotenv').load_dotenv()

# other file we are prepared for backend and ml model interaction
# @NOTE: from ai import sentiment_analysis # we use like that in our real code.
# But we can use a placholder like this
def sentiment_analysis(text: str = "Turkcell çok iyi. TurkTelekom tam bir rezaletti. Ama iyi ki değiştirdim.") -> dict[str, list[dict[str,str]]]:
    return dict({
        'entity_list': ['Turkcell', 'TurkTelekom'],
        'results': [
            {
                'entity': 'Turkcell',
                'sentiment': 'Olumlu'
            },
            {
                'entity': 'TurkTelekom',
                'sentiment': 'Olumsuz'
            }
        ]
    })

slash = '/'
if os.name == 'nt':
    slash = '\\'

app = FastAPI(title='HEZARTECH.AI')

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.post('/predict')
async def predict(request: Request):
    try:
        _input = await request.json()

        if "text" not in _input.keys():
            return {
                "success": False,
                "message": "You don't set ``text`` key for your JSON POST Request.",
                "status": 500
            }

        result: dict[str, list[dict[str, str]]] = sentiment_analysis(_input.get('text'))

        is_passed: bool = save_prompt_to_db(_input.get('text'), result)
        if not is_passed:
            print('Cannot save datas to database.')

        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        return JSONResponse(status_code=500, detail=str(e))



@app.post('/result', response_class=HTMLResponse)
async def read_result(request: Request, text: str = Form(...)):
    try:
        return templates.TemplateResponse(
            "result.html", {"request": request, "text": text}
        )
    except Exception as e:
        raise HTTPException(status_code=5000, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    try:
        return templates.TemplateResponse(
            name="index.html", context={"request": request}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# database code
client = pg.MongoClient(os.getenv('DB_URI') + '?retryWrites=true&w=majority', tlsCAFile=certifi.where())
db = client.get_database('hezartech')

def random_id_gen() -> str:
    return str(uuid.uuid4())

def save_prompt_to_db(_input: str, _response: dict[str, list[dict[str, str]]]) -> bool:
    try:
        _time: str = datetime.now(UTC).strftime("%m/%d/%Y, %H:%M:%S")

        db.ai.insert_one({
            "_id": random_id_gen(),
            "text": _input,
            "entity_list": _response.get('entity_list'),
            'results': _response.get('results'),
            "time": _time
        })

        return True

    except Exception as e:
        print("Error at save_prompt_to_db function: " + str(e))
        return False


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host='0.0.0.0', port=1453, reload=True)
