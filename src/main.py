from fastapi import FastAPI, Request, HTTPException, Form, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel, Field
"""
from datetime import datetime, UTC #(for python3.11+)
"""

# For python 3.10
from datetime import datetime
from datetime import timezone
UTC = timezone.utc


import certifi
import os
import uuid

from ai import (
    make_predict
)

import pymongo as pg

__import__('dotenv').load_dotenv()

def sentiment_analysis(text: str) -> dict[str, list[dict[str,str]]]:
    return make_predict(text)


slash = '/' if os.name != 'nt' else '\\'

app = FastAPI(title='HEZARTECH.AI')

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class TextRequestModel(BaseModel):
    text: str = Field(..., example="Turkcell çok iyi. TurkTelekom tam bir rezalet. İyi ki değiştirmişim.")

class SentimentResponseModel(BaseModel, JSONResponse):
    entity_list: list[str] = Field(..., example=['Turkcell', 'TurkTelekom'])
    results: list[dict[str, str]] = Field(..., example=[
        {'entity': 'Turkcell', 'sentiment': 'Olumlu'},
        {'entity': 'TurkTelekom', 'sentiment': 'Olumsuz'},
    ])
@app.post('/predict', response_model=SentimentResponseModel, summary="Predict sentiment for a given text. (For just backend usage.)", description="Accepts a JSON payload with a `text` field and returns sentiment analysis results.")
async def predict(request: Request):
    '''
        Bu fonksiyon cURL ya da herhangi bir client ile bağlanıp verileri
        json formatında isteyip döndürmek için kullanılıyor.

        Usage:
            curl http://HOST:PORT/predict -H Content-Type:application/json -d "{\"text\": \"$INPUT$\"}
    '''

    try:
        _input = await request.json()

        if "text" not in _input.keys():
            return JSONResponse(status_code=500, content={
                "success": False,
                "message": "You don't set ``text`` key for your JSON POST Request.",
                "status": 500
            })

        result: dict[str, list[dict[str, str]]] = sentiment_analysis(_input.get('text'))
        print(result)
        is_passed: bool = save_prompt_to_db(_input.get('text'), result)
        if not is_passed:
            print('Cannot save datas to database.')

        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content=str(e))

async def result_predict(text: str) -> dict[str, list[dict[str, str]]]:
    '''
        Bu fonksiyon /result endpoint'i için yapay zeka ile haberleşen bir fonksiyondur
    '''

    result = sentiment_analysis(text)
    print(result)
    is_passed: bool = save_prompt_to_db(text, result)

    if not is_passed:
        print('Cannot save datas to database.')

    return result

@app.post('/result', response_class=HTMLResponse, summary="Display the result page.", description="Displays a result page with the provided text from index page form input. Predict the result via our model and print it.")
async def read_result(request: Request, text: str = Form(...)):
    '''
        Bu fonksiyon `/` endpoint'indeki girdiyi yapay zeka ile analiz edip döndürmek için
        kullanılıyor. Aynı zamanda bir arayüz ile tek tek duygu-firma eşleşmelerini görüntülemek için
        kullanılıyor.
    '''
    try:
        results = sentiment_analysis(text)['results']
        print(results)
        return templates.TemplateResponse("result.html", {
            "request": request,
            "text": text,
            "results": results,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse, summary="Home page", description="Displays the home page.")
async def read_index(request: Request):
    '''
        Ana sayfa. (Bir girdi ile sorgu yapmak için kullanılıyor.)
    '''
    try:
        return templates.TemplateResponse(name="index.html", context={"request": request})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Database Code Section For Saving Data to Retrain our model later.
client = pg.MongoClient(os.getenv('DB_URI') + '?retryWrites=true&w=majority', tlsCAFile=certifi.where())
db = client.get_database('hezartech')

def random_id_gen() -> str:
    return str(uuid.uuid4())

def save_prompt_to_db(_input: str, _response: dict[str, list[dict[str, str]]]) -> bool:
    try:
        _time = datetime.now(UTC).strftime("%m/%d/%Y, %H:%M:%S") + 'UTC'
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
    uvicorn.run("main:app", host='0.0.0.0', port=9568, reload=True)
