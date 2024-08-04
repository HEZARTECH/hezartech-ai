#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: database.py
@author: Yiğit GÜMÜŞ
@date: 2024-08-04 17:26:15
"""
import pymongo as pg
from bson.objectid import ObjectId

__import__('dotenv').load_dotenv()

import certifi
import os

from util import Request, Response

client = pg.MongoClient(os.getenv('DB_URI') + '?retryWrites=true&w=majority', tlsCAFile=certifi.where())
db = client.get_database('hezartech')

def save_prompt_to_db(_input: Request, _response: Response):
    _dict_input = _input.model_dump()
    _dict_response = _response.model_dump()

