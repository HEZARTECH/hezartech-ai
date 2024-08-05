#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@title: setup.py
@author: Yiğit GÜMÜŞ
@date: 2024-08-05 20:52:26
"""

import os
import uuid

def make_dotenv_mongodb():
    """
    Prompts the user to enter database credentials for MongoDB client and
    creates a .env file with the provided information.

    This function prompts the user to enter the following information:

    - Database URL (if not provided, default is connected to our project)
    - Is Dev (yes/no/1/y/n/true/false/t/f)
    - Custom path for save .env file

    It then generates a random UUID and creates a .env file in the specified
    path with the provided information.

    Returns:
        str: The contents of the .env file. (Optional you dont require to get return data)

    Example:
        make_dotenv_mongodb()
    """
    SECRET_KEY: str = str(uuid.uuid4())
    IS_DEV: str | int = input('[Input Options: Positive = (y/Y/yes/Yes/YES/true/True/TRUE,1), Negative = (n/N/no/No/NO/false/False/FALSE,0)]\n\
Is this a development environment (if you don\'t know what to do, press to \'1\' and enter): ')

    if IS_DEV.lower() in ['y', 'yes', 'true', 't', ''] or int(IS_DEV) == 1:
        IS_DEV = True
    else:
        IS_DEV = False

    if os.name == 'nt':
        slash = '\\'
    else:
        slash = '/'


    DB_URI: str = input('(if not provided a MongoDB connection url, default is connected to our public project database, just press enter)\n\
Database URL: ')

    if DB_URI == '':
        import requests

        '''
            You can make a curl request to:
                https://hezartech-db-url-provider-server.vercel.app/

            Code:
                curl -X GET https://hezartech-db-url-provider-server.vercel.app/
        '''
        response = requests.post(
            'https://hezartech-db-url-provider-server.vercel.app/api/get-db-url',
            headers={
                'Content-Type': 'application/json'
            },
            json={
                "username": "misafir",
                "password": "1234five:)"
            }).json()

        DB_URI = response['url']
        user_cred = response['website_user_credentials']
        print("""
Database URL: %s
----
This is a informative message.

{
    website_user_credentials:  {
        username: '%s',
        password: '%s'
    },
    message: '%s',
    note: '%s'
}
""" % (DB_URI, user_cred['username'], user_cred['password'], response['message'], response['note']))

    _path = os.path.dirname(os.path.abspath(__file__))
    with open(_path + f'{slash}src{slash}.env', 'w') as file:
        file.write(f"{DB_URI=}\n{SECRET_KEY=}\n{IS_DEV=}")

    with open(_path + f'{slash}.env', 'w') as file:
        file.write(f"{DB_URI=}\n{SECRET_KEY=}\n{IS_DEV=}")


    return f"{DB_URI=}\n{SECRET_KEY=}\n{IS_DEV=}"

if __name__ == '__main__':
    make_dotenv_mongodb()

    listdir: str = os.listdir()
    current_path: str = os.getcwd()

    slash = '/'
    if os.name == 'nt':
        slash = '\\'

    current_path = current_path.replace('\\', '\\\\')
    if "src" in listdir:
        with open('.env.shared', 'w+') as file:
            file.write(f'APP_PATH={current_path+slash*2}src')



