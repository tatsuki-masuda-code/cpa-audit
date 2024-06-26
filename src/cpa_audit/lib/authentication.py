from logging import getLogger
logger = getLogger(__name__)

import os
import getpass
import openai
from logging import getLogger
logger = getLogger(__name__)
def openai_auth():
    print("This code requires openai API key.")
    print("The API key you inserted will be saved into the environment variable 'OPENAI_API_KEY'.")
    API_KEY = getpass.getpass(prompt="Insert your api key of openai:")
    os.environ["OPENAI_API_KEY"] = API_KEY
    openai.api_key = API_KEY

def check_openai_api_key():
    openai.Model.list()
    return None