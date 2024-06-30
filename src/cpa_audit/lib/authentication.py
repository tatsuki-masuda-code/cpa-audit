from logging import getLogger
logger = getLogger(__name__)

import os
import getpass
import openai
from logging import getLogger
logger = getLogger(__name__)
def openai_auth():
    if "OPENAI_API_KEY" in os.environ:
        logger.info("Openai API key is already set.")
        API_KEY = os.environ["OPENAI_API_KEY"]
    else:
        logger.info("This code requires openai API key.")
        logger.info("The API key you inserted will be saved into the environment variable 'OPENAI_API_KEY'.")
        API_KEY = getpass.getpass(prompt="Insert your api key of openai:")
    os.environ["OPENAI_API_KEY"] = API_KEY
    openai.api_key = API_KEY

def check_openai_api_key():
    openai.Model.list()
    return None