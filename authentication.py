
import os
import getpass

def openai_auth():
    print("This code requires openai API key.")
    print("The API key you inserted will be saved into the environment variable 'OPENAI_API_KEY'.")
    print("Insert your api key of openai:")
    API_KEY = getpass.getpass()
    os.environ["OPENAI_API_KEY"] = API_KEY