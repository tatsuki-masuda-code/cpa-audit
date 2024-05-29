import openai
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

from pdf2chroma import load_chromadb

class GPTAgent():
  def __init__(self, model_name:str, system_prompt:str):
    self.model_name = model_name
    self.system_prompt = system_prompt

  def run(self, query:str)->str:
    response = openai.ChatCompletion.create(
      model=self.model_name,
      messages=[
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ],
      temperature=0
      )

    return response['choices'][0]['message']['content']

class RAGAgent():
  def __init__(self, main_model_name:str, sub_model_name:str, max_tokens:int, rag_path):
    tools = load_chromadb(rag_path,
                          sub_model_name,
                          max_tokens=max_tokens)

    #main agent
    llm = ChatOpenAI(
        temperature=0,
        model= main_model_name,
        max_tokens=max_tokens
    )

    self.agent = initialize_agent(
        agent=AgentType.OPENAI_FUNCTIONS,
        tools=tools,
        llm=llm,
        verbose=False,
        max_tokens=max_tokens
    )

  def run(self, query:str)->str:
    self.agent.run({"input": query})
