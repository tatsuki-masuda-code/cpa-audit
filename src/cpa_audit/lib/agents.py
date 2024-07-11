from logging import getLogger
logger = getLogger(__name__)

import openai
from llama_cpp import Llama
from pydantic import BaseModel

from .pdf2chroma import load_chromadb

class BaseAgent(BaseModel):
  agent_type:str
  system_prompt:str=None
  agent:object=None
  main_model_name:str=None

  def run(self, query:str)->str:
    raise NotImplementedError


class GPTAgent(BaseAgent):
  agent_type="gpt"

  def __init__(self, main_model_name:str, system_prompt:str):
    super().__init__()
    self.main_model_name = main_model_name
    self.system_prompt = system_prompt

  def run(self, query:str)->str:
    response = openai.ChatCompletion.create(
      model=self.main_model_name,
      messages=[
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ],
      temperature=0,
      seed=0
      )
    return response['choices'][0]['message']['content']

class LlamaAgent(BaseAgent):
  agent_type="llama"
  def __init__(self, model_path:str, main_model_name:str, system_prompt:str):
    super().__init__()
    self.system_prompt = system_prompt
    self.agent = Llama(
              model_path=model_path + "/" + main_model_name + '/ggml-model-Q4_K_M-v2.gguf',
              chat_format = "llama-2",
              seed=0,
              verbose=False
    )

  def run(self, query:str)->str:
    response = self.agent.create_chat_completion(
      messages=[
        {"role": "system", "content": self.system_prompt},
        {"role": "user", "content": query}
      ],
      seed=0,
      temperature=0
    )
    return response['choices'][0]['message']['content']

class GPTRagAgent(BaseAgent):
  agent_type="gpt"
  def __init__(self, main_model_name:str, sub_model_name:str, system_prompt:str, max_tokens:int, rag_path):
    super().__init__()
    from langchain.agents import initialize_agent
    from langchain.agents import AgentType
    from langchain.chat_models import ChatOpenAI

    DeprecationWarning("This class is deprecated. Use GPTAgent instead.")
    self.system_prompt = system_prompt
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
    return self.agent.run({"input": self.system_prompt + query})

class LlamaRagAgent(BaseAgent):
  agent_type="llama"
  def __init__(*args, **kwargs):
    super().__init__()
    pass