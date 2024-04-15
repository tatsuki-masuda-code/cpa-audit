
import re
import getpass
import pandas  as pd
import os
from datetime import datetime


import openai
import tiktoken
import chromadb
import langchain
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

from authentication import check_openai_api_key, openai_auth
from pdf2chroma import mk_chromadb, load_chromadb

def gen_questions_jcpa (**kwargs):
  """mk_question function

    Generate questions given in the dictionary format into a string prompt.  
    This follows real j-cpa exam format which consists of 4 sentences with 6 choices. 

    Args:
        kwargs(Dict): question dictionary. its key must contain question, ア, イ, ウ, エ, 1, 2, 3, 4, 5 and 6.
    
    Returns:
        question(str): prompt
  """
  q_dic = kwargs
  question = f"""{q_dic["question"]}
ア. {q_dic["ア"]}
イ. {q_dic["イ"]}
ウ. {q_dic["ウ"]}
エ. {q_dic["エ"]}
1. {q_dic["1"]} 2. {q_dic["2"]} 3. {q_dic["3"]} 4. {q_dic["4"]} 5. {q_dic["5"]} 6. {q_dic["6"]}"""
  return question

def gen_questions(**kwargs):
  """gen_questions function

    Generate questions given in the dictionary format into a string prompt.
    This is regular True/False question format with the header and the question sentence.

    Args:
        kwargs(Dict): question dictionary. its key must contain question and answer.
    
    Returns:
        question(str): prompt with the header and the question sentence.
        answer(str): answer in 〇 and ×.
  """
  a_dic = {1:["〇", "〇", "×", "×"], 2:["〇", "×", "〇", "×"], 3:["〇", "×", "×", "〇"], 4:["×", "〇", "〇", "×"], 5:["×", "〇", "×", "〇"], 6:["×", "×", "〇", "〇"], "-":["-","-", "-", "-"]}
  q_dic = kwargs
  a_ls = a_dic[int(q_dic["a_no"])] # transform the answer number into the answer list with True/False.
  for s, a in zip(["ア", "イ", "ウ", "エ"], a_ls):
    yield f"{q_dic['question']}\n{q_dic[s]}", a

def calc_token_tiktoken(chat, model_name):
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(chat))
    return num_tokens


class CpaAgent:
  def __init__(self, 
               data_path:str, # path to the folder where the cpa data is stored
               rag_path:str, # path to the folder where the chroma db is stored
               result_path:str, # path to the folder where the result is stored
               pre_prompt:str="################\n問題:\n", # prompt which is given before the question
               post_prompt:str="""########\n指示:\n文章が「〇」か「×」か判別し、その回答と根拠条文番号を教えてください。\n以下のフォーマットで出力しなさい。\n回答:\n根拠条文:\n########""" # prompt which is given after the question
              ):
    self.pre_prompt=pre_prompt
    self.post_prompt=post_prompt
    self.DATA_PATH = data_path
    self.RAG_PATH = rag_path
    self.RESULT_PATH = result_path
    

  def _load_df(self):
    df_cpa = pd.read_csv(f"{self.DATA_PATH}/CPA_AUDIT.csv")
    df_cpa = df_cpa.rename(columns={1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6"})
    df_cpa = df_cpa[df_cpa["abnormal_flg"]==0]
    return df_cpa.copy()

  def _set_agent(self, 
                 main_model_name:str, 
                 sub_model_name
                 ) -> None:
    #sub agent
    #make chromadb(self.RAG_PATH)
    tools = load_chromadb(self.RAG_PATH, sub_model_name)

    #main agent
    llm = ChatOpenAI(
        temperature=0,
        model= main_model_name,
        max_tokens=500
    )

    self.agent = initialize_agent(
        agent=AgentType.OPENAI_FUNCTIONS,
        tools=tools,
        llm=llm,
        verbose=False,
        max_tokens=500
    )
    return None
  
  def _infer(self, df:pd.DataFrame, year:str, model_name:str, is_rag:bool)->None:
    df_prod = df[df["key"] == year].copy()
    ls_res = []
    cum_token = 0
    logger = open(f"{self.RESULT_PATH}/log/{year}_{model_name}_rag_{str(is_rag)}.txt", "w", encoding='UTF-8')
    logger.write(f"""
                model:{model_name}\n
                year:{year}\n
                rag:{is_rag}\n
                count:{df_prod['question'].count()}\n
                {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n
                """)
    print(f"start {model_name}, {year}")
    for i, row in df_prod.iterrows():
      q_dic = row.to_dict()
      questions = gen_questions(**q_dic)
      for q, a in questions:
        query = self.pre_prompt + q + self.post_prompt
        logger.write(query)
        try:
          if not is_rag:
            response = openai.ChatCompletion.create(
              model=model_name,
              messages=[
                    #{"role": "system", "content": ""},
                    {"role": "user", "content": query},
                ]
              )

            result = response['choices'][0]['message']['content']
          else:        
            result = self.agent.run({"input": query})
          ls_res.append(result)
        except Exception as e:
          logger.write(str(e))
          ls_res.append(str(e))
          continue
        token_num = calc_token_tiktoken(query, model_name)
        cum_token += token_num
        logger.write(f'{result}\n正解:{a}\n######\n######\n')
    pd.Series(ls_res).to_csv(f"{self.RESULT_PATH}/csv/{year}_{model_name}_rag_{str(is_rag)}.csv")
    logger.write("##########\nTotal Tokens: " + str(cum_token))
    logger.close()

  def start(self, 
            year_set:set, 
            main_model_name:str, 
            is_rag:bool, 
            sub_model_name:str="gpt-3.5-turbo-0125"
            )->None:
    self.df_cpa = self._load_df()
    if is_rag:
      self._set_agent(main_model_name, sub_model_name)
    for year in year_set:
      self._infer(self.df_cpa, year, main_model_name, is_rag)


if __name__ == "__main__":
    openai_auth()
    check_openai_api_key()
    #mk_chromadb()
    cpa = CpaAgent(data_path="./data", rag_path="./vectorstore_agents", result_path="./result")
    cpa.start(year_set={"R3"}, main_model_name="gpt-3.5-turbo-0125", is_rag=True, sub_model_name="gpt-3.5-turbo-0125")