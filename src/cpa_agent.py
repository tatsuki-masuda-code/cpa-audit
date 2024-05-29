import pandas  as pd
from datetime import datetime
import openai
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

from pdf2chroma import load_chromadb
from util import gen_questions, calc_token_tiktoken

class CpaAgent:
  """CpaAgent

  This class is a CPA agent which is used to generate questions and answers for the CPA exam.

  Attributes:
      DATA_PATH(str): path to the folder where the cpa data is stored
      RAG_PATH(str): path to the folder where the chroma db is stored
      RESULT_PATH(str): path to the folder where the result is stored
      SYSTEM_PROMPT(str): system prompt
      PRE_PROMPT(str): prompt which is given before the question
      POST_PROMPT(str): prompt which is given after the question
      MAX_TOKENS(int): maximum number of tokens one llm can provide
      agent: agent which is used to generate questions and answers
      df_cpa: dataframe which contains the cpa data
  """
  def __init__(self,
               data_path:str, # path to the folder where the cpa data is stored
               rag_path:str, # path to the folder where the chroma db is stored
               result_path:str, # path to the folder where the result is stored
               system_prompt:str="########\n指示:\n与えた文章が正しいか誤っているか判別し、正しければ「〇」、誤っていたら「×」を出力しなさい。それ以外には何も含めないことを厳守してください。", # system prompt
               pre_prompt:str="################\n問題:\n", # prompt which is given before the question
               post_prompt:str="""回答:""", # prompt which is given after the question
               max_tokens:int=500 # maximum number of tokens
               ):

    self.DATA_PATH = data_path
    self.RAG_PATH = rag_path
    self.RESULT_PATH = result_path
    self.SYSTEM_PROMPT = system_prompt
    self.PRE_PROMPT=pre_prompt
    self.POST_PROMPT=post_prompt
    self.MAX_TOKENS=max_tokens

  def _load_df(self):
    df_cpa = pd.read_csv(f"{self.DATA_PATH}/CPA_AUDIT.csv")
    df_cpa = df_cpa.rename(columns={1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6"})
    df_cpa = df_cpa[df_cpa["abnormal_flg"]==0]
    return df_cpa.copy()

  def _set_agent(self,
                 main_model_name:str,
                 sub_model_name:str
                 ) -> None:
    """_set_agent

    This function sets the agent which is used to generate questions and answers.

    Args:
        main_model_name(str): main model name
        sub_model_name(str): sub model name
    """
    #sub agent
    #make chromadb(self.RAG_PATH)
    tools = load_chromadb(self.RAG_PATH,
                          sub_model_name,
                          max_tokens=self.MAX_TOKENS)

    #main agent
    llm = ChatOpenAI(
        temperature=0,
        model= main_model_name,
        max_tokens=self.MAX_TOKENS
    )

    self.agent = initialize_agent(
        agent=AgentType.OPENAI_FUNCTIONS,
        tools=tools,
        llm=llm,
        verbose=False,
        max_tokens=self.MAX_TOKENS
    )
    return None

  def _infer(self,
             df:pd.DataFrame,
             year:str,
             model_name:str,
             is_rag:bool
             )->None:
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
        query = self.PRE_PROMPT + q + self.POST_PROMPT
        logger.write(query)
        try:
          if not is_rag:
            response = openai.ChatCompletion.create(
              model=model_name,
              messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
              temperature=0
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
    logger.write("##########\nTotal Tokens(Main Model): " + str(cum_token))
    logger.close()

  def start_inference(self,
            year_set:list | set | tuple,
            main_model_name:str,
            is_rag:bool,
            sub_model_name:str
            )->None:
    self.df_cpa = self._load_df()
    if is_rag:
      self._set_agent(main_model_name, sub_model_name)
    for year in year_set:
      self._infer(self.df_cpa, year, main_model_name, is_rag)

