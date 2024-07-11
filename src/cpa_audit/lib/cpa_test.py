from logging import getLogger
logger = getLogger(__name__)

import pandas  as pd
from datetime import datetime

from .agents import LlamaAgent, GPTAgent, GPTRagAgent, LlamaRagAgent
from .util import gen_questions, calc_token_tiktoken, load_qdata

class CpaTest:
  """CpaTest

  This class is a CPA test class which is used to generate questions and answers for the CPA exam.

  Attributes:
      DATA_PATH(str): path to the folder where the cpa data is stored
      RAG_PATH(str): path to the folder where the chroma db is stored
      RESULT_PATH(str): path to the folder where the result is stored
      SYSTEM_PROMPT(str): system prompt
      PRE_QUESTION(str): prompt which is given before the question
      POST_QUESTION(str): prompt which is given after the question
      MAX_TOKENS(int): maximum number of tokens one llm can provide
      agent: agent which is used to generate questions and answers
      df_cpa: dataframe which contains the cpa data
  """
  def __init__(self,
               data_path:str, # path to the folder where the cpa data is stored
               result_path:str, # path to the folder where the result is stored
               rag_path:str=None, # path to the folder where the chroma db is stored
               system_prompt:str="########\n指示:\n与えた文章が正しいか誤っているか判別し、正しければ「〇」、誤っていたら「×」を出力しなさい。それ以外には何も含めないことを厳守してください。", # system prompt
               pre_question:str="################\n問題:\n", # prompt which is given before the question
               post_question:str="""回答:""", # prompt which is given after the question
               max_tokens:int=500, # maximum number of tokens
               ):
    if rag_path is None:
      logger.warning("RAG path is not specified.")
    self.DATA_PATH = data_path
    self.RAG_PATH = rag_path
    self.RESULT_PATH = result_path
    self.SYSTEM_PROMPT = system_prompt
    self.PRE_QUESTION = pre_question
    self.POST_QUESTION = post_question
    self.MAX_TOKENS=max_tokens

  def _set_agent(self, model_path, main_model_name, sub_model_name, is_rag)->None:
    if main_model_name.startswith("gpt"):
      if not is_rag:
        self.agent = GPTAgent(main_model_name, self.SYSTEM_PROMPT)
      else:
        self.agent = GPTRagAgent(main_model_name, sub_model_name, self.SYSTEM_PROMPT, self.MAX_TOKENS, self.RAG_PATH)
    else:
      if not is_rag:
        self.agent = LlamaAgent(model_path, main_model_name, self.SYSTEM_PROMPT)
      else:
        self.agent = LlamaRagAgent(model_path, main_model_name, sub_model_name, self.MAX_TOKENS, self.RAG_PATH)

  def _infer(self,
             df:pd.DataFrame,
             subject:str,
             year:str,
             model_name:str,
             is_rag:bool
             )->None:
    df_prod = df[df["key"] == year].reset_index(drop=True).copy()
    if len(df_prod)==0:
      raise ValueError("No data. Please make sure you specified the right year.")
    ls_res = []
    ls_ans = []
    cum_token = 0
    logger.info(f"start the inference for model: subject:{subject}, {model_name}, year:{year}, rag:{is_rag}...")
    log_file = open(f"{self.RESULT_PATH}/log/{subject}_{year}_{model_name}_rag_{str(is_rag)}.txt", "w", encoding='UTF-8')
    log_file.write(f"model:{model_name}\n")
    log_file.write(f"year:{year}\n")
    log_file.write(f"rag:{is_rag}\n")
    log_file.write(f"count:{df_prod['question'].count()}\n")
    log_file.write(f"{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
    for i, row in df_prod.iterrows():
      logger.info(f"start question:{i}...")
      q_dic = row.to_dict()
      q_a = gen_questions(**q_dic)
      log_file.write(f"q{i}\n")
      for q, a in q_a:
        query = self.PRE_QUESTION + q + self.POST_QUESTION
        log_file.write(query)
        try:
          result = self.agent.run(query)
          ls_res.append(result)
          ls_ans.append(a)
        except NotImplementedError as e:
          raise
        except Exception as e:
          logger.exception(e)
          log_file.write(str(e))
          ls_res.append(str(e))
          ls_ans.append(a)
          log_file.write("skipped due to an error.\n######\n######\n")
          continue
        if self.agent.agent_type == "llama":
          token_num = 0
          #token_num = self.agent.agent.tokenize(query)['input_ids'].shape[1]
        else:
          token_num = calc_token_tiktoken(query, model_name)
        cum_token += token_num
        log_file.write(f'{result}\n正解:{a}\n######\n######\n')
    pd.DataFrame([ls_res, ls_ans], index=["result", "answer"]).T.to_csv(f"{self.RESULT_PATH}/csv/{subject}_{year}_{model_name}_rag_{str(is_rag)}.csv", index=False)
    log_file.write("##########\nTotal Tokens(Main Model): " + str(cum_token))
    log_file.close()
    logger.info(f"Finish the inference for model:subject:{subject}, model:{model_name}, year:{year}, rag:{is_rag}!")

  def inference(self,
            subject,
            year_set:list | set | tuple,
            main_model_name:str,
            is_rag:bool,
            sub_model_name:str=None,
            llama_model_path:str=None,
            )->None:
    if is_rag is True and self.RAG_PATH is None:
      raise ValueError("RAG path is not specified at the initialization of the class.")
    self.df_cpa = load_qdata(self.DATA_PATH, subject)
    self._set_agent(llama_model_path, main_model_name, sub_model_name, is_rag)
    for year in year_set:
      self._infer(self.df_cpa, subject, year, main_model_name, is_rag)

