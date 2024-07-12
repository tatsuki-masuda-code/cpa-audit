from logging import getLogger
logger = getLogger(__name__)

import tiktoken
import numpy as np
import pandas as pd


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

def ans_str2ls(x:str)->list[bool]:
   res_ls = [False, False, False, False]
   if "ア" in x: res_ls[0] = True
   if "イ" in x: res_ls[1] = True
   if "ウ" in x: res_ls[2] = True
   if "エ" in x: res_ls[3] = True
   assert sum(res_ls)==2, f"the number of correct answers must be 2. {x}"
   return res_ls

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
  q_dic = kwargs
  opt_ls = [q_dic[str(i)] for i in range(1, 7)]
  if not all(isinstance(x, str) for x in opt_ls): # if type is not string, then there should be 5 choices.
     raise NotImplementedError(f"This function does not support the question with 5 choices.\n {q_dic}")
  a_dic = {i + 1:ans_str2ls(v) for i, v in  enumerate(opt_ls)}
  a_ls = a_dic[int(q_dic["a_no"])] # transform the answer number into the answer list with True/False.
  for s, a in zip(["ア", "イ", "ウ", "エ"], a_ls):
    yield f"{q_dic['question']}\n{q_dic[s]}\n", a

def calc_token_tiktoken(chat, model_name):
    """calc_token_tiktoken function

    This function calculates the number of tokens in the chat.

    Args:
        chat(str): chat history
        model_name(str): model name

    Returns:
        num_tokens(int): number of tokens
    """
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(chat))
    return num_tokens

def load_qdata(DATA_PATH:str, subject:str)->pd.DataFrame:
    if subject=="audit":
      df_cpa = pd.read_csv(f"{DATA_PATH}/CPA_AUDIT.csv")
      df_cpa = df_cpa.rename(columns={1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6"})
      df_cpa = df_cpa[df_cpa["abnormal_flg"]==0]
    elif subject=="co_act":
      df_cpa = pd.read_csv(f"{DATA_PATH}/CPA_CO_ACT.csv")
      df_cpa = df_cpa.rename(columns={1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6"})
      # no abnormal flg
    else:
       raise ValueError("subject must be audit or co_act")
    return df_cpa
