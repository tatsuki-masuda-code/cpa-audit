from logging import getLogger
logger = getLogger(__name__)

import tiktoken
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
  a_dic = {1:["〇", "〇", "×", "×"],
           2:["〇", "×", "〇", "×"],
           3:["〇", "×", "×", "〇"],
           4:["×", "〇", "〇", "×"],
           5:["×", "〇", "×", "〇"],
           6:["×", "×", "〇", "〇"],
           "-":["-","-", "-", "-"]
           }
  q_dic = kwargs
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
