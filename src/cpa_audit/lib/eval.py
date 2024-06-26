from logging import getLogger
logger = getLogger(__name__)

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from .util import load_qdata

# [todo] merge these two similar functions into one
def load_res(RESULT_PATH:str, subject:str, year:str, model_name:str, is_rag:bool)->pd.DataFrame:
    df_res = pd.read_csv(f"{RESULT_PATH}/csv/{subject}_{year}_{model_name}_rag_{is_rag}.csv")
    return df_res

def get_pred(df_res:pd.DataFrame) -> pd.Series:
    df_res["〇"] = df_res["0"].str.contains("〇").map({True:"〇", False:""})
    df_res["×"] = df_res["0"].str.contains("×").map({True:"×", False:""})
    return df_res["〇"] + df_res["×"]

def get_ans(DATA_PATH:str, subject:str, year:str)->np.ndarray:
    a_dic={"1":["〇", "〇", "×", "×"],
           "2":["〇", "×", "〇", "×"],
           "3":["〇", "×", "×", "〇"],
           "4":["×", "〇", "〇", "×"],
           "5":["×", "〇", "×", "〇"],
           "6":["×", "×", "〇", "〇"],
           "-":["-", "-", "-", "-"]
           }
    df_cpa = load_qdata(DATA_PATH, subject)
    ans_mtx = df_cpa[df_cpa.key==year]["a_no"].map(str).map(a_dic).values
    ans_vec = np.stack(ans_mtx).ravel()
    return ans_vec

def agg_result(RESULT_PATH:str, DATA_PATH:str, subject:str, year:str, model_name:str, is_rag:bool):
    df_res = load_res(RESULT_PATH, subject, year, model_name, is_rag)
    df_res["pred"] = get_pred(df_res)
    df_res["ans"] = get_ans(DATA_PATH, subject, year)
    df_res.to_csv(f"{RESULT_PATH}/agg/{subject}_{year}_{model_name}_rag_{is_rag}_agg.csv")
    return df_res

def get_metrics(DATA_PATH:str,
                RESULT_PATH:str,
                subject:str, 
                year:str,
                model_name:str,
                is_rag:bool,
                metrics_name:str="weighted avg"
                )->set:
    #explain this function
    """get_metrics

    This function calculates the metrics of the model.

    Args:
        DATA_PATH (str): path of data
        RESULT_PATH (str): path of result
        subject (str): subject (like audit, co_act,...)
        year (str): year(ex. R2-1)
        model_name (str): model name
        is_rag (bool): using RAG or not
        metrics_name (str, optional): metrics name. Defaults to "weighted avg".

    Returns:
        set: metrics_dic
    """
    df_res = agg_result(RESULT_PATH, DATA_PATH, subject, year, model_name, is_rag)
    df_res = df_res[df_res["pred"].isin(["〇", "×"])]
    class_report_dic = classification_report(df_res["ans"], df_res["pred"], output_dict=True)
    acc = class_report_dic["accuracy"]
    metrics_dic = class_report_dic[metrics_name]
    metrics_dic["accuracy"] = acc
    return metrics_dic

def output_metrics(data_path: str,
                    result_path: str,
                    subject:str, 
                    year_set: list | set | tuple,
                    model_name: str,
                    is_rag: bool
                    ):
    """output_metrics

    This function outputs the metrics of the model
    and dump it to the result path.

    Args:
        data_path (str): path of data
        result_path (str): path of result
        year_set (list | set | tuple): year set
        model_name (str): model name
        is_rag (bool): using RAG or not
    """

    eval_df = pd.DataFrame(index=["accuracy", "precision", "recall", "f1-score", "support"])
    for year in year_set:
        metrics_dic = {}
        metrics_dic = get_metrics(data_path, result_path, subject, year, model_name, is_rag)
        eval_df[year] = metrics_dic
    eval_df = eval_df.T
    eval_df.to_csv(f"{result_path}/summary/summary_{subject}_{model_name}_rag_{is_rag}.csv")
    return None

#if __name__ == "__main__":
#    output_metrics(data_path="./data",result_path="./result", year_set=["H31_1", "H31_2", "R2_1", "R2_2", "R3", "R4_1", "R4_2", "R5_1", "R5_2"], model_name="gpt-3.5-turbo-0125", is_rag=True)
