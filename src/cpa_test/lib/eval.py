from logging import getLogger
logger = getLogger(__name__)

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

def get_metrics(RESULT_PATH:str,
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
        RESULT_PATH (str): path of result
        subject (str): subject (like audit, co_act,...)
        year (str): year(ex. R2-1)
        model_name (str): model name
        is_rag (bool): using RAG or not
        metrics_name (str, optional): metrics name. Defaults to "weighted avg".

    Returns:
        set: metrics_dic
    """
    df_res = pd.read_csv(f"{RESULT_PATH}/csv/{subject}_{year}_{model_name}_rag_{is_rag}.csv")
    df_res["result"] = df_res["result"].astype(str)
    df_res["answer"] = df_res["answer"].astype(str)
    df_res = df_res[df_res["result"].isin(["True", "False"])] # remove the rows which are not answered in True or False
    acc = accuracy_score(df_res["answer"], df_res["result"])
    tn, fp, fn, tp = confusion_matrix(df_res["answer"],
                          df_res["result"],
                          labels=["False", "True"]
                          ).flatten()
    pre, rec, f1, _= precision_recall_fscore_support(df_res["answer"],
                                           df_res["result"],
                                           beta=1,
                                           labels=["False", "True"],
                                           pos_label="True",
                                           average="binary")
    sup = len(df_res)
    metrics_dic = {"TN":tn, "FP":fp, "FN":fn, "TP":tp, "accuracy":acc, "precision":pre, "recall":rec, "f1-score":f1, "support":sup}
    return metrics_dic

def output_metrics(result_path: str,
                    subject:str,
                    year_set: list | set | tuple,
                    model_ls: list | set | tuple,
                    is_rag: bool
                    )->pd.DataFrame:
    """output_metrics

    This function outputs the metrics of the model
    and dump it to the result path.

    Args:
        result_path (str): path of result
        year_set (list | set | tuple): year set
        model_ls (list | set | tuple): model lists included in the output
        is_rag (bool): using RAG or not
    """

    metrics_list = [
        {"model_name": model_name, "year":year, **get_metrics(result_path, subject, year, model_name, is_rag)}
        for model_name in model_ls for year in year_set
    ]
    eval_df = pd.DataFrame(metrics_list).reindex(columns=["model_name", "year", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1-score", "support"])
    eval_df.to_csv(f"{result_path}/summary/summary_{subject}_rag_{is_rag}.csv")
    return eval_df