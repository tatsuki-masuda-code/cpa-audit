import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def  load_df(RESULT_PATH:str, year:str, model_name:str, is_rag:bool)->pd.DataFrame:
    df_res = pd.read_csv(f"{RESULT_PATH}/csv/{year}_{model_name}_rag_{is_rag}.csv")
    return df_res

def get_pred(df_res:pd.DataFrame) -> pd.Series:
    df_res["〇"] = df_res["0"].str.contains("〇").map({True:"〇", False:""}) 
    df_res["×"] = df_res["0"].str.contains("×").map({True:"×", False:""})
    return df_res["〇"] + df_res["×"]

def get_ans(DATA_PATH:str, year:str)->np.ndarray:
    a_dic={1:["〇", "〇", "×", "×"], 
           2:["〇", "×", "〇", "×"], 
           3:["〇", "×", "×", "〇"], 
           4:["×", "〇", "〇", "×"], 
           5:["×", "〇", "×", "〇"], 
           6:["×", "×", "〇", "〇"], 
           "-":["-","-", "-", "-"]
           }
    df_cpa = pd.read_csv(f"{DATA_PATH}/CPA_AUDIT.csv")
    ans_mtx = df_cpa[df_cpa.key==year]["a_no"].map(int).map(a_dic).values
    ans_vec = np.stack(ans_mtx).ravel()
    return ans_vec

def get_metrics(DATA_PATH:str, RESULT_PATH:str, year, model_name, is_rag, metrics_name:str="weighted avg")->set:
    df_res = load_df(RESULT_PATH, year, model_name, is_rag)
    df_res["pred"] = get_pred(df_res)
    df_res["ans"] = get_ans(DATA_PATH, year)
    df_res = df_res[df_res["pred"].isin(["〇", "×"])]
    class_report_dic = classification_report(df_res["ans"], df_res["pred"], output_dict=True)
    acc = class_report_dic["accuracy"]
    metrics_dic = class_report_dic[metrics_name]
    metrics_dic["accuracy"] = acc
    return metrics_dic
 
def output_metrics(data_path: str, 
         result_path: str, 
         year_set: list | set | tuple, 
         model_name: str, 
         is_rag: bool
         ):
    eval_df = pd.DataFrame(index=["accuracy", "precision", "recall", "f1-score", "support"])    
    for year in year_set:
        metrics_dic = {}
        metrics_dic = get_metrics(data_path, result_path, year, model_name, is_rag)
        eval_df[year] = metrics_dic
    eval_df = eval_df.T
    eval_df.to_csv(f"{result_path}/csv/summary_{model_name}_rag_{is_rag}.csv")
    return None

#if __name__ == "__main__":
#    output_metrics(DATA_PATH="./data",RESULT_PATH="./result", year_set=["R3"], model_name="gpt-3.5-turbo-0125", is_rag=True)