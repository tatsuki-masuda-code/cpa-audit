from authentication import check_openai_api_key, openai_auth
from pdf2chroma import mk_chromadb
from cpa_agent import CpaAgent
from eval_result import output_metrics

if __name__ == "__main__":
    
    sub_model_name = "gpt-3.5-turbo-0125"
    year_ls = ["H31_1", "H31_2", "R2_1", "R2_2", "R3", "R4_1", "R4_2", "R5_1", "R5_2"]
    
    openai_auth()
    check_openai_api_key()
    mk_chromadb()
    for main_model_name in ["gpt-3.5-turbo-0125", "gpt-4-turbo-2024-04-09"]:
        for is_rag in [False, True]: 
            cpa = CpaAgent(data_path="./data", rag_path="./vectorstore_agents", result_path="./result")
            cpa.start(year_set=year_ls, main_model_name=main_model_name, is_rag=is_rag, sub_model_name=sub_model_name)
            output_metrics(data_path="./data",result_path="./result", year_set=year_ls, model_name=main_model_name, is_rag=is_rag)