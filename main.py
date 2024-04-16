

from authentication import check_openai_api_key, openai_auth
from pdf2chroma import mk_chromadb
from cpa_agent import CpaAgent
from eval_result import output_metrics

if __name__ == "__main__":
    
    main_model_name = "gpt-3.5-turbo-0125"
    sub_model_name = "gpt-3.5-turbo-0125"
    year_ls = ["H31-1", "H31-2", "R2-1", "R2-2", "R3", "R4-1", "R4-2", "R5-1", "R5-2"]
    is_rag = True

    openai_auth()
    check_openai_api_key()
    mk_chromadb()
    cpa = CpaAgent(data_path="./data", rag_path="./vectorstore_agents", result_path="./result")
    cpa.start(year_set=year_ls, main_model_name=main_model_name, is_rag=is_rag, sub_model_name=sub_model_name)
    output_metrics(DATA_PATH="./data",RESULT_PATH="./result", year_set=["R3"], model_name="gpt-3.5-turbo-0125", is_rag=True)