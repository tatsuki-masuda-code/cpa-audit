from . import check_openai_api_key, openai_auth
from . import mk_chromadb
from . import CpaTest
from . import output_metrics
from . import get_logger

import asyncio

def main():
    year_ls = {"co_act": ["H28_1", "H28_2", "H29_1", "H29_2", "H30_1", "H30_2", "H31_1", "H31_2", "R2_1", "R2_2", "R3"],
               "audit": ["H31_1", "H31_2", "R2_1", "R2_2", "R3", "R4_1", "R4_2", "R5_1", "R5_2"],
               }
    model_ls = ["gpt-3.5-turbo-0125", "gpt-4-0613", "gpt-4-turbo-2024-04-09", "gpt-4o-2024-05-13"]
    model_path = "./models"
    subject_ls = ["co_act"]
    is_rag_ls = [False]

    logger = get_logger(dir_path="./log")
    logger.info("Start CPA audit test... (ver.0.1.0)")
    for m in model_ls:
        if m.startswith("gpt") or True in is_rag_ls:
            openai_auth()
            check_openai_api_key()
            if True in is_rag_ls:
                mk_chromadb()
                pass
            break

    cpa = CpaTest(data_path="./data",
                  result_path="./result",
                  system_prompt="与えた文章が正しいか誤っているか判別し、正しければ”True”、誤っていたら”False”を出力しなさい。それ以外には何も含めないことを厳守してください。",
                  pre_question="問題:",
                  post_question="回答:",
                  )

    for is_rag in is_rag_ls:
        for subject in subject_ls:
            for main_model_name in model_ls:
                task = asyncio.run(
                    cpa.inference(subject=subject,
                                year_set=year_ls[subject],
                                llama_model_path=model_path,
                                main_model_name=main_model_name,
                                is_rag=is_rag,
                                )
                )
            output_metrics(result_path="./result",
                subject=subject,
                year_set=year_ls[subject],
                model_ls=model_ls,
                is_rag=is_rag,
                )
    logger.info(f"Finished CPA test.")

if __name__=="__main__":
    main()