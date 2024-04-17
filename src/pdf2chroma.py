import os
from tqdm import tqdm

from langchain.agents import Tool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

def mk_chromadb(PDF_PATH:str="./reports"):
    """mk_chromadb function

    This function generates a Chroma database from the PDF files in the PDF_PATH directory.
    
    Args:
        PDF_PATH(str): path to the PDF files.

    Returns:
    
    """
    pdf_files = [{"name":f[:-4], "path":f"{PDF_PATH}/{f}"} for f in os.listdir(PDF_PATH)]


    embeddings = OpenAIEmbeddings() 

    for f in tqdm(pdf_files, total=len(pdf_files)):
        loader = PyPDFLoader(f["path"])
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0, 
            separators=[" ", ",", "\n"] # [todo] optimize the separators for Japanese
            #separators=["第", "."]
        )
        texts = text_splitter.split_documents(pages)
        db_tmp = Chroma.from_documents(texts, embeddings, persist_directory=f"./vectorstore_agents/{f['name']}")
        db_tmp.persist()

def load_chromadb(PDF_PATH:str, model_name:str, max_tokens:int):
    """load_chromadb function

    This function loads the Chroma database from the vectorstore_agents directory.
    
    Args:
        PDF_PATH(str): path to the PDF files.
        model_name(str): model name.
        max_tokens(int): maximum number of tokens. 

    Returns:
        dbs(list): list of Chroma databases.
    """
    
    PDF_PATH = f"./reports"
    pdf_files = [{"name":f[:-4], "path":f"{PDF_PATH}/{f}"} for f in os.listdir(PDF_PATH)]


    
    embeddings = OpenAIEmbeddings() 

    # number in filename must be the full-width
    num_converter = str.maketrans({"0":"０", "1":"１", "2":"２", "3":"３", "4":"４", "5":"５", "6":"６", "7":"７", "8":"８", "9":"９"})
    llm = ChatOpenAI(
        temperature=0,
        model=model_name
        ,max_tokens=max_tokens
    )

    tools = []
    for i, f in tqdm(enumerate(pdf_files), total=len(pdf_files)):
        db_tmp=Chroma(persist_directory=f"./vectorstore_agents/{f['name']}", 
                      embedding_function=embeddings
                      )
        file_name = f["name"].translate(num_converter)
        tools.append(
                Tool(
                    name=i, # [note] maybe file_name?
                    description=f"{file_name}について使用できます。",
                    func=RetrievalQA.from_chain_type(llm=llm, retriever=db_tmp.as_retriever())
                )
        )
    return tools