from tqdm import tqdm

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def mk_chromadb():
    """mk_chromadb function

    This function generates a Chroma database from the PDF files in the PDF_PATH directory.
    
    Args:

    Returns:
    
    """
    PATH = __file__.replace("pdf2chroma.py", "")
    PDF_PATH = f"{PATH}/reports"
    pdf_files = [{"name":f[:-4], "path":f"{PDF_PATH}/{f}"} for f in os.listdir(PDF_PATH)]


    embeddings = OpenAIEmbeddings() 

    for f in tqdm(pdf_files, total=len(pdf_files)):
        loader = PyPDFLoader(f["path"])
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
        )
        texts = text_splitter.split_documents(pages)
        db_tmp = Chroma.from_documents(texts, embeddings, persist_directory=f"{PATH}/vectorstore_agents/{f['name']}")
        db_tmp.persist()

def load_chromadb(model_name:str="gpt-3.5-turbo-16k"):
    """load_chromadb function

    This function loads the Chroma database from the vectorstore_agents directory.
    
    Args:

    Returns:
        dbs(list): list of Chroma databases.
    """
    PATH = __file__.replace("pdf2chroma.py", "")
    pdf_files = [{"name":f[:-4], "path":f"{PDF_PATH}/{f}"} for f in os.listdir(PDF_PATH)]


    
    embeddings = OpenAIEmbeddings() 

    # number in filename must be the full-width
    num_converter = str.maketrans({"0":"０", "1":"１", "2":"２", "3":"３", "4":"４", "5":"５", "6":"６", "7":"７", "8":"８", "9":"９"})
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo-16k"
        ,max_tokens=100
    )

    tools = []
    for i, f in tqdm(enumerate(pdf_files), total=len(pdf_files)):
        db_tmp=Chroma(persist_directory=f"{PATH}/vectorstore_agents/{f['name']}", 
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