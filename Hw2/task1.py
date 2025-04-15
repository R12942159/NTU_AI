from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

LLM_MODEL = ""
EMBEDDINGS = ""

QUERY = "Who is XXX?"
CV_FILE = ""
DB_PATH = ""

# ========== Step 1: build LLM ==========
tokenizer = # AutoTokenizer ... 
model = # AutoModelForCausalLM ...

pipe = # pipeline ...
llm = # HuggingFacePipeline ... 

def wo_RAG():
    print("\n🧪 [只用 LLM 回答]：")
    only_llm_response = llm(QUERY)
    print(only_llm_response)

def w_RAG():
    # ========== Step 2: build knowledge ==========
    loader = PyPDFLoader(CV_FILE)
    pages = loader.load_and_split()

    splitter = # RecursiveCharacterTextSplitter ...
    docs = splitter.split_documents(pages)

    embedding = # HuggingFaceEmbeddings ...
    vectordb = # Chroma.from_documents ...
    retriever = # vectordb.as_retriever ...

    # ========== Step 3: build RAG chain ==========
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain =  # use prompt and llm to build qa chain
    chain = # build a retrieval chain
    result = chain.invoke({"input": QUERY})

    print("\n🧠 [使用 RAG 回答]：")
    print(result['answer'])


if __name__ == '__main__':
    # without RAG
    wo_RAG()
    # with RAG
    w_RAG()