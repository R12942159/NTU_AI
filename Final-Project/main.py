import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

LLM_MODEL = "microsoft/phi-2"
EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"

QUERY = "Who is Liang-Ming Chiu?"
CV_FILE = "./CV-English.pdf"
DB_PATH = "./chroma_cv"

# ========== Step 1: build LLM ==========
# 加載 tokenizer
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype="auto",
    device_map="auto",
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.7,
    top_p=0.85,
    repetition_penalty=1.15,
    max_new_tokens=256,
    pad_token_id=tokenizer.eos_token_id,
    truncation=True,
    do_sample=True,
)
llm = HuggingFacePipeline(pipeline=pipe)

def wo_RAG():
    print("\n🧪 [只用 LLM 回答]：")
    only_llm_response = llm(QUERY)
    print(only_llm_response)

def w_RAG():
    # ========== Step 2: build knowledge ==========
    if os.path.exists(DB_PATH):
        vectordb = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embedding
        )
    else:
        loader = PyPDFLoader(CV_FILE)
        pages = loader.load_and_split()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,      # size of each text chunk
            chunk_overlap=50,    # overlap between chunks
        )
        docs = splitter.split_documents(pages)

        embedding = HuggingFaceEmbeddings(model_name=EMBEDDINGS)
        vectordb = Chroma.from_documents(documents=docs, embedding=embedding)
        vectordb.persist()  

    retriever = vectordb.as_retriever()
    # system_prompt = (
    #     "Use the given context to answer the question. "
    #     "If you don't know the answer, say you don't know. "
    #     "Use three sentences maximum and keep the answer concise. "
    #     "Context: {context}"
    # )
    system_prompt = (
        "Use the provided context to answer the question as accurately as possible. "
        "If the answer is not clear, respond with 'I don't know.' "
        "Please keep your response concise, with a maximum of three sentences, focusing on the most relevant details. "
        "Context: {context}"
    )


    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=question_answer_chain)
    result = chain.invoke({"input": QUERY})

    print("\n🧠 [使用 RAG 回答]：")
    print(result['answer'])


if __name__ == '__main__':
    wo_RAG()
    w_RAG()