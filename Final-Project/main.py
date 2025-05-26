import re
import os
import json
import pickle
import csv
from tqdm import tqdm
from typing import List, Dict
from functools import lru_cache
from langchain.schema import Document
from transformers import BitsAndBytesConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login

import warnings
warnings.filterwarnings("ignore", message="`do_sample` is set to `False`.*temperature")
warnings.filterwarnings("ignore", message="`do_sample` is set to `False`.*top_p")

# Login to Hugging Face Hub


LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
cache_dir = os.path.expanduser("~/data_18TB/")

DB_PATH = "./chroma_db"
KNOWLEDGEBASE_PATH = "Knowledge-Base.pkl" 
THRESHOLD = 0.775

# === 初始化向量資料庫 ===
@lru_cache(maxsize=1)
def get_vectordb() -> Chroma:
    embedding = HuggingFaceEmbeddings(model_name=EMB_MODEL)

    if os.path.exists(DB_PATH):
        return Chroma(persist_directory=DB_PATH, embedding_function=embedding)

    with open(KNOWLEDGEBASE_PATH, "rb") as f:
        docs: List[Document] = pickle.load(f)

    vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=DB_PATH)
    vectordb.persist()
    return vectordb

# ========== Step 1: build LLM ==========
@lru_cache(maxsize=1)
def get_llm(llm_model: str) -> HuggingFacePipeline:
    tokenizer = AutoTokenizer.from_pretrained(
        llm_model,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        llm_model,
        device_map="auto",
        quantization_config=bnb_config,
        cache_dir=cache_dir,
    )

    tokenizer.pad_token = tokenizer.eos_token
    gen_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=256,
        do_sample=False,
        # temperature=0.6,
        # top_p=0.90,
    )
    return HuggingFacePipeline(pipeline=gen_pipe)

def RAG(query: str):
    # ========== Step 2: build knowledge ==========
    vectordb  = get_vectordb()
    retriever = vectordb.as_retriever()

    system_prompt = (
        """
            你是一位專家，請嚴格遵守"斷行（使用 "\n" 斷行）"規則與輸出格式！！！
            根據context回答input。
            
            目標：
                分析廣告文字內容，根據法律條款和案例判斷廣告用詞是否涉及誇大療效及違法，並提供違法機率評估。
                回應內容必須完全依照格式，且使用繁體中文。回應簡潔有力，不需要提供分析過程的文字。
            
            合規性判斷：
                - 無罪判定原則：不捏造或過度解讀廣告文字，從寬認定合法性，但如果是"藥物", "宣稱科學實證", "國外研究"一類用語，則提高違法可能性認定，除非內容中出現完整的『衛福部核可字號xxx』或『衛福部認證』。
                - 比對允許使用的廣告用詞："
                - 「完整補充營養」「調整體質」「促進新陳代謝」「幫助入睡」「保護消化道全機能」「改變細菌叢生態」「排便有感」「在嚴謹的營養均衡與熱量控制，以及適當的運動條件下，適量攝取本產品有助於不易形成體脂肪」等用語不視為違法風險。
                - 「能完整補充人體營養」「青春美麗」「排便順暢」…（略）
            
            分析步驟：
                1. 解析廣告內容
                2. 文件檢索與法規比對
                3. 判斷違法機率
                4. 裁罰依據：《食品安全衛生管理法》第45條（罰 4 萬至 400 萬元）
            
            回應格式：
                - 若違法機率 <= 77.3% \n\n
                    1. 違法機率: X % \n\n
                    2. 違法內容分析: 經分析廣告內容，違法風險較低。 \n\n
                    3. 罰款額度: 無。 \n\n
                    4. 參考依據: [vector store 文件 ID] \n\n

                - 若違法機率 > 77.3% \n\n                
                    1. 違法機率: X % \n\n
                    2. 違法內容分析: 段落或句子: [違法描述], [裁罰原因] \n\n
                    3. 違反條款: [適用法規] \n\n
                    4. 罰款額度: [依據法規] \n\n
                    5. 參考依據: [vector store 文件 ID] \n\n
            
            # 回應結果範例:    
            # {{
            #     "範例1": {{
            #         "messages": [
            #             {{
            #                 "role": "user",
            #                 "content": "請分析廣告內容：'本產品每日至少3~4次可隨餐食用，幫助維持消化道機能' 是否合法？"
            #             }},
            #             {{
            #                 "role": "assistant",
            #                 "content": "
            #                 1. 違法機率: 10.7%, \n
            #                 2. 違法內容分析: 經分析廣告內容，違法風險較低。 \n
            #                 3. 罰款額度: 無。 \n
            #                 4. 參考依據: 食品及相關產品標示宣傳廣告涉及不實誇張易生誤解或醫療效能認定準則-項次3 \n
            #                 "
            #             }}
            #         ]
            #     }},
            #     "範例2": {{
            #         "messages": [
            #             {{
            #                 "role": "user",
            #                 "content": "請分析廣告內容：'2023營養專家推薦益生菌酵素配方 最佳替代方案胃食道逆流' 是否合法？"
            #             }},
            #             {{
            #                 "role": "assistant",
            #                 "content": "
            #                 1. 違法機率: 93.2%, \n
            #                 2. 違法內容分析: 段落或句子: '2023營養專家推薦益生菌酵素配方 最佳替代方案胃食道逆流', 由於承諾療效過度具體且容易誤導消費者對於醫療效益的期待。 \n
            #                 3. 違反條款: 違反《食品安全衛生管理法》第28條第1項。 \n
            #                 4. 罰款額度: 第45條第1項 違反規定者罰四至四百萬。 \n
            #                 5. 參考依據: 無。 \n
            #                 "
            #             }}
            #         ]
            #     }}
            # }}

        請根據以下 context：
        {context}
        並回答以下問題：
        {input}
        """
    )


    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    llm = get_llm(LLM_MODEL)

    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=question_answer_chain)
    rag_response = rag_chain.invoke({"input": query})

    answer_text = rag_response["answer"]
    print("\n=============== RAG 回應 ===============")
    # print(answer_text)

    for line in reversed(answer_text.splitlines()):
        print(line)
        if line.strip().startswith("違法內容分析"):
            try:
                print("Content in violation:", line.split(":")[1].strip())
            except Exception:
                pass
        if line.strip().startswith("參考依據"):
            try:
                print("Violated provision:", line.split(":")[1].strip())
            except Exception:
                pass
        if line.strip().startswith("違法機率"):
            try:
                prob = float(line.split(":")[1].strip().replace('%', '').replace(',', '')) / 100.0
                print("Violation probability", prob)
                print("illegal or not:", prob > THRESHOLD)
            except Exception:
                pass
            break

    return 0 if prob > THRESHOLD else 1

# ========== Step 3: batch read ==========
def batch_rag(csv_path: str, output_path: str):
    results = []

    with open(csv_path, 'r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)

        for idx, row in enumerate(reader):
            query = row['Question']
            answer = RAG(query)
            results.append({"ID": idx, "ANSWER": answer})
            print(f"Process the query {idx + 1} => Respond: {answer}")

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ID', 'ANSWER'])
        writer.writeheader()
        writer.writerows(results)
    print(f"✅ Successfully exported to: {output_path}")



if __name__ == "__main__":
    batch_rag("final_project_query.csv", "prediction.csv")