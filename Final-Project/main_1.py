import os
import csv
import sys
import time
import pickle
from typing import List
from functools import lru_cache
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings


# === OpenAI API 金鑰（建議移除硬編碼）===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    sys.exit("❗ Set the OPENAI_API_KEY environment variable first.")
LLM_MODEL = "gpt-4.1-2025-04-14"  # 或 "gpt-4o"
# LLM_MODEL = "gpt-4-turbo"

# === 配置參數 ===
DB_PATH = "./chroma_db"
KNOWLEDGEBASE_PATH = "Knowledge-Base.pkl"
THRESHOLD = 0.8785  # 違法機率大於此值才視為違法

# === LLM 初始化 ===
def get_llm(model_name="gpt-4.1-2025-04-14"):  #"gpt-4o"
    return ChatOpenAI(model=model_name)

# === 初始化向量資料庫 ===
@lru_cache(maxsize=1)
def get_vectordb() -> Chroma:
    EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    #EMB_MODEL = "intfloat/multilingual-e5-large-instruct"
    embedding = HuggingFaceEmbeddings(model_name=EMB_MODEL)

    if os.path.exists(DB_PATH):
        return Chroma(persist_directory=DB_PATH, embedding_function=embedding)

    with open(KNOWLEDGEBASE_PATH, "rb") as f:
        docs: List[Document] = pickle.load(f)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(docs)

    vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=DB_PATH)
    #vectordb.persist()
    return vectordb

# === 呼叫 OpenAI GPT 判斷是否合法 ===
def RAG(query: str):
    vectordb = get_vectordb()
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.get_relevant_documents(query)

    context = ""
    for doc in retrieved_docs:
        context += doc.page_content + "\n"
    if len(context) > 1500:
        context = context[:1500]

    system_prompt = (
        """
            重要提示：
                你是一位專家，並且只說中文，只要中文回覆!
                請嚴格遵守"斷行"規則與輸出格式！
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
                若違法機率 <= 87.85% \n
                違法機率: X % \n
                違法內容分析: 經分析廣告內容，違法風險較低。 \n
                罰款額度: 無。 \n
                參考依據: [vector store 文件 ID] \n

                若違法機率 > 87.85% \n
                違法機率: X % \n
                違法內容分析: 段落或句子: [違法描述], [裁罰原因] \n
                違反條款: [適用法規] \n
                罰款額度: [依據法規] \n
                參考依據: [vector store 文件 ID] \n

            回應結果範例:
            {{
                "範例1": {{
                    "messages": [
                        {{
                            "role": "user",
                            "content": "請分析廣告內容：'本產品每日至少3~4次可隨餐食用，幫助維持消化道機能' 是否合法？"
                        }},
                        {{
                            "role": "assistant",
                            "content": "違法機率: 10.37%, 違法內容分析: 經分析廣告內容，違法風險較低。罰款額度: 無。參考依據: 食品及相關產品標示宣傳廣告涉及不實誇張易生誤解或醫療效能認定準則-項次3"
                        }}
                    ]
                }},
                "範例2": {{
                    "messages": [
                        {{
                            "role": "user",
                            "content": "請分析廣告內容：'2023營養專家推薦益生菌酵素配方 最佳替代方案胃食道逆流' 是否合法？"
                        }},
                        {{
                            "role": "assistant",
                            "content": "違法機率: 91.05%, 違法內容分析: 段落或句子: '2023營養專家推薦益生菌酵素配方 最佳替代方案胃食道逆流', 由於承諾療效過度具體且容易誤導消費者對於醫療效益的期待。 違反條款: 違反《食品安全衛生管理法》第28條第1項。 罰款額度: 第45條第1項 違反規定者罰四至四百萬。 參考依據: 無。"
                        }}
                    ]
                }}
            }}

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
    print("\n=== RAG 回應 ===")
    # print(answer_text)

    prob = 0.0
    for line in answer_text.splitlines():
        if line.strip().startswith("違法機率"):
            try:
                prob = float(line.split(":")[1].strip().rstrip("%")) / 100.0
                print("Violation probability", prob)
                print("illegal or not:", prob > THRESHOLD)
            except Exception:
                pass
            break
        if line.strip().startswith("2. 違法內容分析"):
            try:
                print("Content in violation:", line.split(":")[1].strip())
            except Exception:
                pass
        if line.strip().startswith("4. 參考依據"):
            try:
                print("Violated provision:", line.split(":")[1].strip())
            except Exception:
                pass

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
            time.sleep(5)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ID', 'ANSWER'])
        writer.writeheader()
        writer.writerows(results)
    print(f"✅ Successfully exported to: {output_path}")

# === 主程式入口 ===
if __name__ == "__main__":
    batch_rag("final_project_query.csv", "prediction.csv")