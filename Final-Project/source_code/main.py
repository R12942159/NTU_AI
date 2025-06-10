import os
import re
import csv
import time
import pickle
from typing import List, Dict, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


from openai import RateLimitError
from tqdm import tqdm
import argparse

import os, sys

# === OpenAI API 金鑰（建議移除硬編碼）===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    sys.exit("❗ Set the OPENAI_API_KEY environment variable first.")

LLM_MODEL = "gpt-4.1-2025-04-14"  # 或 "gpt-4o"

# === 配置參數 ===
DB_PATH = "./chroma_db"
KNOWLEDGEBASE_PATH = "Knowledge-Base.pkl"
THRESHOLD = 0.80  # 違法機率大於此值才視為違法
batch_size = 10  # 每次處理的廣告數量

# === LLM 初始化 ===
def get_llm(model_name="gpt-4.1-2025-04-14"):  #"gpt-4o"
    return ChatOpenAI(model=model_name)

# === 初始化 Retriever ===
def get_retriver(top_k, top_k_bm25, use_ensemble) -> EnsembleRetriever:
    EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    embedding = HuggingFaceEmbeddings(model_name=EMB_MODEL)

    with open(KNOWLEDGEBASE_PATH, "rb") as f:
        docs: List[Document] = pickle.load(f)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_documents(docs)

    if os.path.exists(DB_PATH):
        vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embedding)

    else:
        vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=DB_PATH)

    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})

    weights = [1.0]
    retrivers = [retriever]

    if use_ensemble:
        bm25_retriever = BM25Retriever.from_documents(docs, k=top_k_bm25)
        retrivers = [retriever, bm25_retriever]
        weights = [0.75, 0.25]
    return EnsembleRetriever(retrievers=retrivers, weights=weights)

def chat_with_retry(fn, *args, retries=6, base_wait=2, **kwargs):
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except RateLimitError as e:
            if attempt == retries - 1:
                raise                   
            wait_sec = extract_wait_time(str(e)) or base_wait * (2 ** attempt)
            print(f"🔄  Rate-limited，{wait_sec:.1f}s 後重試（第 {attempt+1}/{retries} 次）")
            time.sleep(2*wait_sec)

def extract_wait_time(msg: str) -> Optional[float]:
    m = re.search(r'try again in ([\d.]+)s', msg)
    return float(m.group(1)) if m else None


# === 呼叫 OpenAI GPT 判斷是否合法 ===
def RAG(args, queries: list, retriever):
    query = ""
    context = ""
    idx = 0
    for q in queries:
        query += q

        retriever_docs = retriever.get_relevant_documents(q)
        if args.use_rag:
            context += f"廣告 {idx+1} 相關判決例子：\n"
            for doc in retriever_docs:
                context += doc.page_content + "\n"
            idx += 1

    system_prompt = (
        """
            你是一位專家，請參考以下輸出格式，根據context回答input。

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
                若違法機率 <= 80% \n
                違法機率: X % \n
                違法內容分析: 經分析廣告內容，違法風險較低。 \n
                罰款額度: 無。 \n
                參考依據: [vector store 文件 ID] \n

                若違法機率 > 80% \n
                違法機率: X % \n
                違法內容分析: 段落或句子: [違法描述], [裁罰原因] \n
                違反條款: [適用法規] \n
                罰款額度: [依據法規] \n
                參考依據: [vector store 文件 ID] \n

            可參考之法條:
                健康食品管理法:
                    第14條: "健康食品之標示或廣告，不得涉及醫療效能之內容。"
                    第24條-第1項: "違反第14條規定者，處新臺幣五十萬元以下罰鍰。"
                    第24條-第2項: "違反第14條第2項規定者，處新臺幣四十萬元以上二百萬元以下罰鍰。"
                    第24條-第3項: "前二款之罰鍰，應按次連續處罰至該違規廣告停止刊播為止，並應勒令停止其健康食品之製造、輸入及營業。"

                食品安全衛生管理法:
                    第3條: "食品：指供人食用或飲用之產品及其原料。（定義條款，無罰則）"
                    第28條-第1項: "食品、食品添加物、食品用洗滌劑及經中央主管機關公告之食品器具、食品容器或包裝，其標示、宣傳或廣告不得涉及醫療效能。"
                    第28條-第2項: "食品不得為醫療效能之標示、宣傳或廣告。"
                    第45條-第1項: "違反第28條第1項或中央主管機關依第28條第2項所定基準者，處新臺幣六十萬元以上五百萬元以下罰鍰。"
                    第45條-第2項: "違反前項廣告規定之食品業者，應按次處罰至其停止刊播，並處新臺幣十萬元以下罰鍰。"

                化粧品衛生管理法:
                    第10條: "化粧品之標示、宣傳及廣告內容，不得有虛偽或誇大之情事。"
                    第20條: "違反第10條第1項規定或依第4條所定準則有關宣傳或廣告之內容、方式之規定者，處新臺幣四萬元以上二十萬元以下罰鍰；違反同條第2項規定者，處新臺幣六十萬元以上五百萬元以下罰鍰；情節重大者，並得令其停業及廢止其公司、商業、工廠之全部或部分登記事項。"

                藥事法:
                    第66條: "第1項 藥商刊播藥物廣告時，應於刊播前將所有文字、圖畫或言詞，申請中央或直轄市衛生主管機關核准，未經核准或經核准後廢止者，不得刊播。第2項 藥物廣告在核准登記、刊播期間不得變更原核准事項。第3項 傳播業者不得刊播未經中央或直轄市衛生主管機關核准、與核准事項不符、已廢止或經令立即停止刊播並限期改善而尚未改善之藥物廣告。第4項 受委託刊播之傳播業者，應自廣告之日起六個月，保存委託刊播廣告者之姓名（公司或團體名稱）、身分證或事業登記證字號、住居所（事務所或營業所）及電話等資料，且於主管機關要求提供時，不得規避、妨礙或拒絕。"
                    第68條: "藥物廣告不得以左列方式為之：一、假借他人名義為宣傳者。二、利用書刊資料保證其效能或性能。三、藉採訪或報導為宣傳。四、以其他不正當方式為宣傳。"
                    第69條: ""非本法所稱之藥物，不得為醫療效能之標示或宣傳。"
                    第70條-第2項-併-第92條-第4項: "違反第69條規定者，處新臺幣六十萬元以上二千五百萬元以下罰鍰，其違法物品沒入銷毀；違反第66條第1項、第2項、第67條、第68條規定之一者，處新臺幣二十萬元以上五百萬元以下罰鍰。"

                醫療器材管理法:
                    第6條: "本法所稱醫療器材廣告，指利用傳播方法，宣傳醫療效能，以達招徠銷售醫療器材為目的之行為。採訪、報導或宣傳之內容暗示或影射醫療器材之醫療效能，以達招徠銷售醫療器材為目的者，視為醫療器材廣告。"
                    第40條: "非醫療器材商不得為醫療器材廣告。"
                    第41條-併-第65條-第2項: 第41條第1項規定：醫療器材商刊播醫療器材廣告時，應由許可證所有人或登錄者於刊播前，檢具廣告所有文字、圖畫或言詞，依醫療器材商登記所在地，在直轄市者向直轄市主管機關，在縣（市）者向中央主管機關，申請核准刊播；經核准後，應向傳播業者送驗核准文件，始得刊播。第41條第2項規定：醫療器材廣告於核准刊播期間，不得變更原核准事項而為刊播。第65條第2項第1、2、3款規定：有下列情形之一者，處新臺幣20萬元以上500萬元以下罰鍰：一、違反第40條規定，非醫療器材商為醫療器材廣告。二、違反第41條第1項規定，醫療器材廣告未於刊播前申請核准或向傳播業者送驗核准文件。三、違反第41條第2項規定，醫療器材廣告未經核准擅自變更原核准事項。
                    第46條-併-第65條-第1項: "第46條規定：非醫療器材，不得為醫療效能之標示或宣傳。但其他法律另有規定者，不在此限。第65條第1項規定：違反第46條規定，非醫療器材為醫療效能之標示或宣傳者，處新臺幣60萬元以上2,500萬元以下罰鍰。"

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
                            "content": "違法機率: 10%, 違法內容分析: 經分析廣告內容，違法風險較低。罰款額度: 無。參考依據: 食品及相關產品標示宣傳廣告涉及不實誇張易生誤解或醫療效能認定準則-項次3"
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
                            "content": "違法機率: 90%, 違法內容分析: 段落或句子: '2023營養專家推薦益生菌酵素配方 最佳替代方案胃食道逆流', 由於承諾療效過度具體且容易誤導消費者對於醫療效益的期待。 違反條款: 違反《食品安全衛生管理法》第28條第1項。 罰款額度: 第45條第1項 違反規定者罰四至四百萬。 參考依據: 無。"
                        }}
                    ]
                }}
            }}

        回答以下問題：
        {input}
        請根據以下例句，其中包括違法與合法的例句：
        {context}
        共有十則廣告，請在每一則回答前面加上 "廣告 X：\n"，其中 X 為廣告的序號（1-10）。
        """
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    llm = get_llm(LLM_MODEL)

    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=question_answer_chain)
    rag_response = chat_with_retry(rag_chain.invoke, {"input": query} )

    answer_text = rag_response["answer"]

    if args.print_response:
        print("\n=== Query 內容 ===")
        print(query)
        print("\n=== Retriever 相關內容 ===")
        print(context)
        print("\n=== RAG 回應 ===")
        print(answer_text)

    answer_list = []
    for i in range(1, batch_size + 1):
        idx_1 = answer_text.find(f"廣告 {i}：")
        try:
            idx_2 = answer_text.find(f"廣告 {i+1}：")
            answer = answer_text[idx_1:idx_2].strip()
        except Exception:
            answer = answer_text[idx_1:].strip()
        answer_list.append(answer)

    batch_info = {"Probability": [], "Content analysis": [], "Fine amount": [], "Reference": []}
    prob = []
    for i in range(batch_size):
        idx = answer_list[i].find("違法機率")
        line = answer_list[i][idx:]
        line = line.split("\n")[0].strip()
        try:
            batch_info["Probability"].append(float(line.split(":")[1].strip().rstrip("%")) / 100.0)
        except Exception:
            batch_info["Probability"].append(0.0)
            pass

    for i in range(batch_size):
        idx_1 = answer_list[i].find("違法內容分析")
        idx_2 = answer_list[i].find("罰款額度")
        idx_3 = answer_list[i].find("參考依據")
        try:
            analysis = answer_list[i][idx_1:idx_2].strip()
            analysis = analysis.replace("違法內容分析:", "").strip()
            batch_info["Content analysis"].append(analysis)
        except Exception:
            batch_info["Content analysis"].append("無")
            pass

        try:
            fine_amount = answer_list[i][idx_2:idx_3].strip()
            fine_amount = fine_amount.replace("罰款額度:", "").strip()
            batch_info["Fine amount"].append(fine_amount)
        except Exception:
            batch_info["Fine amount"].append("無")
            pass

        try:
            reference = answer_list[i][idx_3:].strip()
            reference = reference.replace("參考依據:", "").strip()
            batch_info["Reference"].append(reference)
        except Exception:
            batch_info["Reference"].append("無")
            pass

    return batch_info

# === 批次處理 CSV ===
def batch_process_queries(args, retriever):
    results = []
    full_info = []
    try:
        with open(args.query_csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            totalrows = len(rows)

            if args.print_response:
                print("📄 輸入欄位名稱:", reader.fieldnames)
            queries = []
            count = 1
            for idx, row in enumerate(tqdm(rows, total=totalrows)):
                query = row['Cleaned_Question']
                queries.append(f"廣告 {count}：{query}\n")
                count += 1
                if count > batch_size:
                    count = 1
                    batch_info = RAG(args, queries, retriever)
                    idx = idx - batch_size + 1
                    probabilities = batch_info["Probability"]
                    analyses = batch_info["Content analysis"]
                    fine_amounts = batch_info["Fine amount"]
                    references = batch_info["Reference"]

                    for prob, analysis, fine_amount, reference in zip(probabilities, analyses, fine_amounts, references):
                        answer = 0 if prob > THRESHOLD else 1
                        results.append({"ID": idx, "Answer": answer})
                        full_info.append({"ID": idx, "Probability": f"{prob:.2%}", "Content Analysis": analysis, "Fine Amount": fine_amount, "Reference": reference})
                        idx += 1
                    queries = []

                time.sleep(5)
    except Exception as e:
        print("❌ 無法讀取輸入 CSV:", e)
        return
        
    try:
        with open(args.output_csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['ID', 'Answer'])
            writer.writeheader()
            writer.writerows(results)
        print(f"✅ 結果已成功寫入: {args.output_csv_path}")
    except Exception as e:
        print("❌ 寫入 CSV 發生錯誤:", e)

    try:
        with open(args.output_info_csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['ID', 'Probability', 'Content Analysis', 'Fine Amount', 'Reference'])
            writer.writeheader()
            writer.writerows(full_info)
        print(f"✅ 結果已成功寫入: {args.output_info_csv_path}")
    except Exception as e:
        print("❌ 寫入 CSV 發生錯誤:", e)

# === 主程式入口 ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="廣告合法性檢測")
    parser.add_argument("--query-csv-path", type=str, default="cleaned_final_project_query.csv", help="輸入的查詢 CSV 檔案路徑")
    parser.add_argument("--output-csv-path", type=str, default="output.csv", help="輸出的結果 CSV 檔案路徑")
    parser.add_argument("--output-info-csv-path", type=str, default="output_info.csv", help="輸出的詳細資訊 CSV 檔案路徑")
    parser.add_argument("--top-k", type=int, default=3, help="檢索的相關文件數量")
    parser.add_argument("--top-k-bm25", type=int, default=1, help="使用 BM25 檢索的相關文件數量")
    parser.add_argument("--use-rag", type=int, default=1, help="是否使用 RAG 模型 (1: 是, 0: 否)")
    parser.add_argument("--use-ensemble", type=int, default=1, help="是否使用集成檢索器 (1: 是, 0: 否)")
    parser.add_argument("--print-response", type=int, default=0, help="是否打印 RAG 回應 (1: 是, 0: 否)")
    args = parser.parse_args()
    retriever = get_retriver(args.top_k, args.top_k_bm25, args.use_ensemble)

    batch_process_queries(args, retriever)
