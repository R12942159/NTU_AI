# Final Project: Taipei City Health Department Project

## Requirements
環境安裝
首先確保 `python` 版本為 `3.13.2`，之後執行以下命令
```
pip install -r requirements.txt
```
並且進入 `source_code` 這個資料夾，以下程式都在此資料夾底下執行
## 資料庫處理
首先確認 `法規及案例 Vector Stores` 這個資料夾放在 `source_code` 資料夾中。
1. 資料庫處理是使用 `Knowledge Base.ipynb` 可使用 `VSCode` 中安裝 `Jupyter` 的 extension 後打開。
2. 點選上方的 `全部執行`。
2. 清理過後的 query 會輸出成 `Knowledge-Base.pkl`，放置於同樣的資料夾下。

## 資料清洗
資料清洗會使用 ChatGpt 進行資料的萃取。因此要輸入 OpenAI API 金鑰
1. 在 terminal 中設定你的 OpenAI API 金鑰：
```
export OPENAI_API_KEY="sk-proj-xxx"
```
2. 先將包含 query 的文件改名成 `final_project_query.csv` 後，放置於 `source_code` 資料夾中。
3. 執行以下命令
    ```
    python clean_data.py
    ```
4. 清理過後的 query 會輸出成 `cleaned_final_project_query.csv`。
### Note 
由於 ChatGPT 產生的結果有隨機性，我們也附上了我們所生成的檔案，因此可以跳過資料清洗的步驟直接使用我們的 `cleaned_final_project_query.csv`。
若要執行自己的資料清洗，麻煩先將我們清理後的 csv 檔改名，以免覆蓋。

## 執行主要程式
1. 請確認在 terminal 中已經設定好你的 OpenAI API 金鑰：
2. 接著就可以執行以下命令
    ```
    python main.py --query-csv-path "your query csv path after data cleaning" \
                    --output-csv-path "output.csv" \
                    --output-info-csv-path "output_info.csv" \
                    --top-k 3 \
                    --top-k-bm25 1 \
                    --use-rag 1 \
                    --use-ensemble 1 \
                    --print-response 0
    ```
參數說明：
- `query-csv-path`: 資料清洗後的查詢 CSV 檔案路徑，默認為 `cleaned_final_project_query` 於同一資料夾下
- `output-csv-path`: 輸出結果的 CSV 檔案路徑
- `output-info-csv-path`: 輸出資訊，如違法內容分析、罰鍰、參考法源等的 CSV 檔案路徑
- `top-k`: retriever 回傳的 top K 答案數量
- `use-ensemble`: 是否使用 ensemble 模型進行回答生成，1 代表使用，0 代表不使用。若使用則是與 BM25 結合
- `top-k-bm25`: 使用 BM25 回傳的 top K 答案數量
- `use-rag`: 是否使用 RAG 模型進行回答生成，1 代表使用，0 代表不使用
- `print-response`: 是否在終端機輸出回答，1 代表輸出，0 代表不輸出

若要復現我們的結果，將清洗後的 csv 命名為 `cleaned_final_project_query.csv`，並將其放在同一資料夾中後，執行
```
bash run.sh
```
最後的結果會是 `output.csv`。並且使用 rag 與沒有使用 rag 的結果會分別存放在 `output_rag.csv` 與 `output_without_rag.csv`。
詳細資訊分別存在 `output_info_rag.csv` 與 `output_info_without_rag.csv`。
