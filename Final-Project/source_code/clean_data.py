import openai
import os
import time
from tqdm import tqdm
import pandas as pd
import sys

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    sys.exit("❗ Set the OPENAI_API_KEY environment variable first.")

client = openai.OpenAI()

def clean_text_with_gpt(text_to_clean, model_name="gpt-4.1-2025-04-14"):
    """
    使用指定的GPT模型清洗文本。
    移除產地、銷售管道、藥商資訊、品牌(元數據部分)、數量、保存方式等。
    """
    system_prompt = """
    你是一位專業的文本清理助手。
    你的任務是清理產品廣告文本，專注於提取產品本身的宣傳、特性、聲稱功效和描述性內容。
    你需要移除以下類型的資訊：
    - 產地資訊 (例如："產地:台灣", "台灣台北市", "加拿大製造")
    - 銷售管道、購買地點、店家名稱 (例如："POYA寶雅線上買", "Costco 好市多", "屈臣氏網路商店", "momo購物網", "蝦皮購物")
    - 網站連結或指示性文字 (例如："Visit site", "COMPARE PRICES", "PRODUCT DETAILS", "MORE DETAILS", "USER REVIEWS", "ONLINE SHOPS", "Sort Default")
    - 價格資訊 (例如："NT$990.00")
    - 藥商/廠商的詳細名稱、地址、電話 (例如："新普利股份有限公司", "台北市松山區復興北路309號7樓", "02-2717-0877")
    - "品牌"標籤及其後的品牌名 (如果產品描述本身已包含品牌名，則保留描述中的品牌名)
    - 內容物數量、規格、包裝單位、淨重/容量 (例如："30錠/盒", "100g", "每一份0.6公克", "60錠", "15包入")
    - 保存方式、保存期限、有效日期 (例如："常溫", "製造期後1095日內", "999天")
    - 食用方式的詳細步驟 (如果食用方式是產品宣傳的一部分，例如「睡前兩顆，提升睡眠品質」，則保留)

    - 內容量 (例如："每一份0.6公克 (本包裝含30份)")
    - 貨源 (例如："公司貨")
    - 劑型 (例如："錠")
    - 食品添加物名稱
    - 投保產品責任險資訊 (例如："已投保富邦產物產品責任險 保險單號碼:...")
    - 薦證廣告聲明 (例如："薦證廣告 : 無")
    - 其他揭露事項
    - 完整的營養標示數據表格或列表 (例如："熱量 : 2.2大卡", "蛋白質 : 0.1公克")
    - 任何與產品宣稱、特性、功效無直接關聯的元數據或制式條款。

    目標是提取出最能代表該產品“廣告宣稱”的核心文本。
    請只返回清理後的文本，不要添加任何前導詞或解釋。如果原始文本幾乎全為需移除內容，則返回盡可能精簡的核心產品名或空字串。
    """

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"請清理以下廣告文本：\n\n{text_to_clean}"}
            ],
            temperature=0.1
        )
        cleaned_text = response.choices[0].message.content.strip()
        return cleaned_text
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"Error during cleaning: {e}"

sample_text = """
酵素 - Google Shopping 【Simply 新普利】夜間代謝酵素錠(30錠)Visit siteNT$990.00 · POYA寶雅線上買COMPARE PRICESPRODUCT DETAILSForm: TabletSupplement Ingredient: ProbioticsBenefit: Digestive Support, Sleep Support基礎款夜酵素-最適合夜食習慣者 針對加強夜食消化,睡眠與代謝都有幫助 必需胺基酸促進代謝,色胺酸更幫助入睡。 80種蔬果酵素,經3次發酵活性較高。 添加專利益生菌,隔天起床排便順暢。 睡前兩顆,睡眠期間幫助代謝,提升睡眠品質 內含80種蔬果酵素 添加綜合胺基酸,好眠配方幫助入眠 專利益生菌,幫助消化排便更順暢 項目 : 說明 商品名稱 : 《Simply新普利》夜間代謝酵素錠 30錠/盒 品牌 : Simply新普利 數量 : 100g 保存方式 : 常溫 食用方式 : 每日睡前食用2錠,配合開水食用 內容物成份 : 麥芽糊精、鳳梨酵素(鳳梨梗提取物、麥芽糊精)、綜合必須胺基酸(L-白胺酸、L-二胺基己酸、L-異白胺酸、L-α胺基異戊酸、L-蛋胺酸、L-羥丁胺酸、L-苯丙胺酸、L-色胺酸、卵磷脂)、脂肪分解酵素(麥芽糊精、糊精、脂肪酵素)、澱粉分解酵素(麥芽糊精、澱粉酵素、糊精)、專利複合式植物萃取(關華豆、刺槐豆、阿拉伯膠、石榴、β-胡蘿蔔素、束絲藻)、栗子皮萃取、專利益生菌(麥芽糖醇、麥芽糊精、半乳寡糖、普魯蘭膠、葡萄柚果汁粉末、檸檬酸、凝結芽孢桿菌(Bacilluscoagulans)、蘋果酸、葡萄柚香料、副乾酪乳桿菌(Lactobacillus paracasei)、比菲德氏-龍根菌(Bifidobacterium longum)、乳酸、乾酪乳桿菌(Lactobacillus casei) )、L-肉酸、巴西酵素(昆布、裙帶菜、紫菜、香菇、檸檬馬鞭草、苦艾、西番蓮、金盞草、大飛揚草、牛筋草、檸檬草、大茴香、馬黛茶、魚腥草、紫蘇、薰衣草、貓薄荷、洋甘菊、生薑、益母草、芭樂葉、丁香、咸豐草、迷迭香、燕麥片、玉米、大麥、豌豆、紅豆、薏仁、芝麻、刺槐豆、小米、大豆、爆裂玉米、黑麥、黑豆、鷹嘴豆、扁豆、糙米、鳳梨、香蕉、蘋果、木瓜、芭樂、哈密瓜、酪梨、梅子、西印度櫻桃、檸檬、葡萄乾、芒果、楊桃、西瓜、腰果、核桃、奇異果、西洋梨、巴西莓、卡姆果、萊姆、玫瑰果、橘子、樹莓、柳橙、水蜜桃、蘿蔔、高麗菜、蓮藕、牛蒡、菊苣、甘藷、南瓜、木薯、番茄、羽衣甘藍、青椒、紅甜菜、胡蘿蔔、長型南瓜) 膜衣成份: 羥丙基甲基纖維素、二氧化鈦、食用黃色4號、食用紅色6號 內容量 ( g/ml ) : 每一份0.6公克 (本包裝含30份) 貨源 : 公司貨 劑型 : 錠 食品添加物名稱 : 產地(依序填寫國家-縣市-鄉鎮) : 台灣台北市 廠商名稱 : 新普利股份有限公司 廠商電話號碼 : 02-2717-0877 廠商地址 : 台北市松山區復興北路309號7樓 商品有效日期 : 製造期後1095日內 投保產品責任險字號 : 已投保富邦產物產品責任險 保險單號碼:0500字第21AML0000997號 薦證廣告 : 無 其他揭露事項 : 無 食品業者登錄字號 : A-127972230-00000-0 營養標示 每一份量 : 0.6公克 本包裝含 : 30份 每份 每100公克 熱量 : 2.2大卡 大卡 蛋白質 : 0.1公克公克 脂肪 : 0公克公克 飽和脂肪 : 0公克公克 反式脂肪 : 0公克公克 膽固醇(自願標示者) : 0毫克毫克 碳水化合物 : 0.4公克公克 糖 : 0公克公克 膳食纖維(自願標示者) : 0公克公克 鈉 : 0.4毫克 毫克 藥商許可執照: 北府板藥販 字第 623101M316號 藥商名稱 : 遠百企業股份有限公司 藥商地址: 新北市板橋區四川路1段389號1樓 藥商諮詢專線: (02)7703-3955 規格:30錠 產地:台灣 保存期限:999天 貨源:公司貨 劑型:錠
"""

print("--- 原始文本 ---")
print(sample_text)
print("\n--- 清理後的文本 ---")
cleaned_sample = clean_text_with_gpt(sample_text)
print(cleaned_sample)

print("\n\n--- 開始處理 CSV 檔案 ---")
input_csv_path = 'final_project_query.csv'
output_csv_path = 'cleaned_final_project_query.csv'

try:
    df = pd.read_csv(input_csv_path)
    if 'Question' not in df.columns:
        print(f"錯誤：CSV 檔案 '{input_csv_path}' 中找不到 'Question' 欄位。")
    else:
        cleaned_questions = []
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Cleaning CSV"):
            original_question = str(row['Question']) # 確保是字串
            if pd.isna(original_question) or not original_question.strip():
                cleaned_questions.append("") # 如果原始為空或NaN，則清理後也為空
            else:
                cleaned_text = clean_text_with_gpt(original_question)
                cleaned_questions.append(cleaned_text)
                time.sleep(2) # 避免過於頻繁地呼叫API，根據API限制調整

        df['Cleaned_Question'] = cleaned_questions
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"\nCSV 檔案處理完成，已儲存至 '{output_csv_path}'")
        print("\n前5筆清理結果預覽：")
        print(df[['ID', 'Question', 'Cleaned_Question']].head())

except FileNotFoundError:
    print(f"錯誤：找不到檔案 '{input_csv_path}'。請確保檔案已放置於相同資料夾下。")
except Exception as e:
    print(f"處理 CSV 時發生錯誤：{e}")