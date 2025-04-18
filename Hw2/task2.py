import os
import json
import pandas as pd
import torch
import json
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM, GenerationConfig
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


LLM_MODEL = "Qwen/Qwen1.5-7B-Chat"
# LLM_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
PHI4_MODEL = "microsoft/Phi-4-multimodal-instruct"
GEMMA_MODEL = "google/gemma-1.1-7b-it"
EMBEDDINGS = "BAAI/bge-small-en-v1.5"
FILES = "./ntu_hw2_data/AI.pdf"
DB_PATH = "./chroma_db"
SAVE_DIR = "./pdf_images/"
cache_dir = os.path.expanduser("~/data_18TB/")

# ========== Step 1: build LLM ==========
# 加載 tokenizer 並指定 cache_dir
tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL,
    trust_remote_code=True,
    cache_dir=cache_dir,
)

model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    cache_dir=cache_dir,
)

tokenizer.pad_token = tokenizer.eos_token
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.0,           # 穩定輸出
    top_p=1.0,
    repetition_penalty=1.05,   # 輕微防止重複
    pad_token_id=tokenizer.eos_token_id,
    truncation=True,
    do_sample=False,           # 不做 sampling
    max_new_tokens=5,           # 最多產生5 tokens就好（數字只需要這麼短）
)
llm = HuggingFacePipeline(pipeline=pipe)

device = "cuda" if torch.cuda.is_available() else "cpu"
if os.path.exists("captions.json"):
    print("找到 captions.json，正在讀取...")
    with open("captions.json", "r", encoding="utf-8") as f:
        captions = json.load(f)
    print("讀取完成...")
else:
    # blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL, cache_dir=cache_dir)
    # blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL, cache_dir=cache_dir).to(device)
    phi4_processor = AutoProcessor.from_pretrained(PHI4_MODEL, trust_remote_code=True, cache_dir=cache_dir)
    phi4_model = AutoModelForCausalLM.from_pretrained(
        PHI4_MODEL, 
        device_map="cuda", 
        torch_dtype="auto", 
        trust_remote_code=True,
        _attn_implementation='flash_attention_2',
        cache_dir=cache_dir,
    ).to(device)
    # gemma_processor = AutoProcessor.from_pretrained(GEMMA_MODEL, cache_dir=cache_dir)
    # gemma_model = AutoModelForCausalLM.from_pretrained(
    #     GEMMA_MODEL,
    #     device_map="auto",
    #     torch_dtype=torch.bfloat16,
    #     cache_dir=cache_dir,
    # ).to(device)
       
    def evaluate_captioning(model, processor, image, model_name):
        if model_name == "BLIP":
            inputs = processor(images=image, return_tensors="pt", padding=True).to(device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs)
                caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        elif model_name == "Phi-4":
            generation_config = GenerationConfig.from_pretrained(PHI4_MODEL)
            # prompt = ["<|user|><|image_1|>Extract all text you see.<|end|><|assistant|>"]
            prompt = ["<|user|><|image_1|>Describe the concept illustrated in the image. Focus on explaining the visual and logical structure without copying text. Summarize the figure’s meaning and purpose in natural language.<|end|><|assistant|>"]
            inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda:0')

            with torch.no_grad():
                generated_ids = model.generate(
                                **inputs,
                                max_new_tokens=1000,
                                generation_config=generation_config,
                                )
                generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
                caption = processor.batch_decode(generated_ids, 
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False,)
        
        elif model_name == "Gemma":
            prompt = "Provide a thorough caption on the text information shown on this slide, use only what is said on the slide, don't make up bullshit"
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            ).to(model.device)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                generation_output = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
            generation_tokens = generation_output[0][input_len:]
            caption = processor.decode(generation_tokens, skip_special_tokens=True).strip()
            del inputs, generation_output, generation_tokens

        return caption
    
    captions = []
    for idx in tqdm(range(463)):
        img_path = f"{SAVE_DIR}/page_{idx+1}.png"
        raw_image = Image.open(img_path).convert('RGB')

        # blip_captions = evaluate_captioning(blip_model, blip_processor, raw_image, "BLIP")
        phi4_caption = evaluate_captioning(phi4_model, phi4_processor, raw_image, "Phi-4")
        # gemma_caption = evaluate_captioning(gemma_model, gemma_processor, raw_image, "Gemma")

        captions.append({
            "page_label": str(idx+1),
            "caption": phi4_caption,
        })
    
    with open("captions.json", "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=4)

# ========== Step 2: build knowledge ==========
docs = []
for item in captions:
    page_label = item['page_label']
    content = item['caption']
    docs.append(Document(page_content=content[0], metadata={"page_label": page_label}))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 每塊大小
    chunk_overlap=200,    # 重疊部分
)
embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDINGS,
    encode_kwargs={"normalize_embeddings": True},
)
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory=DB_PATH,
)

retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}, # 回傳前5個最相似的 chunk
)

# ========== Step 3: build RAG system ==========
system_prompt = (
    "You are an expert assistant answering questions based on retrieved documents."
    "Each document corresponds to a page from a 463-page textbook."
    "Your goal is to **identify which page** contains the correct answer."
    "Carefully read the provided documents."
    "**Do not guess.**"
    "**Only answer with the page number** (page_label) of the document that contains the most relevant information."
    "If multiple documents are relevant, choose the one that is most complete."
    "If none of the documents answer the question, reply 'Unknown'."

    "Context: {context}"
    "Question: {input}"
    "Your Answer: (only the page number)"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=question_answer_chain)

# ========== Step 4: make presiction ==========
query_df = pd.read_csv("./ntu_hw2_data/HW2_query.csv")
# print(query_df.head())

results = []
for idx, row in tqdm(query_df.iterrows(), total=len(query_df)):
    query = row["Question"]
    result = chain.invoke({"input": query})
    page_number = result['context'][0].metadata['page_label']
    results.append({"ID": idx, "Answer": page_number})

output_df = pd.DataFrame(results)
output_df.to_csv("submission.csv", index=False)
