import json
import time
import torch
import evaluate
import requests
import numpy as np

from PIL import Image
from tqdm import tqdm
from evaluate import load
from datasets import load_dataset
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM, GenerationConfig


# Define models and processors
custom_cache_dir = "/home/r12942159/data_18TB"
blip_model_name = "Salesforce/blip-image-captioning-base"
phi4_model_name = "microsoft/Phi-4-multimodal-instruct"

blip_processor = BlipProcessor.from_pretrained(blip_model_name, cache_dir=custom_cache_dir)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name, cache_dir=custom_cache_dir).to("cuda")

phi4_processor = AutoProcessor.from_pretrained(phi4_model_name, trust_remote_code=True, cache_dir=custom_cache_dir)
phi4_model = AutoModelForCausalLM.from_pretrained(
    phi4_model_name, 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True,
    _attn_implementation='flash_attention_2',
    cache_dir=custom_cache_dir,
).cuda()

# Load datasets
datasets_cache_dir = '/home/r12942159/data_18TB/datasets'
datasets = {
    "MSCOCO-Test": load_dataset("nlphuji/mscoco_2014_5k_test_image_text_retrieval",
                                cache_dir=datasets_cache_dir),
    "Flickr30k": load_dataset("nlphuji/flickr30k",
                              cache_dir=datasets_cache_dir)
}

# Evaluation function
def evaluate_captioning(model, processor, image, model_name):
    if model_name == "BLIP":
        inputs = processor(images=image, return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs)
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
    elif model_name == "Phi-4":
        generation_config = GenerationConfig.from_pretrained(phi4_model_name)
        prompt = ["<|user|><|image_1|>Describe the image in detail.<|end|><|assistant|>" for _ in range(len(image))]
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
    
    return caption

# Compute evaluation metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge", rouge_types=["rouge1", "rouge2"])
meteor = evaluate.load("meteor")

def compute_metrics(references, prediction):
    smoothie = SmoothingFunction().method1
    bleu_score = sentence_bleu([ref.split() for ref in references], prediction.split(), smoothing_function=smoothie)

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)
    rouge_scores = [rouge.score(ref, prediction) for ref in references]

    best_rouge1 = max(score["rouge1"].fmeasure for score in rouge_scores)
    best_rouge2 = max(score["rouge2"].fmeasure for score in rouge_scores)

    meteor_score = meteor.compute(predictions=[prediction], references=[references])["meteor"]

    return {
        "BLEU": bleu_score,
        "ROUGE-1": best_rouge1,
        "ROUGE-2": best_rouge2,
        "METEOR": float(meteor_score),
    }

# Process images from datasets
results = {}
batch_size = 8

for dataset_name, dataset in datasets.items():
    results[dataset_name] = {}

    if dataset_name == "MSCOCO-Test":
        img_id = "cocoid"
    else:
        img_id = "img_id"

    dataset_samples = dataset['test']
    num_samples = len(dataset_samples)

    elapsed_time = - time.time()

    for i in tqdm(range(0, num_samples, batch_size), desc=f"Processing {dataset_name} in Batches"):
        batch_samples = dataset_samples[i : i + batch_size]  # 取出 batch
        batch_images = [sample for sample in batch_samples['image']]
        batch_gt_captions = [sample for sample in batch_samples["caption"]]
        batch_ids = [sample for sample in batch_samples[img_id]]

        blip_captions = evaluate_captioning(blip_model, blip_processor, batch_images, "BLIP")
        phi4_caption = evaluate_captioning(phi4_model, phi4_processor, batch_images, "Phi-4")

        for j in range(len(batch_ids)):
            results[dataset_name][batch_ids[j]] = {
                "GT": batch_gt_captions[j],
                "BLIP": blip_captions[j],
                "Phi-4": phi4_caption[j],
                "Metrics": {
                    "BLIP": compute_metrics(batch_gt_captions[j], blip_captions[j]),
                    "Phi-4": compute_metrics(batch_gt_captions[j], phi4_caption[j]),
                }
            }

    elapsed_time += time.time()
    print(f"Time taken for {dataset_name}: {elapsed_time:.2f} seconds")

# Compute average metrics
def compute_avg_scores(results, dataset_name):
    metrics = {"BLIP": {"BLEU": [], "ROUGE-1": [], "ROUGE-2": [], "METEOR": []},
               "Phi-4": {"BLEU": [], "ROUGE-1": [], "ROUGE-2": [], "METEOR": []}}
    
    for sample in results.get(dataset_name, {}).values():
        if "Metrics" in sample:
            for model in ["BLIP", "Phi-4"]:
                metrics[model]["BLEU"].append(sample["Metrics"][model]["BLEU"])
                metrics[model]["ROUGE-1"].append(sample["Metrics"][model]["ROUGE-1"])
                metrics[model]["ROUGE-2"].append(sample["Metrics"][model]["ROUGE-2"])
                metrics[model]["METEOR"].append(sample["Metrics"][model]["METEOR"])
    
    avg_metrics = {
        model: {k: np.mean(v) if v else 0 for k, v in metrics[model].items()}
        for model in ["BLIP", "Phi-4"]
    }
    
    return avg_metrics

mscoco_avg = compute_avg_scores(results, "MSCOCO-Test")
flickr_avg = compute_avg_scores(results, "Flickr30k")

print("\tMSCOCO-Test\t\t\tFlickr30k")
print("\tBLEU\tROUGE-1\tROUGE-2\tMETEOR\tBLEU\tROUGE-1\tROUGE-2\tMETEOR")
for model in ["BLIP", "Phi-4"]:
    print(f"{model}\t{mscoco_avg[model]['BLEU']:.4f}\t{mscoco_avg[model]['ROUGE-1']:.4f}\t"
          f"{mscoco_avg[model]['ROUGE-2']:.4f}\t{mscoco_avg[model]['METEOR']:.4f}\t"
          f"{flickr_avg[model]['BLEU']:.4f}\t{flickr_avg[model]['ROUGE-1']:.4f}\t"
          f"{flickr_avg[model]['ROUGE-2']:.4f}\t{flickr_avg[model]['METEOR']:.4f}")