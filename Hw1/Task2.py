import os
import json
import torch

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from diffusers import StableDiffusion3Pipeline
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


from huggingface_hub import login
login("")

def load_phi4_model(phi4_model_name, custom_cache_dir):
    processor = AutoProcessor.from_pretrained(phi4_model_name, trust_remote_code=True, cache_dir=custom_cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        phi4_model_name, 
        device_map="cuda", 
        torch_dtype="auto", 
        trust_remote_code=True,
        _attn_implementation='flash_attention_2',
        cache_dir=custom_cache_dir,
    ).cuda()

    return processor, model

def read_images(img_dir):
    img_paths = [os.path.join(img_dir, f"{str(i).zfill(6)}.jpg") for i in range(1, 101)]
    images = [Image.open(path) for path in img_paths]

    return images

def generate_text_prompt(image, processor, model, phi4_model_name):
    # instruction = "<|user|><|image_1|>Describe this person in a simple, playful cartoon-style, like Snoopy characters. Keep the person's features but render them in a colorful, exaggerated, and minimalist cartoon style. The character should have the same hair color, expression, and outfit as in the image, but depicted in a Peanuts-like, childlike manner with a simple background. <|end|><|assistant|>"
    instruction = "<|user|><|image_1|>Describe this person in the exact artistic style of Peanuts comics (Snoopy-style). Ensure the description makes the character look like they belong in a Charles Schulz comic strip. The person’s features—such as hair color, expression, and outfit—should remain the same, but they must be transformed into the signature Peanuts cartoon style: simple, bold outlines, flat colors, round heads, dot eyes, and minimal shading. The background should be minimalistic, similar to classic Peanuts comic settings. <|end|><|assistant|>"
    generation_config = GenerationConfig.from_pretrained(phi4_model_name)
    inputs = processor(text=instruction, images=image, return_tensors='pt').to('cuda:0')

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1000,
            generation_config=generation_config,
        )
        generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
        prompt = processor.batch_decode(generated_ids, 
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False,)
    
    return prompt

def generate_text_prompts(images, length):
    prompts = []
    for i in tqdm(range(length)):
        prompt = generate_text_prompt(
            images[i], 
            phi4_processor, 
            phi4_model, 
            "microsoft/Phi-4-multimodal-instruct"
        )
        prompts.append(prompt)
    
    return prompts

def save_prompts2json(prompts, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({"prompts": prompts}, f, indent=4)
    
    print(f"Prompts successfully saved to {filename}")

def load_sd3_pipeline(sd3_model_name, cache_dir):
    pipe = StableDiffusion3Pipeline.from_pretrained(sd3_model_name, 
                                                   cache_dir = cache_dir,
                                                   torch_dtype=torch.float16,
                                                   )
    pipe = pipe.to("cuda")
    
    return pipe

def generate_stylized_image(prompt, sd3_pipe):
    image = sd3_pipe(
        prompt,
        negative_prompt="blurry, distorted, low quality",
        num_inference_steps=32,
        guidance_scale=7.0,
    ).images[0]

    return image

def resize_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).permute(1, 2, 0).mul(255).byte().numpy()

def generate_stylized_images(prompts, sd3_pipe, length):
    images = []
    for i in tqdm(range(length)):
        image = generate_stylized_image(prompts[i], sd3_pipe)
        image = Image.fromarray(resize_image(image))
        images.append(image)
    
    return images

def save_images(stylized_resize_images, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for i, resized_image in enumerate(stylized_resize_images):
        output_filename = f"{i+1:06d}.jpg"
        output_path = os.path.join(output_folder, output_filename)
        resized_image.save(output_path)
        
        print(f"Processed {output_filename}")


phi4_processor, phi4_model = load_phi4_model(phi4_model_name = "microsoft/Phi-4-multimodal-instruct", 
                                             custom_cache_dir = "/home/r12942159/data_18TB")

images = read_images(img_dir = '/home/r12942159/NTU_AI/Hw1/content_image/')

prompts = generate_text_prompts(images, len(images))
save_prompts2json(prompts, 'task2-1_prompts.json')

with open("task2-1_prompts.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)

sd3_pipe = load_sd3_pipeline("stabilityai/stable-diffusion-3-medium-diffusers",
                             cache_dir = "/home/r12942159/data_18TB")

stylized_resize_images = generate_stylized_images(prompts["prompts"], sd3_pipe, len(prompts["prompts"]))
save_images(stylized_resize_images, '/home/r12942159/NTU_AI/Hw1/hw1_r12942159_stylized_images/')
