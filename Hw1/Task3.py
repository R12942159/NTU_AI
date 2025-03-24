import os
import json
import torch

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from diffusers import StableDiffusionImg2ImgPipeline, AutoPipelineForImage2Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


def read_images(img_dir):
    img_paths = [os.path.join(img_dir, f"{str(i).zfill(6)}.jpg") for i in range(1, 101)]
    images = [Image.open(path) for path in img_paths]

    return images

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

def generate_text_prompt(image, instruction, max_new_tokens, processor, model, phi4_model_name):
    generation_config = GenerationConfig.from_pretrained(phi4_model_name)
    inputs = processor(text=instruction, images=image, return_tensors='pt').to('cuda:0')

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            generation_config=generation_config,
        )
        generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
        prompt = processor.batch_decode(generated_ids, 
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False,)
    
    return prompt

def generate_text_prompts(images, instruction, max_new_tokens, length, processor, model):
    prompts = []
    for i in tqdm(range(length)):
        prompt = generate_text_prompt(
            images[i],
            instruction,
            max_new_tokens, 
            processor, 
            model, 
            "microsoft/Phi-4-multimodal-instruct"
        )
        prompts.append(prompt)
    
    return prompts

def save_prompts2json(prompts, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({"prompts": prompts}, f, indent=4)
    
    print(f"Prompts successfully saved to {filename}")

def load_stable_diffusion(model_id, cache_dir):
    # pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    #     model_id, 
    #     torch_dtype=torch.float16,
    #     cache_dir=cache_dir,
    #     safety_checker = None,
    #     feature_extractor = None,
    # ).to("cuda")

    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=False,
        cache_dir="/home/r12942159/data_18TB",
        safety_checker = None,
        feature_extractor = None,
    )
    pipe.enable_model_cpu_offload()

    return pipe

def read_prompts(prompt_path):
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    return prompts['prompts']

def resize_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).permute(1, 2, 0).mul(255).byte().numpy()

def generate_stylized_images(pipe, prompt, image, strength=0.75, guidance_scale=7.5, seed=41):
    image = Image.fromarray(resize_image(image))
    image = pipe(
        prompt=prompt, 
        image=image, 
        strength=strength, 
        guidance_scale=guidance_scale, 
        generator=torch.manual_seed(seed),
    ).images[0]
    
    return image

def save_images(stylized_resize_images, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for i, resized_image in enumerate(stylized_resize_images):
        output_filename = f"{i+1:06d}.jpg"
        output_path = os.path.join(output_folder, output_filename)
        resized_image.save(output_path)
        
        print(f"Processed {output_filename}")

# Main()
phi4_processor, phi4_model = load_phi4_model(phi4_model_name = "microsoft/Phi-4-multimodal-instruct", 
                                             custom_cache_dir = "/home/r12942159/data_18TB")

instruction = "<|user|><|image_1|>Describe this person in the exact artistic style of Peanuts comics (Snoopy-style). Ensure the description makes the character look like they belong in a Charles Schulz comic strip. The person’s features—such as hair color, expression, and outfit—should remain the same, but they must be transformed into the signature Peanuts cartoon style: simple, bold outlines, flat colors, round heads, dot eyes, and minimal shading. The background should be minimalistic, similar to classic Peanuts comic settings. <|end|><|assistant|>"
# instruction = "<|user|><|image_1|>Generate a prompt for a drawing in the style of Charles Schulz’s Snoopy comics. The prompt should describe a simple, cartoonish scene with bold outlines and minimal shading, using lighthearted and whimsical language.<|end|><|assistant|>"
# instruction = "<|user|><|image_1|>Here are three examples of prompts that describe a Snoopy-style drawing: A happy beagle with big black ears sits on top of a red doghouse, looking at the stars. The style is simple, cartoonish, with clean black outlines and no shading. A small bird with tiny wings and a tuft of feathers on its head flutters near a dog, both smiling in a minimal, newspaper comic strip style. A relaxed dog, lying on his back with a dreamy expression, while a tiny yellow bird perches on his nose. The lines are hand-drawn, expressive, and playful. Now, generate a new prompt in the same style. <|end|><|assistant|>"
instruction = "<|user|><|image_1|>Generate a text prompt for an AI art model that produces an illustration in the style of Snoopy comics. The scene should contain a dog and a bird, use a limited color palette (black, white, and simple solid colors), and be drawn with thick, hand-drawn outlines. The mood should be lighthearted and whimsical. <|end|><|assistant|>"
images = read_images(img_dir = '/home/r12942159/NTU_AI/Hw1/content_image/')

prompts = generate_text_prompts(
    images, 
    instruction,
    75, 
    len(images), 
    phi4_processor, 
    phi4_model,
)
# save_prompts2json(prompts, 'Task2-2_prompts.json')

pipe = load_stable_diffusion(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", 
    "/home/r12942159/data_18TB",
)

stylized_resize_images = [generate_stylized_images(pipe, prompts[i], images[i], strength=0.75, guidance_scale=7.5, seed=41) for i in range(len(images))]
save_images(stylized_resize_images, './hw1_Task2-2_output/')