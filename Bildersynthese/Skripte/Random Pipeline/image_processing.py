from diffusers import AutoPipelineForImage2Image
from PIL import Image
import os
import glob
import random
import string

def generate_random_string(length=5):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

def process_images_with_model(model_name, input_path, output_folder, strength, batch_size):
    model_abbreviations = {
        "runwayml/stable-diffusion-v1-5": "SD15",
        "stabilityai/stable-diffusion-2-1": "SD21",
        "stabilityai/stable-diffusion-xl-refiner-1.0": "SDXL",
        "nota-ai/bk-sdm-small": "SDDist"
    }
    model_abbr = model_abbreviations.get(model_name, "UnknownModel")
    pipeline = AutoPipelineForImage2Image.from_pretrained(model_name, use_safetensors=True)

    i=1
    while i <= batch_size:
        init_image = Image.open(input_path).convert("RGB")
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        random_str = generate_random_string()
        output_filename = f"{base_name}_{model_abbr}_{strength:.3f}_{random_str}.jpg"
        output_path = os.path.join(output_folder, output_filename)
        generated_image = pipeline("", image=init_image, strength=strength).images[0]
        generated_image.save(output_path, quality=50)  # Adjust quality as needed, bigger quality = bigger image file size
        print(f"Saved {output_path}")
        i+=1
