import time
from diffusers import AutoPipelineForImage2Image
from PIL import Image
import os
import glob

#####################################################################################################
#### these are the parameters you need to adapt to run the code yourself ####
input_folder = "/home/student_01/Projects/MA_Cahide/Datasets/Base/20_80_10_10_min4/13_Typen"  
output_folder = "Bilder/SD1.5/"
#####################################################################################################

# Start the timer
start_time = time.time()

# makedirs if output doesn't exist (yet)
os.makedirs(output_folder, exist_ok=True)
input_folder = os.path.abspath(input_folder)

# Initialize the pipeline
pipeline = AutoPipelineForImage2Image.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)

# Strength values to iterate over
strength_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# Iterate over all images in the input folder and subfolders
for input_path in glob.glob(os.path.join(input_folder, '**', '*.jpg'), recursive=True):
    print(f"Processing {input_path}")
    init_image = Image.open(input_path).convert("RGB")
    base_name = os.path.splitext(os.path.basename(input_path))[0]  # Extract base name without extension

    # Iterate over the specified strength values
    for strength in strength_values:
        try:
            # Generate the image with the current strength value and no prompt
            generated_image = pipeline("", image=init_image, strength=strength).images[0]

            # Construct the output filename
            output_filename = f"{base_name}_strength_{strength}.jpg"
            output_path = os.path.join(output_folder, output_filename)

            # Save the generated image
            generated_image.save(output_path, quality=50)
            print(f"Saved {output_path}")
        except Exception as e:
            print(f"Error generating image for {input_path} with strength {strength}: {e}")

# End the timer
end_time = time.time()

# Calculate total time elapsed
total_time = end_time - start_time
print(f"Total time elapsed: {total_time:.2f} seconds.")
