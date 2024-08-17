import os
import shutil
import random
from image_processing import process_images_with_model
import string
import glob

#####################################################################################################
#### these are the parameters you need to adapt to run the code yourself ####
input_folder = "/home/student_01/Projects/MA_Cahide/Datasets/Base/20_80_10_10_min4/13_Typen"  
output_folder = "generierteBilder/"

#How many images extra should be generated per found subfolder
target_images = 50

#which models should be used

models = [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    "nota-ai/bk-sdm-small"
]

#how many images with the same random number and model should be made
batch = 1

#how strong the image should be changes (currently between 5% - 25%)
strength = round(random.uniform(0.050, 0.250), 3)

#####################################################################################################

input_folder=os.path.abspath(input_folder)

# Function to copy the directory structure without the images
def copy_directory_structure(src, dest):
    for root, dirs, files in os.walk(src):
        for dir in dirs:
            src_dir = os.path.join(root, dir)
            dest_dir = os.path.join(dest, os.path.relpath(src_dir, src))
            os.makedirs(dest_dir, exist_ok=True)

# Copy directory structure from original_train_folder to new_output_folder
copy_directory_structure(input_folder, output_folder)

# Generate new images for each class in the new output directory
for class_folder in os.listdir(output_folder):
    original_class_path = os.path.join(input_folder, class_folder)  # Original images for reference
    output_class_path = os.path.join(output_folder, class_folder)  # Path to enhance with generated images
    print(original_class_path)
    counter=0
    while counter < target_images:
        
        # Select a random model and strength
        model_name = random.choice(models)
        
        
        # Generate a single image at a time for flexibility and control
        left = target_images - counter
        print("***~~~***    ", left, "    ***~~~***")
        
        input_image = random.sample(glob.glob(original_class_path + "/*.jpg"), 1)[0]
        counter+=1
        try:
            process_images_with_model(model_name, input_image, output_class_path, strength, batch)
        except RuntimeError:
            counter-=1
            print("Error")
        
