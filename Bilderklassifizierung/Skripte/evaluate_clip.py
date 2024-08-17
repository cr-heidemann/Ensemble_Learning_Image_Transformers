import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


##################################################################
### Adjustable paramaters                                      ###


dataset_path = 'datasets/obj_det/' #cached hf dataset
model_path = 'Modelle/clip' #finetuned model







##################################################################


# Load dataset
dataset = load_dataset('imagefolder', data_dir=dataset_path, split='eval')

# Load model and processor
model = CLIPModel.from_pretrained(model_path)
processor = CLIPProcessor.from_pretrained(model_path)

# Step 2: Prepare the evaluation set
results = {
    "images": [],
    "true_labels": [],
    "predicted_labels": [],
    "similarity": []
}

# Helper function to calculate cosine similarity between two texts
def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]

# Step 3: Make predictions
for example in dataset:
    image_path = example['image']
    true_caption = example['label']  # Assuming 'label' contains the caption
    
    # Load and process the image
    image = processor(images=image_path, return_tensors="pt")['pixel_values']
    
    # Make prediction
    outputs = model.generate(image)
    predicted_caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Step 4: Calculate similarity
    similarity = calculate_similarity(true_caption, predicted_caption)
    
    # Save results
    results['images'].append(image_path)
    results['true_labels'].append(true_caption)
    results['predicted_labels'].append(predicted_caption)
    results['similarity'].append(similarity)

# Step 5: Save results to an Excel file
df = pd.DataFrame(results)
df.to_excel('evaluation_results.xlsx', index=False)

print("Evaluation complete. Results saved to evaluation_results.xlsx")
