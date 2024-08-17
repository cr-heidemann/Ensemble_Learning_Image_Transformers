#####################################################################################################
#### diese Parameter müssen angepasst werden, um das Skript selbst auszuführen ####

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


use_wandb = False  # Set to True to enable wandb logging
project_name = "Test"  # Put your project name here otherwise it will make its own called huggingface
wandb_api_key = "your_wandb_api_key_here"  # Replace with your actual wandb API key, i deleted mine otherwise it will be logged in my project

dataset = os.path.abspath("/home/student_01/Projects/MA_Cahide/Datasets/10/10_SD")

# Specify which ViTHybrid model to use
model_name = "google/vit-hybrid-base-bit-384"

num_train_epochs = 10
learning_rate = 5e-05
batch_size = 8
logging_steps = 500
eval_steps = 500
save_steps = 500
save_total_limit = 2
output_dir = "tmp/vithybrid_models/10_SD"  # Temporary output directory for the best 2 models
final_model_save_path = "Modelle/10/ViTHybrid_10_SD"  # Final model save path

# Create directories if they do not exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(final_model_save_path, exist_ok=True)

#####################################################################################################

#import wandb



print("~~~loading dataset~~~")
from datasets import load_dataset

# load the custom dataset
ds = load_dataset("imagefolder", data_dir=dataset)
print(ds)


import requests
import torch
from PIL import Image
import transformers
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import ViTHybridModel, ViTHybridImageProcessor, ViTHybridForImageClassification, AutoImageProcessor
from transformers.modeling_outputs import SequenceClassifierOutput

# the model name
model_name = "google/vit-hybrid-base-bit-384"
# load the image processor
image_processor = ViTHybridImageProcessor.from_pretrained(model_name)
# loading the pre-trained model
model = ViTHybridForImageClassification.from_pretrained(model_name)


import urllib.parse as parse


# a function to determine whether a string is a URL or not
def is_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False
    
# a function to load an image
def load_image(image_path):
    if is_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)

def get_prediction(model, url_or_path):
  # load the image
  img = load_image(url_or_path)
  # preprocessing the image
  pixel_values = image_processor(img, return_tensors="pt")["pixel_values"].to(device)
  # perform inference
  output = model(pixel_values)
  # get the label id and return the class name
  return model.config.id2label[int(output.logits.softmax(dim=1).argmax())]

labels = ds["train"].features["label"]
#print(labels)
#print(labels.int2str(ds["train"][532]["label"]))


#print("\n")
print("~~~preprocessing data~~~")

def transform(examples):
  # convert all images to RGB format, then preprocessing it
  # using our image processor
  inputs = image_processor([img.convert("RGB") for img in examples["image"]], return_tensors="pt")
  # we also shouldn't forget about the labels
  inputs["labels"] = examples["label"]
  return inputs

# use the with_transform() method to apply the transform to the dataset on the fly during training
dataset = ds.with_transform(transform)
"""
for item in dataset["train"]:
  #print(item)  
  print(item["pixel_values"].shape)
  print(item["labels"])
  break
"""
# extract the labels for this dataset
labels = ds["train"].features["label"].names
#print(labels)


def collate_fn(batch):
  return {
      "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
      "labels": torch.tensor([x["labels"] for x in batch]),
  }

from evaluate import load
import numpy as np

# load the accuracy and f1 metrics from the evaluate module
accuracy = load("accuracy")
f1 = load("f1")

def compute_metrics(eval_pred):
  # compute the accuracy and f1 scores & return them
  accuracy_score = accuracy.compute(predictions=np.argmax(eval_pred.predictions, axis=1), references=eval_pred.label_ids)
  f1_score = f1.compute(predictions=np.argmax(eval_pred.predictions, axis=1), references=eval_pred.label_ids, average="macro")
  return {**accuracy_score, **f1_score}

print("~~~train model~~")
# load the ViTHybrid model
model = ViTHybridForImageClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes=True,
)

from transformers import TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,  # Output directory
    per_device_train_batch_size=batch_size,  # Batch size per device during training
    evaluation_strategy="steps",  # Evaluation strategy to adopt during training
    num_train_epochs=num_train_epochs,  # Total number of training epochs
    save_steps=save_steps,  # Number of update steps before saving checkpoint
    eval_steps=eval_steps,  # Number of update steps before evaluating
    logging_steps=logging_steps,  # Number of update steps before logging
    save_total_limit=save_total_limit,  # Limit the total amount of checkpoints on disk
    remove_unused_columns=False,  # Remove unused columns from the dataset
    push_to_hub=False,  # Do not push the model to the hub
    report_to="wandb" if use_wandb else None,  # Report metrics to wandb if enabled
    learning_rate=learning_rate,
    load_best_model_at_end=True,  # Load the best model at the end of training
)

from transformers import Trainer

# Initialize the Trainer
trainer = Trainer(
    model=model,  # The instantiated 🤗 Transformers model to be trained
    args=training_args,  # Training arguments, defined above
    data_collator=collate_fn,  # The data collator that will be used for batching
    compute_metrics=compute_metrics,  # The metrics function that will be used for evaluation
    train_dataset=dataset["train"],  # Training dataset
    eval_dataset=dataset["validation"],  # Evaluation dataset
    tokenizer=image_processor,  # The processor that will be used for preprocessing the images
)

# Start training
trainer.train()
# Evaluate the model
trainer.evaluate(dataset["test"])
# Save the final model
trainer.save_model(final_model_save_path)
