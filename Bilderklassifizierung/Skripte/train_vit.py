#####################################################################################################
#### these parameters need to be adjusted to run the script on your own ####

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_wandb = False  # Set to True to enable wandb logging
project_name = "Test"  # Put your project name here otherwise it will make its own called huggingface
wandb_api_key = "your_wandb_api_key_here"  # Replace with your actual wandb API key

dataset_path = "datasets/multilabel"  # Path to the dataset
model_name = "google/vit-base-patch16-224"  # Specify which ViT model to use

num_train_epochs = 1
learning_rate = 5e-05
batch_size = 8
logging_steps = 500
eval_steps = 500
save_steps = 500
save_total_limit = 2
output_dir = "tmp/vit_models/"  # Temporary output directory for the best 2 models
final_model_save_path = "models/final/vit/"  # Final model save path

#####################################################################################################

import sys
print(sys.executable)
from datasets import load_from_disk
import torch
from PIL import Image
from transformers import ViTForImageClassification, TrainingArguments, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput
from evaluate import load
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create directories if they do not exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(final_model_save_path, exist_ok=True)
print("~~~loading dataset~~~")

# Load the dataset
dataset_dict = load_from_disk(dataset_path)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["eval"]
test_dataset = dataset_dict["test"]
print(dataset_dict)

# Load the pre-trained model
model = ViTForImageClassification.from_pretrained(model_name, num_labels=len(train_dataset.features['encoded_label'].feature))

print("~~~preprocessing data~~~")

def transform(examples):
    # Convert all images to RGB format and preprocess using our image processor
    inputs = image_processor([img.convert("RGB") for img in examples["image"]], return_tensors="pt")
    # Labels
    inputs["labels"] = examples["encoded_label"]
    return inputs

# Use the with_transform() method to apply the transform to the dataset on the fly during training
train_dataset = train_dataset.with_transform(transform)
eval_dataset = eval_dataset.with_transform(transform)
test_dataset = test_dataset.with_transform(transform)

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }

# Load the metrics from the evaluate module
accuracy_metric = load("accuracy")
precision_metric = load("precision")
recall_metric = load("recall")
f1_metric = load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    
    # Compute metrics
    accuracy_score = accuracy_metric.compute(predictions=preds, references=labels, average='samples')['accuracy']
    precision_score = precision_metric.compute(predictions=preds, references=labels, average='samples')['precision']
    recall_score = recall_metric.compute(predictions=preds, references=labels, average='samples')['recall']
    f1_score = f1_metric.compute(predictions=preds, references=labels, average='samples')['f1']

    return {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score
    }

print("~~~train model~~")
# Load the ViT model
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=len(train_dataset.features['encoded_label'].feature),
    ignore_mismatched_sizes=True,
)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,  # output directory
    per_device_train_batch_size=batch_size,  # batch size per device during training
    evaluation_strategy="steps",  # evaluation strategy to adopt during training
    num_train_epochs=num_train_epochs,  # total number of training epochs
    save_steps=save_steps,  # number of update steps before saving checkpoint
    eval_steps=eval_steps,  # number of update steps before evaluating
    logging_steps=logging_steps,  # number of update steps before logging
    save_total_limit=2,  # limit the total amount of checkpoints on disk
    remove_unused_columns=False,  # remove unused columns from the dataset
    push_to_hub=False,  # do not push the model to the hub
    load_best_model_at_end=True,  # load the best model at the end of training
    learning_rate=learning_rate,
)

# Initialize wandb if enabled
if use_wandb:
    wandb.login(key=wandb_api_key)
    wandb.init(project=project_name)

# Initialize trainer
trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    data_collator=collate_fn,  # the data collator that will be used for batching
    compute_metrics=compute_metrics,  # the metrics function that will be used for evaluation
    train_dataset=train_dataset,  # training dataset
    eval_dataset=eval_dataset,  # evaluation dataset
)

# Start training
trainer.train()

# Evaluate on test set
test_results = trainer.evaluate(test_dataset)
print(test_results)

# Save final model
trainer.save_model(final_model_save_path)
