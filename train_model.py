import torch
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import os

# config
MODEL = "distilbert-base-uncased"
DATA_FILES = {"train": "train.csv", "test": "test.csv"}
MAX_LEN = 384
BATCH_SIZE = {"train": 8, "eval": 16}
LR = 2e-5
EPOCHS = 2
OUT_DIR = "./results"
MODEL_DIR = "./best_model"

def load_data():
    # check if files exist
    for file_type, file_path in DATA_FILES.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"'{file_path}' not found. Make sure train.csv and test.csv are in the current directory.")
    
    print(f"Found data files: {', '.join(DATA_FILES.values())}")
    
    # load dataset from csv files
    dataset = load_dataset("csv", data_files=DATA_FILES)
    print(f"Loaded {len(dataset['train'])} training and {len(dataset['test'])} test examples")
    
    # rename columns if needed
    features = dataset["train"].features
    if 'text_input' in features:
        dataset = dataset.rename_column("text_input", "text")
    if 'class' in features:
        dataset = dataset.rename_column("class", "label")
    
    # split train into train/validation
    split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    
    return DatasetDict({
        'train': split['train'],
        'validation': split['test'],
        'test': dataset['test']
    })

def tokenize_texts(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

def compute_metrics(eval_pred):
    # calculate accuracy and f1 score
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    metrics = evaluate.combine(["accuracy", "f1"])
    results = metrics.compute(predictions=predictions, references=labels, average="binary")
    
    return {"accuracy": results["accuracy"], "f1": results["f1"]}

def main():
    # load and prepare data
    datasets = load_data()
    print(f"Dataset loaded with {len(datasets['train'])} training examples")
    
    # initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    
    # tokenize datasets
    tokenized = datasets.map(
        lambda x: tokenize_texts(x, tokenizer), 
        batched=True
    )
    
    # format for pytorch and rename label column
    tokenized = tokenized.remove_columns(["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    
    # set up training arguments
    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE["train"],
        per_device_eval_batch_size=BATCH_SIZE["eval"],
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        report_to="tensorboard",
        fp16=torch.cuda.is_available(),
    )
    
    # initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # train and evaluate
    print("Starting training...")
    trainer.train()
    
    print("Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=tokenized["test"])
    print(f"Test results: {test_results}")
    
    # save best model
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    main() 