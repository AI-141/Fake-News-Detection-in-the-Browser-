import os
import numpy as np
import torch
import onnxruntime as ort
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate

# model paths
fp32_path = "onnx_fp32/model.onnx"
int8_path = "quantized_model/model_int8.onnx"

# check model files exist
for path in [fp32_path, int8_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

# load models and tokenizer
print("Loading models and tokenizer...")
fp32_session = ort.InferenceSession(fp32_path)
int8_session = ort.InferenceSession(int8_path)
tokenizer = AutoTokenizer.from_pretrained("best_model")

# load and prepare dataset
print("Loading test data...")
test_data = load_dataset("csv", data_files={"test": "test.csv"})["test"]
test_data = test_data.rename_column("text_input", "text")
test_data = test_data.rename_column("class", "label")

# take a sample for quicker evaluation
sample_size = min(1000, len(test_data))
eval_data = test_data.select(range(sample_size))

# tokenize data
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=384)

tokenized_data = eval_data.map(tokenize, batched=True)

# evaluation
print("Running evaluation...")
fp32_preds = []
int8_preds = []
batch_size = 16

# process batches
for i in range(0, len(tokenized_data), batch_size):
    batch = tokenized_data[i:i+batch_size]
    
    # prepare inputs
    inputs = {
        "input_ids": np.array(batch["input_ids"], dtype=np.int64),
        "attention_mask": np.array(batch["attention_mask"], dtype=np.int64),
    }
    
    # run models
    fp32_logits = fp32_session.run(None, inputs)[0]
    int8_logits = int8_session.run(None, inputs)[0]
    
    # get predictions
    fp32_preds.extend(np.argmax(fp32_logits, axis=1).tolist())
    int8_preds.extend(np.argmax(int8_logits, axis=1).tolist())

# calculate metrics
true_labels = tokenized_data["label"]
metrics = evaluate.combine(["accuracy", "f1"])

fp32_results = metrics.compute(predictions=fp32_preds, references=true_labels, average="binary")
int8_results = metrics.compute(predictions=int8_preds, references=true_labels, average="binary")

# print results
print("\n--- Evaluation Results ---")
print(f"FP32 Model: Accuracy={fp32_results['accuracy']:.4f}, F1={fp32_results['f1']:.4f}")
print(f"INT8 Model: Accuracy={int8_results['accuracy']:.4f}, F1={int8_results['f1']:.4f}")

