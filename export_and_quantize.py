import os
import torch
import shutil
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, QuantType

# setup paths
model_path = "best_model"
onnx_dir = "onnx_fp32"
quant_dir = "quantized_model"
onnx_model = os.path.join(onnx_dir, "model.onnx")
quant_model = os.path.join(quant_dir, "model_int8.onnx")

# create output directories
os.makedirs(onnx_dir, exist_ok=True)
os.makedirs(quant_dir, exist_ok=True)

# load model and tokenizer
print("Loading model from", model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# export to onnx format
print(f"Exporting to ONNX: {onnx_model}")
input_shape = (1, 384)  # batch_size, sequence_length
dummy_input = torch.ones(input_shape, dtype=torch.int64)
attention_mask = torch.ones(input_shape, dtype=torch.int64)

torch.onnx.export(
    model,
    (dummy_input, attention_mask),
    onnx_model,
    export_params=True,
    opset_version=14,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    }
)

# quantize the model
print(f"Quantizing model to INT8: {quant_model}")
quantize_dynamic(
    onnx_model,
    quant_model,
    weight_type=QuantType.QUInt8
)

# copy tokenizer files
print("Copying tokenizer files")
# copy vocab files from tokenizer
for filename in tokenizer.vocab_files_names.values():
    src = os.path.join(model_path, filename)
    dst = os.path.join(quant_dir, os.path.basename(filename))
    if os.path.exists(src):
        shutil.copy2(src, dst)

# copy config files
for config_file in ["tokenizer_config.json", "config.json"]:
    src = os.path.join(model_path, config_file)
    dst = os.path.join(quant_dir, config_file)
    if os.path.exists(src):
        shutil.copy2(src, dst)

# report on file sizes
if os.path.exists(onnx_model) and os.path.exists(quant_model):
    fp32_size = os.path.getsize(onnx_model) / (1024 * 1024)
    int8_size = os.path.getsize(quant_model) / (1024 * 1024)
    reduction = 100 * (1 - int8_size / fp32_size) if fp32_size > 0 else 0
    
    print(f"\nModel sizes:")
    print(f"FP32: {fp32_size:.2f} MB")
    print(f"INT8: {int8_size:.2f} MB")
    print(f"Size reduction: {reduction:.1f}%") 