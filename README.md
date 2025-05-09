# Fake News Detector

Browser-based fake news detector that classifies news articles as real or fake.

## overview
- runs 100% client-side with no server calls
- returns probability score (0-1) and verdict badge
- uses quantized ONNX model for fast browser inference

## project structure
```
test1/
├── fakenews-detector-app/    # React application for the fake news detector
│   ├── public/
│   │   ├── bert-uncased/         # Model files
│   │   │   ├── config.json       # Model configuration
│   │   │   ├── onnx/             # ONNX model directory
│   │   │   │   └── model.onnx    # ONNX model file
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer_config.json
│   │   │   ├── tokenizer.json    
│   │   │   └── vocab.txt         
│   │   ├── onnx-assets/          # WASM files for ONNX Runtime
│   │   │   ├── ort-wasm-simd-threaded.wasm
│   │   │   ├── ort-wasm-simd.wasm
│   │   │   ├── ort-wasm-threaded.wasm
│   │   │   └── ort-wasm.wasm
│   │   └── vite.svg            
│   ├── src/
│   │   ├── App.jsx              # Main app component
│   │   ├── App.css              
│   │   ├── assets/              
│   │   │   └── react.svg        
│   │   ├── index.css            
│   │   ├── main.jsx             
│   │   └── tokenizerUtils.js    # Tokenizer utilities
│   ├── eslint.config.js       
│   ├── index.html               
│   ├── package.json             
│   └── vite.config.js         
├── best_model/               # Trained PyTorch model
├── onnx_fp32/                # FP32 ONNX model
├── quantized_model/          # Quantized INT8 ONNX model
├── dataset_prep.ipynb        # Data preparation notebook
├── train_model.py            # Model training script
├── export_and_quantize.py    # Model conversion & quantization script
├── compare_fp32_vs_int8_onnx_models.py # Script to compare FP32 and INT8 model performance
├── server.js                 # Simple server for hosting the app
├── Screen Recording.mov      # Demo video
```

## Implementation Details

### Model

- Base model: `distilbert-base-uncased` (66M parameters, 6 layers)
- Achieves ~97% of BERT's performance at 40% of the size and with 60% fewer parameters.
- Fine-tuned on the Fake and Real News dataset from Kaggle:
    - **Size**: 23k fake + 21k real articles
    - **License**: CC0-like (Public Domain)
    - **Dataset Link**: [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Converted to ONNX format for browser inference
- Quantized from 250MB FP32 to ~67MB INT8 (dynamic quantization) using ONNX Runtime.

### Technologies Used

- ONNX Runtime Web for model execution in the browser
- @xenova/transformers for tokenization
- React + Vite for the frontend application


## Quick Setup

Follow these steps to set up the environment and run the project:

1.  **Setup Backend (Python & Model):**
    Open your terminal and run:
    ```bash
    # Create and activate Conda environment
    conda create -n fakenews python=3.10
    conda activate fakenews

    # Install Python dependencies
    pip install "transformers>=4.40" datasets evaluate accelerate "optimum[onnxruntime,gpu]" onnxruntime-web==1.17.0

    # Convert and quantize the model (ensure your 'best_model' is ready)
    python convert_to_onnx.py 
    # (This script is an example; adapt if your conversion/quantization script is named differently or requires other steps)
    ```

3. **run frontend:**
```bash
cd fakenews-detector-app
npm install
npm run dev
```

## Implementation

The application provides:
- A verdict badge (REAL NEWS or FAKE NEWS)
- A probability bar showing confidence level
- Inference time measurement for performance monitoring


## Technical Implementation

### Model Conversion Process

The model conversion process follows these steps:

1. Export the fine-tuned PyTorch model to ONNX format
2. Apply dynamic INT8 quantization to reduce the model size
3. Save the model and tokenizer files to the appropriate directories

The conversion process is handled by the `export_and_quantize.py` scripts.

### Browser Integration

The web application uses ONNX Runtime Web to load the model and perform inference directly in the browser:

```javascript
// Initialize ONNX Runtime
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

// Load the model
const session = await ort.InferenceSession.create('/bert-uncased/onnx/model.onnx');

// Create tensor inputs
const feeds = {
    'input_ids': new ort.Tensor('int64', tokens.inputIds, [1, tokens.inputIds.length]),
    'attention_mask': new ort.Tensor('int64', tokens.attentionMask, [1, tokens.attentionMask.length])
};

// Run inference
const output = await session.run(feeds);
```

## Model Training and Evaluation

### Training Hyperparameters

- **Task**: Sequence classification (binary: fake/real)
- **Epochs**: 2
- **Learning Rate**: 2e-5
- **Max Sequence Length**: 384 tokens
- **Batch Size**: 8
- **Optimizer**: AdamW with weight decay 0.01
- **Warm-up Steps**: 500

### Evaluation Metrics

#### Original Model Metrics (during training on validation set)
- **F1 Score**: 0.9997 (99.97%)
- **Accuracy**: 0.9997 (99.97%)
- **Loss**: 0.0005

#### ONNX Model Evaluation Results (on test set)
- **FP32 Model Accuracy**: 1.0000 (100.00%)
- **FP32 Model F1 Score**: 1.0000 (100.00%)
- **INT8 Quantized Model Accuracy**: 1.0000 (100.00%)
- **INT8 Quantized Model F1 Score**: 1.0000 (100.00%)
- **Model Agreement**: 100.00% (FP32 and INT8 models make identical predictions)

## Performance

The model provides:
- Reduced model size through quantization. The INT8 quantization achieved a 74.85% size reduction (from 255.54 MB to 64.26 MB) while fully preserving the model's perfect accuracy on the test set.
- Inference time: typically <200ms on desktop devices
- Comparable accuracy to the full-precision model (in this case, identical performance was maintained).

## Implementation Notes

- Uses local model loading with @xenova/transformers
- Model expects tokenized text with input_ids and attention_mask
- Class 1 represents "real" news, Class 0 represents "fake" news
- Inference typically takes ~10,000ms depending on text length (using wasm)



