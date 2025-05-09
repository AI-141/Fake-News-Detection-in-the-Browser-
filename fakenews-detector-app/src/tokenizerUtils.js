import { AutoTokenizer, env } from '@xenova/transformers';

// configure the transformers.js environment
env.allowRemoteModels = false;
env.localModelPath = '/';
env.backends.onnx.wasm.wasmPaths = '/onnx-assets/';
env.useLocalFiles = true;
env.useFSCache = false;

const MODEL_ID = "bert-uncased";

export async function loadXenovaTokenizer() {
  console.log(`Loading tokenizer for model: ${MODEL_ID}`);

  try {
    const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
      local: true,
      quantized: false
    });

    console.log("Tokenizer loaded successfully.");
    return tokenizer;
  } catch (error) {
    console.error("Error loading tokenizer:", error);
    throw error;
  }
}

export async function prepareONNXFeed(tokenizerInstance, text, maxLength = 384) {
  if (!tokenizerInstance) {
    throw new Error("Tokenizer instance is not available for preparing ONNX feed.");
  }

  if (text === null || text === undefined) {
    throw new Error("Text input cannot be null or undefined");
  }

  const textStr = String(text);
  console.log(`Preparing ONNX feed for text: "${textStr.substring(0, 50)}..."`);

  try {
    const encodedInput = await tokenizerInstance(textStr, {
      padding: 'max_length',
      truncation: true,
      max_length: maxLength,
      return_tensors: 'ort'
    });

    if (!encodedInput.input_ids || !encodedInput.attention_mask) {
      throw new Error("Tokenizer did not return expected input_ids or attention_mask tensors.");
    }

    const feed = {
      input_ids: encodedInput.input_ids,
      attention_mask: encodedInput.attention_mask
    };

    return feed;
  } catch (error) {
    console.error("Error preparing ONNX feed:", error);
    throw error;
  }
}