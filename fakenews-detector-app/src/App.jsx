import React, { useState, useEffect, useRef } from 'react';
import * as ort from 'onnxruntime-web';
import './App.css';
import { loadXenovaTokenizer, prepareONNXFeed } from './tokenizerUtils.js';

// configure onnx runtime
ort.env.wasm.wasmPaths = '/onnx-assets/';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Fake News Detector</h1>
        <ModelLoader />
      </header>
    </div>
  );
}

// modelloader component handles the complex task of loading ml models,this is separated from the main app to handle loading states independently
function ModelLoader() {
  // status tracks the current loading state (loading, error, or ready)
  const [status, setStatus] = useState('loading');
  const [message, setMessage] = useState('Initializing...');
  const [error, setError] = useState(null);

  // using refs to store model and tokenizer allows us to keep them
  // between renders without triggering re-renders when they change
  const modelRef = useRef(null);
  const tokenizerRef = useRef(null);

  // this effect runs once when component mounts and loads the ml model
  useEffect(() => {
    // flag to prevent state updates if component unmounts during loading
    let isMounted = true;

    async function initializeModel() {
      try {
        // step 1: load the tokenizer (converts text to numbers the model understands)
        setMessage('Loading tokenizer...');
        const tokenizer = await loadXenovaTokenizer();
        if (!isMounted) return;
        tokenizerRef.current = tokenizer;

        // step 2: load the onnx model (the actual ai model)
        setMessage('Loading ONNX model...');
        const modelPath = '/bert-uncased/onnx/model.onnx';
        console.log(`Loading model from: ${modelPath}`);

        // configure model execution settings
        const sessionOptions = {
          executionProviders: ['wasm'], // use webassembly for inference
          graphOptimizationLevel: 'all'  // optimize for performance
        };

        const session = await ort.InferenceSession.create(modelPath, sessionOptions);
        if (!isMounted) return;
        modelRef.current = session;

        // step 3: warm up the model with a test inference
        // this makes the first real inference faster for users
        setMessage('Warming up model...');
        const warmupText = "This is a test article about news.";

        try {
          const warmupFeed = await prepareONNXFeed(tokenizer, warmupText);
          await session.run(warmupFeed);
        } catch (warmupError) {
          console.warn("Warmup inference failed:", warmupError);
          // continue even if warmup fails - it's just for optimization
        }

        // step 4: everything is ready!
        if (isMounted) {
          setMessage('Model initialized and ready!');
          setStatus('ready');
        }
      } catch (error) {
        console.error("Initialization error:", error);
        if (isMounted) {
          setMessage(`Error: ${error.message}`);
          setError(error.stack || 'No stack trace available');
          setStatus('error');
        }
      }
    }

    // start the initialization process
    initializeModel();

    return () => {
      isMounted = false;
    };
  }, []);

  if (status === 'loading') {
    return (
      <div className="model-loader">
        <p className="loading-message">{message}</p>
        <div className="loading-spinner"></div>
      </div>
    );
  } else if (status === 'error') {
    return (
      <div className="error-container">
        <p className="error-message">{message}</p>
        {error && (
          <div className="error-details">
            <h3>Error Details:</h3>
            <pre>{error}</pre>
          </div>
        )}
        <button onClick={() => window.location.reload()}>Retry</button>
      </div>
    );
  } else {
    return <NewsDetector model={modelRef.current} tokenizer={tokenizerRef.current} />;
  }
}

// newsdetector is the main component where users interact with the model, it handles text input and displays prediction results
function NewsDetector({ model, tokenizer }) {
  // user input text
  const [text, setText] = useState('');
  // model prediction results
  const [result, setResult] = useState(null);
  // performance tracking
  const [inferenceTime, setInferenceTime] = useState(0);
  // loading state while running prediction
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);

  // update text state when user types in textarea
  const handleInputChange = (event) => {
    setText(event.target.value);
  };

  // main function that runs the fake news detection 
  const runInference = async () => {
    if (!model || !tokenizer) {
      setError('Model or tokenizer not available');
      return;
    }

    if (!text || text.trim() === '') {
      setError('Please enter some text to analyze');
      return;
    }

    setProcessing(true);
    setError(null);

    try {
      // prepare the input - convert text to tensors the model can process
      const feed = await prepareONNXFeed(tokenizer, text);

      // run the model and measure performance
      const startTime = performance.now();
      const outputMap = await model.run(feed);
      const endTime = performance.now();
      setInferenceTime((endTime - startTime).toFixed(2));

      // find the output tensor with the prediction
      let outputLogits = null;
      for (const key of Object.keys(outputMap)) {
        if (key === 'logits' || key === 'output_0' || key === '0' || key.includes('logit')) {
          outputLogits = outputMap[key];
          break;
        }
      }

      if (!outputLogits) {
        // fallback to the first output if we can't find a likely candidate
        const firstKey = Object.keys(outputMap)[0];
        outputLogits = outputMap[firstKey];
      }

      if (!outputLogits) {
        throw new Error("Could not find output tensor in model results");
      }

      // convert model output (logits) to probabilities
      let realProbability;

      if (outputLogits.dims.length < 2) {
        // single value output - interpret as real probability directly
        realProbability = 1 / (1 + Math.exp(-outputLogits.data[0]));
      } else if (outputLogits.dims[1] === 2) {

        // class 1 (index 1) should be "real" news probability
        const realLogit = outputLogits.data[1];
        // convert logit to probability using sigmoid function
        realProbability = 1 / (1 + Math.exp(-realLogit));
      } else if (outputLogits.dims[1] === 1) {
        // single logit - interpret as real probability
        realProbability = 1 / (1 + Math.exp(-outputLogits.data[0]));
      } else {
      
        const logits = Array.from(outputLogits.data); 
        const maxIndex = logits.indexOf(Math.max(...logits));
    
        realProbability = maxIndex === 1 ? 0.9 : 0.1;
      }

      // fake probability is just 1 minus real probability
      const fakeProbability = 1 - realProbability;
      // verdict based on which probability is higher
      const verdict = realProbability > 0.5 ? 'Real' : 'Fake';

      // save results to state to display in ui
      setResult({
        probability: fakeProbability.toFixed(4),
        verdict: verdict
      });
    } catch (error) {
      console.error("Inference error:", error);
      setError(`Inference error: ${error.message}`);
      setResult(null);
    } finally {
      setProcessing(false);
    }
  };

  // ui for the news detector with text input and results display
  return (
    <div className="news-detector">
      {/* text area for user to input news article */}
      <textarea
        value={text}
        onChange={handleInputChange}
        placeholder="Enter news article text here..."
        rows="10"
        cols="80"
        disabled={processing}
      />

      {/* display any errors */}
      {error && <p className="error-message">{error}</p>}

      {/* analyze button - disabled during processing or with empty input */}
      <button
        onClick={runInference}
        disabled={processing || !text}
        className="analyze-button"
      >
        {processing ? 'Analyzing...' : 'Analyze'}
      </button>

      {/* results section - only shown after successful inference */}
      {result && (
        <div className="results">
          <h2>Results:</h2>
          {/* visualization of probabilities as horizontal bars */}
          <div className="probability-bars">
            <div className="probability-bar">
              <div className="bar-label">Real:</div>
              <div className="bar-container">
                <div
                  className="bar real-bar"
                  style={{ width: `${(1 - parseFloat(result.probability)) * 100}%` }}
                ></div>
                <div className="bar-value">{((1 - parseFloat(result.probability)) * 100).toFixed(1)}%</div>
              </div>
            </div>
            <div className="probability-bar">
              <div className="bar-label">Fake:</div>
              <div className="bar-container">
                <div
                  className="bar fake-bar"
                  style={{ width: `${parseFloat(result.probability) * 100}%` }}
                ></div>
                <div className="bar-value">{(parseFloat(result.probability) * 100).toFixed(1)}%</div>
              </div>
            </div>
          </div>
          {/* final verdict with visual indicator */}
          <p className="verdict">
            <strong>Verdict:</strong>
            <span className={`badge ${result.verdict.toLowerCase()}`}>{result.verdict}</span>
          </p>
          {/* performance metrics */}
          <p className="inference-time"><strong>Inference Time:</strong> {inferenceTime} ms</p>
        </div>
      )}
    </div>
  );
}

export default App;
