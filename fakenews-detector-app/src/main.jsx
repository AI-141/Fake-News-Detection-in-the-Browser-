import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';
import * as ort from 'onnxruntime-web';

// configure wasm settings
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = false;
ort.env.wasm.proxy = false;
ort.env.logLevel = 'verbose';
ort.env.executionProviders = ['wasm'];

// make ort available globally
window.ort = ort;

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
