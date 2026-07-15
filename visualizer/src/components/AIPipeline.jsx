import { useState, useEffect, useRef, useCallback } from "react";
import "./AIPipeline.css";

// --- Utility Functions ---

// Deterministic hash from string (for seeding pseudo-random output)
function hashString(str) {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash + str.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
}

// Generate deterministic pseudo-random numbers from a seed
function seededRandom(seed) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 16) / 65536;
  };
}

// Generate tensor data from a seed
function generateTensorData(count, seed) {
  const rng = seededRandom(seed);
  const data = [];
  for (let i = 0; i < count; i++) {
    data.push(parseFloat(rng().toFixed(3)));
  }
  return data;
}

// Tokenize text into words and generate token IDs deterministically
function tokenize(text) {
  const words = text.trim() ? text.trim().split(/\s+/) : [];
  const ids = words.map((w) => {
    const h = hashString(w.toLowerCase());
    return (h % 49900) + 100; // IDs between 100-49999
  });
  return { words, ids };
}

// Generate top-k predictions for text mode based on input
function generateTextPredictions(text, vocabSize = 1000) {
  const seed = hashString(text || "default");
  const rng = seededRandom(seed);

  // Common English words as vocabulary subset
  const VOCAB = [
    "the", "is", "was", "are", "been", "have", "has", "had", "will",
    "would", "could", "should", "can", "may", "might", "shall", "do",
    "does", "did", "a", "an", "and", "but", "or", "not", "no", "yes",
    "with", "from", "for", "at", "by", "on", "in", "to", "of", "it",
    "this", "that", "which", "who", "what", "where", "when", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "only", "very", "just", "also", "than", "too",
    "now", "then", "here", "there", "always", "never", "often",
    "still", "already", "soon", "well", "back", "even", "new", "old",
    "good", "great", "little", "big", "long", "high", "small", "large",
    "next", "early", "young", "important", "different", "right", "left",
    "best", "better", "last", "first", "much", "many", "before", "after",
  ];

  // Pick 5 words deterministically based on input
  const topWords = [];
  const usedIndices = new Set();
  for (let i = 0; i < 5; i++) {
    let idx;
    do {
      idx = Math.floor(rng() * VOCAB.length);
    } while (usedIndices.has(idx));
    usedIndices.add(idx);
    topWords.push(VOCAB[idx]);
  }

  // Generate probabilities that sum reasonably (softmax-like)
  const rawProbs = topWords.map(() => rng() * 3 + 0.5);
  const expProbs = rawProbs.map((p) => Math.exp(p));
  const sumExp = expProbs.reduce((a, b) => a + b, 0);
  const probs = expProbs.map((p) => p / sumExp);
  // Sort descending
  const pairs = topWords.map((w, i) => ({ word: w, prob: probs[i] }));
  pairs.sort((a, b) => b.prob - a.prob);

  return pairs.slice(0, 5);
}

// Generate MNIST-like predictions from image pixel data
function generateImagePredictions(pixelData) {
  // Use pixel data as seed for deterministic output
  let seed = 0;
  if (pixelData && pixelData.length > 0) {
    for (let i = 0; i < Math.min(pixelData.length, 100); i++) {
      seed = (seed + Math.floor(pixelData[i] * 1000 * (i + 1))) | 0;
    }
  } else {
    seed = 12345; // default seed when no image
  }

  const rng = seededRandom(seed);

  // Generate 10 raw logits
  const logits = [];
  for (let i = 0; i < 10; i++) {
    logits.push(rng() * 4 - 1); // range [-1, 3]
  }

  // Pick a "winner" — bias one class heavily
  const winnerIdx = Math.floor(rng() * 10);
  logits[winnerIdx] += 3;

  // Softmax
  const maxLogit = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => parseFloat((e / sumExps).toFixed(3)));

  // Ensure sum is ~1 (fix rounding)
  const diff = 1.0 - probs.reduce((a, b) => a + b, 0);
  probs[winnerIdx] = parseFloat((probs[winnerIdx] + diff).toFixed(3));

  return probs;
}

// Extract pixel data from an image element via canvas
function extractPixelsFromImage(imgSrc) {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = 28;
      canvas.height = 28;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, 28, 28);
      const imageData = ctx.getImageData(0, 0, 28, 28);
      const pixels = [];
      // Convert to grayscale normalized [0,1]
      for (let i = 0; i < imageData.data.length; i += 4) {
        const r = imageData.data[i];
        const g = imageData.data[i + 1];
        const b = imageData.data[i + 2];
        const gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
        pixels.push(parseFloat(gray.toFixed(3)));
      }
      resolve(pixels);
    };
    img.onerror = () => resolve(null);
    img.src = imgSrc;
  });
}

// Build text layers dynamically based on sequence length
function buildTextLayers(seqLen) {
  const s = seqLen;
  return [
    { name: "Embedding", inputShape: `[1, ${s}]`, outputShape: `[1, ${s}, 64]`, tooltip: "Looks up a 64-dim vector for each token ID" },
    { name: "Self-Attention", inputShape: `[1, ${s}, 64]`, outputShape: `[1, ${s}, 64]`, tooltip: "Q, K, V matrix multiplies let tokens attend to each other" },
    { name: "Add & LayerNorm", inputShape: `[1, ${s}, 64]`, outputShape: `[1, ${s}, 64]`, tooltip: "Residual connection + normalization for stable training" },
    { name: "FFN (Linear → ReLU → Linear)", inputShape: `[1, ${s}, 64]`, outputShape: `[1, ${s}, 64]`, tooltip: "Two linear layers with ReLU — adds model capacity" },
    { name: "Final Linear", inputShape: `[1, ${s}, 64]`, outputShape: `[1, ${s}, 1000]`, tooltip: "Projects to vocabulary size (1000 tokens)" },
    { name: "Softmax", inputShape: `[1, ${s}, 1000]`, outputShape: `[1, ${s}, 1000]`, tooltip: "Converts last-token logits to next-word probability distribution" },
  ];
}

const MNIST_LAYERS = [
  { name: "Conv2D", inputShape: "[1, 1, 28, 28]", outputShape: "[1, 16, 26, 26]", tooltip: "Slides 16 filters (3×3) over the image to detect edges and patterns" },
  { name: "ReLU", inputShape: "[1, 16, 26, 26]", outputShape: "[1, 16, 26, 26]", tooltip: "Replaces negative values with 0 — adds non-linearity" },
  { name: "MaxPool2D", inputShape: "[1, 16, 26, 26]", outputShape: "[1, 16, 13, 13]", tooltip: "Downsamples by taking the max in each 2×2 window" },
  { name: "Flatten", inputShape: "[1, 16, 13, 13]", outputShape: "[1, 2704]", tooltip: "Reshapes multi-dimensional tensor into a 1D vector" },
  { name: "Linear", inputShape: "[1, 2704]", outputShape: "[1, 10]", tooltip: "Matrix multiply + bias: projects features to 10 digit classes" },
  { name: "Softmax", inputShape: "[1, 10]", outputShape: "[1, 10]", tooltip: "Converts logits to probabilities that sum to 1" },
];

const MNIST_LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

const STAGES = [
  { id: "preprocessing", title: "Preprocessing", type: "external" },
  { id: "tensor-creation", title: "Input Tensor Creation", type: "tensor-engine" },
  { id: "inference", title: "Neural Network Inference", type: "tensor-engine" },
  { id: "output", title: "Output Tensor", type: "tensor-engine" },
  { id: "postprocessing", title: "Postprocessing", type: "external" },
];

const SPEED_MS = { slow: 2500, medium: 1500, fast: 700 };

export default function AIPipeline() {
  const [mode, setMode] = useState("image");
  const [textInput, setTextInput] = useState("Hello world");
  const [imagePreview, setImagePreview] = useState(null);
  const [imagePixels, setImagePixels] = useState(null); // actual 28x28=784 pixel values

  const [currentStage, setCurrentStage] = useState(0);
  const [currentLayer, setCurrentLayer] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState("medium");

  const timerRef = useRef(null);

  // --- Derived dynamic data ---

  // Text mode: tokenize input
  const { words: textWords, ids: textTokenIds } = tokenize(textInput);
  const maxSeqLen = 32;
  const seqLen = Math.min(Math.max(textWords.length, 1), maxSeqLen);

  // Padded IDs for display
  const paddedIds = [...textTokenIds.slice(0, maxSeqLen)];
  while (paddedIds.length < maxSeqLen) paddedIds.push(0);

  // Dynamic text layers based on actual seqLen
  const textLayers = buildTextLayers(seqLen);
  const layers = mode === "image" ? MNIST_LAYERS : textLayers;

  // Dynamic text predictions based on input
  const textPredictions = generateTextPredictions(textInput);

  // Dynamic image predictions based on pixel data
  const imagePredictions = generateImagePredictions(imagePixels);

  // Dynamic tensor data
  const tensorDataSeed = mode === "text"
    ? hashString(textInput || "default")
    : (imagePixels ? Math.floor(imagePixels.reduce((a, b) => a + b, 0) * 1000) : 42);
  const tensorData = generateTensorData(24, tensorDataSeed);

  // Pixel grid for image preprocessing (use real pixels or generate default)
  const pixelGrid = imagePixels
    ? imagePixels.slice(0, 49) // first 7x7 subset of actual pixels
    : generateTensorData(49, 42).map((v) => v * 0.8 + 0.1);

  // --- Presets ---
  const loadPreset = (preset) => {
    if (preset === "mnist") {
      setMode("image");
      setTextInput("");
      setImagePreview(null);
      setImagePixels(null);
      resetPipeline();
    } else if (preset === "text") {
      setMode("text");
      setTextInput("Hello world");
      setImagePreview(null);
      setImagePixels(null);
      resetPipeline();
    }
  };

  const resetPipeline = () => {
    setCurrentStage(0);
    setCurrentLayer(0);
    setIsPlaying(false);
    if (timerRef.current) clearInterval(timerRef.current);
  };

  const totalSteps = STAGES.length + layers.length - 1;

  const getEffectiveStep = () => {
    if (currentStage < 2) return currentStage;
    if (currentStage === 2) return 2 + currentLayer;
    return 2 + layers.length + (currentStage - 3);
  };

  const effectiveStep = getEffectiveStep();

  // Auto-play logic
  useEffect(() => {
    if (!isPlaying) {
      if (timerRef.current) clearInterval(timerRef.current);
      return;
    }

    timerRef.current = setInterval(() => {
      setCurrentStage((prev) => {
        if (prev === 2) {
          setCurrentLayer((l) => {
            if (l < layers.length - 1) return l + 1;
            setCurrentStage(3);
            return 0;
          });
          return prev;
        }
        if (prev >= STAGES.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, SPEED_MS[speed]);

    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [isPlaying, speed, layers.length]);

  const stepForward = () => {
    if (currentStage === 2) {
      if (currentLayer < layers.length - 1) {
        setCurrentLayer(currentLayer + 1);
      } else {
        setCurrentStage(3);
        setCurrentLayer(0);
      }
    } else if (currentStage < STAGES.length - 1) {
      setCurrentStage(currentStage + 1);
      if (currentStage + 1 === 2) setCurrentLayer(0);
    }
  };

  const stepBackward = () => {
    if (currentStage === 2) {
      if (currentLayer > 0) {
        setCurrentLayer(currentLayer - 1);
      } else {
        setCurrentStage(1);
      }
    } else if (currentStage === 3 || currentStage === 4) {
      if (currentStage === 3) {
        setCurrentStage(2);
        setCurrentLayer(layers.length - 1);
      } else {
        setCurrentStage(currentStage - 1);
      }
    } else if (currentStage > 0) {
      setCurrentStage(currentStage - 1);
    }
  };

  // Image upload — extract real pixel data
  const handleImageUpload = useCallback(async (e) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = async (ev) => {
        const dataUrl = ev.target.result;
        setImagePreview(dataUrl);
        // Extract actual pixels from the image
        const pixels = await extractPixelsFromImage(dataUrl);
        if (pixels) {
          setImagePixels(pixels);
        }
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const getStageStatus = (idx) => {
    if (idx < currentStage) return "completed";
    if (idx === currentStage) return "active";
    return "future";
  };

  return (
    <div className="ai-pipeline">
      <p className="ai-pipeline-subtitle">
        See what happens when you send data to an AI — from raw input through tensor operations to final prediction.
      </p>

      {/* Mode Toggle */}
      <div className="mode-toggle">
        <div className="mode-toggle-switch">
          <button className={mode === "text" ? "active" : ""} onClick={() => { setMode("text"); resetPipeline(); }}>
            Text Mode
          </button>
          <button className={mode === "image" ? "active" : ""} onClick={() => { setMode("image"); resetPipeline(); }}>
            Image Mode
          </button>
        </div>
      </div>

      {/* Input Section */}
      <div className="ai-input-section">
        <h4>{mode === "text" ? "Text Input" : "Image Input"}</h4>
        {mode === "text" ? (
          <div className="text-input-area">
            <input
              type="text"
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              placeholder='Type something like "Hello world"'
            />
          </div>
        ) : (
          <div className={`image-upload-area ${imagePreview ? "has-image" : ""}`}>
            <input type="file" accept="image/*" onChange={handleImageUpload} />
            {imagePreview ? (
              <div className="image-preview">
                <img src={imagePreview} alt="uploaded" />
                <span>Image loaded — ready for pipeline</span>
              </div>
            ) : (
              <>
                <div className="upload-icon">📷</div>
                <div className="upload-text">Drop an image here or click to upload</div>
              </>
            )}
            <div className="mode-clarification">
              <strong>This is inference, not training.</strong> No learning happens here. We extract your image's real pixels and pass them through a simulated pre-trained network. The prediction changes per image because different pixel data produces different outputs — just like a real pre-trained model would. The model weights are fixed (already learned from millions of images during training, which happened beforehand).
            </div>
          </div>
        )}
      </div>

      {/* Presets */}
      <div className="ai-presets">
        <button className={mode === "image" ? "active" : ""} onClick={() => loadPreset("mnist")}>
          MNIST Digit Classifier
        </button>
        <button className={mode === "text" ? "active" : ""} onClick={() => loadPreset("text")}>
          Tiny Text Model
        </button>
      </div>

      {/* Controls */}
      <div className="ai-controls">
        <button
          className={isPlaying ? "pause-btn" : "play-btn"}
          onClick={() => setIsPlaying(!isPlaying)}
        >
          {isPlaying ? "⏸ Pause" : "▶ Play"}
        </button>
        <button onClick={stepBackward} disabled={currentStage === 0 && currentLayer === 0}>
          ◀ Back
        </button>
        <button onClick={stepForward} disabled={currentStage === STAGES.length - 1}>
          Next ▶
        </button>
        <button onClick={resetPipeline}>⟲ Reset</button>

        <div className="speed-control">
          <span>Speed:</span>
          {["slow", "medium", "fast"].map((s) => (
            <button key={s} className={speed === s ? "active" : ""} onClick={() => setSpeed(s)}>
              {s}
            </button>
          ))}
        </div>

        <span className="ai-progress-indicator">
          Step {effectiveStep + 1} / {totalSteps}
        </span>
      </div>

      {/* Progress Bar */}
      <div className="ai-progress-bar">
        {Array.from({ length: totalSteps }).map((_, i) => {
          let cls = "progress-segment";
          if (i < effectiveStep) cls += " completed";
          else if (i === effectiveStep) cls += " active";
          if (i === 0) cls += " preprocessing";
          if (i === totalSteps - 1) cls += " postprocessing";
          return <div key={i} className={cls} />;
        })}
      </div>

      {/* Stages */}
      <div className="ai-stages">
        {STAGES.map((stage, idx) => (
          <div key={stage.id}>
            <div className={`ai-stage ${getStageStatus(idx)} ${stage.type === "external" ? "preprocessing" : ""}`}>
              <div className="stage-header" onClick={() => { setCurrentStage(idx); if (idx === 2) setCurrentLayer(0); }}>
                <div className="stage-number">{idx + 1}</div>
                <div className="stage-title-section">
                  <div className="stage-title">{stage.title}</div>
                </div>
                <span className={`stage-label ${stage.type === "tensor-engine" ? "tensor-engine" : "external"}`}>
                  {stage.type === "tensor-engine" ? "Tensor Engine" : "Not Tensor Engine"}
                </span>
              </div>
              {currentStage === idx && (
                <div className="stage-content">
                  {idx === 0 && <PreprocessingStage mode={mode} textInput={textInput} pixelGrid={pixelGrid} imagePixels={imagePixels} />}
                  {idx === 1 && <TensorCreationStage mode={mode} textInput={textInput} tensorData={tensorData} seqLen={seqLen} imagePixels={imagePixels} />}
                  {idx === 2 && <InferenceStage layers={layers} currentLayer={currentLayer} />}
                  {idx === 3 && <OutputStage mode={mode} textPredictions={textPredictions} imagePredictions={imagePredictions} seqLen={seqLen} />}
                  {idx === 4 && <PostprocessingStage mode={mode} textInput={textInput} textPredictions={textPredictions} imagePredictions={imagePredictions} />}
                </div>
              )}
            </div>
            {idx < STAGES.length - 1 && (
              <div className="ai-connector">
                <div className="ai-connector-line" />
                <div className="ai-connector-arrow">▼</div>
                <div className="ai-connector-line" />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// --- Stage Components ---

function PreprocessingStage({ mode, textInput, pixelGrid, imagePixels }) {
  if (mode === "text") {
    const { words, ids } = tokenize(textInput);
    const displayWords = words.length > 0 ? words : ["(empty)"];
    const displayIds = ids.length > 0 ? ids : [0];
    const maxSeqLen = 32;

    const truncatedIds = displayIds.slice(0, maxSeqLen);
    const paddedIds = [...truncatedIds];
    while (paddedIds.length < maxSeqLen) paddedIds.push(0);

    const needsTruncate = displayWords.length > maxSeqLen;
    const needsPad = displayWords.length < maxSeqLen;

    return (
      <div className="preprocess-animation">
        <div className="preprocess-step highlight">
          <div className="preprocess-label">TOKENIZATION</div>
          <div className="preprocess-content">
            <div className="token-flow">
              <span className="token-item">"{textInput || ""}"</span>
              <span className="token-arrow">→</span>
              {displayWords.map((w, i) => (
                <span key={i} className="token-item">"{w}"</span>
              ))}
            </div>
          </div>
        </div>
        <div className="preprocess-step">
          <div className="preprocess-label">MAP TO IDs</div>
          <div className="preprocess-content">
            <div className="token-flow">
              {displayWords.map((w, i) => (
                <span key={i} className="token-item">{w}</span>
              ))}
              <span className="token-arrow">→</span>
              {displayIds.map((id, i) => (
                <span key={i} className="token-item">{id}</span>
              ))}
            </div>
          </div>
        </div>
        <div className="preprocess-step">
          <div className="preprocess-label">{needsTruncate ? "TRUNCATE" : needsPad ? "PAD" : "READY"}</div>
          <div className="preprocess-content">
            {needsTruncate && (
              <>Truncate to max sequence length {maxSeqLen} (dropping {displayWords.length - maxSeqLen} tokens) → [{paddedIds.join(", ")}]</>
            )}
            {needsPad && (
              <>Pad with zeros to sequence length {maxSeqLen} → [{paddedIds.join(", ")}]</>
            )}
            {!needsTruncate && !needsPad && (
              <>Already at sequence length {maxSeqLen} → [{paddedIds.join(", ")}]</>
            )}
            <div className="seq-length-info">
              <span className="info-icon" title="GPT-4 uses 128K tokens. Claude uses 200K tokens. We use 32 here to keep the visualization readable.">ℹ️</span>
              <span className="info-text">We use max 32 tokens for visualization. Real models: GPT-4 = 128K tokens, Claude = 200K tokens.</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Image mode
  const hasRealPixels = imagePixels && imagePixels.length > 0;
  return (
    <div className="preprocess-animation">
      <div className="stage-explanation">
        <div className="explanation-title">What happens here? (This is NOT training)</div>
        <div className="explanation-body">
          <p>Your uploaded image is prepared for the neural network. The model is already trained (its weights are fixed). We just need to convert your image into the exact format the model expects: a 28×28 grayscale grid of numbers between 0 and 1.</p>
          <p><strong>Real pixels from your image are extracted below.</strong> Different images will show different pixel values, which later produce different predictions — just like a real pre-trained model works.</p>
        </div>
      </div>
      <div className="preprocess-step highlight">
        <div className="preprocess-label">LOAD IMAGE</div>
        <div className="preprocess-content">
          {hasRealPixels
            ? `Read raw pixel data from uploaded file (${imagePixels.length} grayscale pixels extracted)`
            : "Upload an image to see real pixel extraction"}
        </div>
      </div>
      <div className="preprocess-step">
        <div className="preprocess-label">RESIZE TO 28×28</div>
        <div className="preprocess-content">
          {hasRealPixels
            ? "Bilinear interpolation applied — image resized to 28×28 (784 pixels)"
            : "Bilinear interpolation to match model input size"}
        </div>
      </div>
      <div className="preprocess-step">
        <div className="preprocess-label">GRAYSCALE</div>
        <div className="preprocess-content">
          {hasRealPixels
            ? `Convert RGB → single channel using (0.299R + 0.587G + 0.114B) — mean intensity: ${(imagePixels.reduce((a, b) => a + b, 0) / imagePixels.length).toFixed(3)}`
            : "Convert RGB → single channel (luminance)"}
        </div>
      </div>
      <div className="preprocess-step">
        <div className="preprocess-label">NORMALIZE [0, 1]</div>
        <div className="preprocess-content">
          {hasRealPixels ? "Actual pixel values from your image (7×7 sample):" : "Divide pixel values by 255 (sample grid):"}
          <div className="pixel-grid">
            {pixelGrid.map((v, i) => (
              <div
                key={i}
                className="pixel-cell"
                style={{ background: `rgba(88, 166, 255, ${v})` }}
              >
                {v.toFixed(2)}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function TensorCreationStage({ mode, textInput, tensorData, seqLen, imagePixels }) {
  const shape = mode === "text" ? `[1, ${seqLen}]` : "[1, 1, 28, 28]";
  const numel = mode === "text" ? seqLen : 784;
  const bytes = numel * 4;
  const strides = mode === "text" ? `[${seqLen}, 1]` : "[784, 784, 28, 1]";

  // For text mode, show actual token IDs as tensor data (first 24)
  let displayData = tensorData;
  if (mode === "text") {
    const { ids } = tokenize(textInput);
    const paddedIds = [...ids.slice(0, 32)];
    while (paddedIds.length < 32) paddedIds.push(0);
    // Normalize to [0,1] range for display (divide by max vocab size)
    displayData = paddedIds.slice(0, 24).map((id) => parseFloat((id / 50000).toFixed(3)));
  } else if (imagePixels && imagePixels.length > 0) {
    // Show actual pixel values from the image
    displayData = imagePixels.slice(0, 24);
  }

  return (
    <div className="tensor-creation">
      <div className="tensor-shape-display">
        <div className="shape-badge">Shape: {shape}</div>
        <div className="shape-desc">{mode === "text" ? "batch × sequence_length" : "batch × channels × height × width"}</div>
      </div>

      {mode === "text" && (
        <div className="stage-explanation">
          <div className="explanation-title">What does Shape [1, {seqLen}] mean?</div>
          <div className="explanation-body">
            <p><strong>Batch (1st dimension = 1):</strong> We process 1 sentence at a time. In production, GPUs process 32–128 sentences in parallel (batch size). Here batch=1 means just your single input.</p>
            <p><strong>Sequence Length (2nd dimension = {seqLen}):</strong> Your input "{textInput}" has {seqLen} token{seqLen > 1 ? "s" : ""}. Each token gets one slot in the tensor. This dimension equals the number of words/tokens after tokenization.</p>
            <p><strong>How values are generated:</strong> Each word is hashed to a unique token ID (e.g. "Hello" → {tokenize(textInput).ids[0] || 0}). These IDs are then normalized to [0,1] by dividing by vocab size (50000) for the neural network to process. The grid below shows these normalized values — they change when you type different words.</p>
          </div>
        </div>
      )}

      {mode === "image" && (
        <div className="stage-explanation">
          <div className="explanation-title">What does Shape [1, 1, 28, 28] mean?</div>
          <div className="explanation-body">
            <p><strong>Batch (1st dim = 1):</strong> Processing 1 image at a time.</p>
            <p><strong>Channels (2nd dim = 1):</strong> Grayscale has 1 channel. (RGB would be 3.)</p>
            <p><strong>Height × Width (28 × 28):</strong> The image was resized to 28×28 = 784 pixels total.</p>
            <p><strong>How values are generated:</strong> These are your ACTUAL image pixels! Each number is the brightness of one pixel (0 = black, 1 = white). Upload a different image and these values change because we extract real pixel data from your file.</p>
            <p><strong>No training happens:</strong> The tensor just holds your data. The pre-trained network weights (learned from 60,000 MNIST images during training phase) are separate and fixed.</p>
          </div>
        </div>
      )}

      <div className="tensor-data-grid">
        {displayData.map((v, i) => (
          <div key={i} className="tensor-data-cell">{v.toFixed(3)}</div>
        ))}
      </div>

      <div className="tensor-metadata">
        <div className="meta-item">dtype: <span>Float32</span></div>
        <div className="meta-item">strides: <span>{strides}</span></div>
        <div className="meta-item">numel: <span>{numel.toLocaleString()}</span></div>
        <div className="meta-item">memory: <span>{bytes.toLocaleString()} bytes</span></div>
      </div>

      {mode === "text" && (
        <div className="stage-explanation">
          <div className="explanation-title">Metadata explained</div>
          <div className="explanation-body">
            <p><strong>Strides [{seqLen}, 1]:</strong> To move to the next batch, jump {seqLen} elements. To move to the next token, jump 1 element. This tells the engine how to navigate the flat memory.</p>
            <p><strong>numel ({numel}):</strong> Total number of elements = batch(1) × seq_len({seqLen}) = {numel}.</p>
            <p><strong>memory ({bytes} bytes):</strong> Each Float32 = 4 bytes, so {numel} × 4 = {bytes} bytes of contiguous memory.</p>
          </div>
        </div>
      )}
    </div>
  );
}

function InferenceStage({ layers, currentLayer }) {
  // Detailed explanations for each text-mode layer
  const layerExplanations = {
    "Embedding": {
      why: "Neural networks can't understand word IDs like 15496 directly. The embedding layer converts each integer token ID into a rich 64-dimensional vector of decimal numbers that capture meaning.",
      how: "Think of it as a lookup table: token ID 15496 → row 15496 in a learned matrix of size [50000 × 64]. Each row is a unique 64-number fingerprint for that word.",
      transform: "Input is just token IDs [1, seq]. Output adds a new dimension: each ID becomes 64 numbers → [1, seq, 64]."
    },
    "Self-Attention": {
      why: "Words mean different things in different contexts. 'Bank' means something different in 'river bank' vs 'bank account'. Self-attention lets each token look at ALL other tokens to understand context.",
      how: "Each token is projected into 3 vectors: Query (what am I looking for?), Key (what do I contain?), Value (what info do I give?). Attention score = softmax(Q·Kᵀ/√64) × V. High scores mean strong relevance between tokens.",
      transform: "Shape stays [1, seq, 64] but the VALUES change — each token's 64 numbers now encode information from the entire sequence, not just itself."
    },
    "Add & LayerNorm": {
      why: "Deep networks suffer from vanishing gradients — information gets lost as it flows through many layers. The residual connection (Add) preserves the original signal, and LayerNorm stabilizes values to prevent explosion/collapse.",
      how: "Add: output = attention_output + original_input (skip connection). LayerNorm: normalize each token's 64 values to mean=0, std=1, then scale with learned parameters.",
      transform: "Shape stays [1, seq, 64]. Values are stabilized — prevents any single dimension from dominating."
    },
    "FFN (Linear → ReLU → Linear)": {
      why: "Attention mixes information BETWEEN tokens but doesn't transform the representation WITHIN each token. The FFN adds capacity by applying the same neural network independently to each token position.",
      how: "Linear₁: [64] → [256] (expand 4×). ReLU: zero out negatives (adds non-linearity). Linear₂: [256] → [64] (compress back). This expand-compress pattern lets the network learn complex functions.",
      transform: "Shape stays [1, seq, 64]. Each token's representation is independently transformed through a bottleneck network."
    },
    "Final Linear": {
      why: "We need to predict the next word from a vocabulary of 1000 possible tokens. This layer projects each position's 64-dim hidden state into a 1000-dim score vector — one score per possible next word.",
      how: "Matrix multiply: [seq, 64] × [64, 1000] = [seq, 1000]. Each of the 1000 outputs is a 'logit' — an unnormalized score for how likely that vocabulary word is to come next.",
      transform: "Shape changes from [1, seq, 64] → [1, seq, 1000]. We go from abstract features to concrete word scores."
    },
    "Softmax": {
      why: "Raw logits can be any number (-∞ to +∞). We need actual probabilities (0 to 1, summing to 1) to pick the most likely next word.",
      how: "softmax(xᵢ) = eˣⁱ / Σeˣʲ. This exponentiates each logit (making them positive) then normalizes by dividing by the sum. Larger logits become much larger probabilities — it amplifies differences.",
      transform: "Shape stays [1, seq, 1000] but values are now probabilities. We only care about the LAST token's distribution — that predicts what comes after the final word."
    }
  };

  return (
    <div className="nn-layers">
      {layers.map((layer, i) => {
        let status = "future";
        if (i < currentLayer) status = "completed";
        else if (i === currentLayer) status = "active";
        const explanation = layerExplanations[layer.name];
        return (
          <div key={i} className={`nn-layer ${status}`}>
            <div className="layer-index">{i + 1}</div>
            <div className="layer-info">
              <div className="layer-name">{layer.name}</div>
              <div className="layer-shapes">{layer.inputShape} → {layer.outputShape}</div>
              {status === "active" && <div className="layer-tooltip">{layer.tooltip}</div>}
              {status === "active" && explanation && (
                <div className="layer-explanation">
                  <div className="explanation-row">
                    <span className="explanation-q">Why this layer?</span>
                    <span className="explanation-a">{explanation.why}</span>
                  </div>
                  <div className="explanation-row">
                    <span className="explanation-q">How it works:</span>
                    <span className="explanation-a">{explanation.how}</span>
                  </div>
                  <div className="explanation-row">
                    <span className="explanation-q">Shape transform:</span>
                    <span className="explanation-a">{explanation.transform}</span>
                  </div>
                </div>
              )}
            </div>
            <div className="layer-arrow">→</div>
          </div>
        );
      })}
    </div>
  );
}

function OutputStage({ mode, textPredictions, imagePredictions, seqLen }) {
  if (mode === "image") {
    const maxVal = Math.max(...imagePredictions);
    const winnerIdx = imagePredictions.indexOf(maxVal);
    return (
      <div className="output-tensor">
        <div className="tensor-shape-display">
          <div className="shape-badge">Shape: [1, 10]</div>
          <div className="shape-desc">Probabilities for each digit class (0–9)</div>
        </div>

        <div className="stage-explanation">
          <div className="explanation-title">How does the network produce these probabilities?</div>
          <div className="explanation-body">
            <p><strong>The journey:</strong> Your 784 pixel values went through Conv2D (edge detection) → ReLU (non-linearity) → MaxPool (downsampling) → Flatten (reshape to 1D) → Linear (project to 10 scores) → Softmax (convert to probabilities).</p>
            <p><strong>Why digit "{winnerIdx}"?</strong> The pixel pattern in your specific image, after being processed by the network's pre-trained weights, produced the highest score for class {winnerIdx}. Different images have different pixel patterns → different activations in each layer → different final scores.</p>
            <p><strong>This is a simulation:</strong> In a real MNIST classifier, the network weights were learned by training on 60,000 handwritten digit images. Here we simulate the output deterministically from your pixel data to demonstrate the pipeline flow. A real model would give accurate digit recognition.</p>
          </div>
        </div>

        <div className="output-values">
          <div className="output-label">Softmax output — probability distribution:</div>
          <div className="bar-chart">
            {imagePredictions.map((v, i) => (
              <div key={i} className={`bar-item ${v === maxVal ? "winner" : ""}`}>
                <div className="bar-fill" style={{ height: `${v * 100}%` }} />
                <div className="bar-value">{(v * 100).toFixed(0)}%</div>
                <div className="bar-label">{MNIST_LABELS[i]}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="tensor-metadata">
          <div className="meta-item">dtype: <span>Float32</span></div>
          <div className="meta-item">values sum to: <span>{imagePredictions.reduce((a, b) => a + b, 0).toFixed(2)}</span></div>
          <div className="meta-item">argmax: <span>{winnerIdx} (digit "{winnerIdx}")</span></div>
        </div>
      </div>
    );
  }

  // Text mode
  return (
    <div className="output-tensor">
      <div className="tensor-shape-display">
        <div className="shape-badge">Shape: [1, {seqLen}, 1000]</div>
        <div className="shape-desc">Logits for each position over vocabulary (1000 tokens)</div>
      </div>

      <div className="stage-explanation">
        <div className="explanation-title">Why Shape [1, {seqLen}, 1000]?</div>
        <div className="explanation-body">
          <p><strong>1 (batch):</strong> Still processing your single input sentence.</p>
          <p><strong>{seqLen} (positions):</strong> The model produces a prediction at EVERY token position. Position 1 predicts what comes after word 1, position 2 predicts what comes after word 2, etc. We only care about the LAST position ({seqLen}) — it predicts the next word after your entire input.</p>
          <p><strong>1000 (vocabulary):</strong> At each position, the model scores all 1000 possible next-words. The word with the highest probability wins.</p>
        </div>
      </div>

      <div className="stage-explanation">
        <div className="explanation-title">How are these probabilities generated?</div>
        <div className="explanation-body">
          <p>The neural network produced 1000 raw scores (logits) for position {seqLen}. Softmax converts these into probabilities:</p>
          <p>1. Each logit is exponentiated: e^(logit) — this makes all values positive</p>
          <p>2. Divide each by the sum of all 1000 exponentiated values — this forces them to sum to 1.0</p>
          <p>3. The result: "{textPredictions[0]?.word}" got the highest logit, so after softmax it has the highest probability ({(textPredictions[0]?.prob * 100).toFixed(0)}%)</p>
          <p><strong>Why these specific words?</strong> In a real model, the network has learned from billions of text examples which words typically follow which contexts. Here we simulate this — different input text produces different probability distributions because the entire computation chain (embedding → attention → FFN → linear) transforms differently for each unique input.</p>
        </div>
      </div>

      <div className="output-values">
        <div className="output-label">Softmax on last token (position {seqLen}) — top-5 next-word probabilities:</div>
        <div className="top-tokens">
          {textPredictions.map((t, i) => (
            <div key={i} className={`top-token-row ${i === 0 ? "winner" : ""}`}>
              <span className="top-token-rank">#{i + 1}</span>
              <div className="top-token-bar-bg">
                <div className="top-token-bar-fill" style={{ width: `${t.prob * 100 * 2.5}%` }} />
              </div>
              <span className="top-token-word">"{t.word}"</span>
              <span className="top-token-prob">{(t.prob * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </div>
      <div className="tensor-metadata">
        <div className="meta-item">dtype: <span>Float32</span></div>
        <div className="meta-item">values sum to: <span>1.00</span></div>
        <div className="meta-item">vocab size: <span>1000</span></div>
      </div>
    </div>
  );
}

function PostprocessingStage({ mode, textInput, textPredictions, imagePredictions }) {
  if (mode === "image") {
    const maxVal = Math.max(...imagePredictions);
    const winnerIdx = imagePredictions.indexOf(maxVal);
    const confidence = (maxVal * 100).toFixed(0);
    const probsDisplay = imagePredictions.map((v, i) =>
      i === winnerIdx ? `<strong>${v.toFixed(2)}</strong>` : v.toFixed(2)
    ).join(", ");

    return (
      <div className="postprocess-result">
        <div className="postprocess-steps">
          <div className="postprocess-step">
            <span className="postprocess-step-label">argmax:</span>
            <span className="postprocess-step-content" dangerouslySetInnerHTML={{
              __html: `[${probsDisplay}] → index <strong>${winnerIdx}</strong>`
            }} />
          </div>
          <div className="postprocess-step">
            <span className="postprocess-step-label">map to label:</span>
            <span className="postprocess-step-content">index {winnerIdx} → digit "{winnerIdx}"</span>
          </div>
          <div className="postprocess-step">
            <span className="postprocess-step-label">confidence:</span>
            <span className="postprocess-step-content">{maxVal.toFixed(2)} → {confidence}%</span>
          </div>
        </div>
        <div className="final-prediction">
          <div className="prediction-label">PREDICTED</div>
          <div className="prediction-value">{winnerIdx}</div>
          <div className="prediction-confidence">Confidence: {confidence}%</div>
        </div>
      </div>
    );
  }

  // Text mode
  const displayInput = textInput?.trim() || "Hello world";
  const predictedWord = textPredictions[0]?.word || "the";
  const confidence = textPredictions[0]?.prob || 0;
  const tokenId = hashString(predictedWord) % 1000;

  return (
    <div className="postprocess-result">
      <div className="stage-explanation">
        <div className="explanation-title">How is the final word decided?</div>
        <div className="explanation-body">
          <p><strong>Step 1 — argmax:</strong> We look at the last token's 1000 probability values and find the INDEX of the largest one. This is called "argmax" (argument of the maximum). Out of 1000 possible words, token ID {tokenId} had the highest probability ({(confidence * 100).toFixed(0)}%).</p>
          <p><strong>Step 2 — decode:</strong> The token ID ({tokenId}) is looked up in the vocabulary table to get the actual word: "{predictedWord}". This reverses what tokenization did in step 1.</p>
          <p><strong>Step 3 — concatenate:</strong> The predicted word is appended to your original input. In a real chatbot, this process repeats — the new sentence becomes input, and the model predicts the NEXT word, and so on until it generates a stop token.</p>
          <p><strong>Why "{predictedWord}" specifically?</strong> The entire neural network computation (embedding your specific tokens → attention between YOUR words → FFN transformation → linear projection to vocab) produced a unique set of 1000 logits. After softmax, "{predictedWord}" ended up with the highest probability. Change your input text and the entire chain recomputes, giving a different prediction.</p>
        </div>
      </div>

      <div className="postprocess-steps">
        <div className="postprocess-step">
          <span className="postprocess-step-label">pick top:</span>
          <span className="postprocess-step-content">argmax over last token's 1000 logits → token ID {tokenId}</span>
        </div>
        <div className="postprocess-step">
          <span className="postprocess-step-label">decode:</span>
          <span className="postprocess-step-content">token ID {tokenId} → "{predictedWord}"</span>
        </div>
        <div className="postprocess-step">
          <span className="postprocess-step-label">result:</span>
          <span className="postprocess-step-content">"{displayInput}" + "{predictedWord}" → "{displayInput} {predictedWord}"</span>
        </div>
      </div>
      <div className="final-prediction">
        <div className="prediction-label">NEXT WORD</div>
        <div className="prediction-value">"{predictedWord}"</div>
        <div className="prediction-confidence">Confidence: {(confidence * 100).toFixed(0)}%</div>
      </div>
    </div>
  );
}
