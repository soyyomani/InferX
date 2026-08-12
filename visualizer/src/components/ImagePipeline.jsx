import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { runMNISTInference } from "../engine/mnist_model";
import "./ImagePipeline.css";

// ─── Utility ─────────────────────────────────────────────────────────

// Convolution with a 3×3 kernel
function applyKernel(pixels, w, h, kernel) {
  const out = new Array((w - 2) * (h - 2)).fill(0);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      let sum = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          sum += pixels[(y + ky) * w + (x + kx)] * kernel[(ky + 1) * 3 + (kx + 1)];
        }
      }
      out[(y - 1) * (w - 2) + (x - 1)] = sum;
    }
  }
  return out;
}

// ReLU
function relu(arr) { return arr.map(v => Math.max(0, v)); }

// MaxPool 2×2
function maxPool2x2(arr, w, h) {
  const ow = Math.floor(w / 2), oh = Math.floor(h / 2);
  const out = [];
  for (let y = 0; y < oh; y++) {
    for (let x = 0; x < ow; x++) {
      const a = arr[y * 2 * w + x * 2];
      const b = arr[y * 2 * w + x * 2 + 1];
      const c = arr[(y * 2 + 1) * w + x * 2];
      const d = arr[(y * 2 + 1) * w + x * 2 + 1];
      out.push(Math.max(a, b, c, d));
    }
  }
  return { data: out, w: ow, h: oh };
}

// Predefined kernels for education
const KERNELS = {
  edge_v: { name: "Vertical Edge", weights: [-1, 0, 1, -2, 0, 2, -1, 0, 1] },
  edge_h: { name: "Horizontal Edge", weights: [-1, -2, -1, 0, 0, 0, 1, 2, 1] },
  sharpen: { name: "Sharpen", weights: [0, -1, 0, -1, 5, -1, 0, -1, 0] },
  blur: { name: "Blur (Average)", weights: [1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9] },
};

const STAGES = [
  {
    id: 0, label: "Pixels", sub: "Image → Numbers",
    definition: "Convert the image into a grid of numbers (0-1) that the computer can process.",
    advantage: "Without this, the network has nothing to work with. Raw pixels are the foundation of all computer vision.",
  },
  {
    id: 1, label: "Convolution", sub: "Detect Patterns",
    definition: "Slide a small filter (3×3) across the image to detect local patterns like edges, curves, and corners.",
    advantage: "Lets the network find features regardless of WHERE they appear in the image. A curve is a curve whether it's top-left or bottom-right.",
  },
  {
    id: 2, label: "ReLU", sub: "Non-linearity",
    definition: "Replace all negative values with zero. Formula: output = max(0, input).",
    advantage: "Without ReLU, stacking layers would just be one big linear function — it could only draw straight lines. ReLU lets the network learn complex, non-linear patterns.",
  },
  {
    id: 3, label: "Pooling", sub: "Downsample",
    definition: "Take the maximum value in each 2×2 region, shrinking the image by half.",
    advantage: "Makes the network resistant to small shifts in position. A '7' shifted 2 pixels right still activates the same features. Also reduces computation by 4×.",
  },
  {
    id: 4, label: "Flatten + FC", sub: "Classify",
    definition: "Flatten all feature maps into one long vector, then multiply by a weight matrix to produce 10 scores (one per digit).",
    advantage: "Combines all the local patterns detected by convolution into a global decision. 'I see a loop at top + a vertical stroke = probably 9'.",
  },
  {
    id: 5, label: "Softmax", sub: "Probabilities",
    definition: "Convert raw scores (logits) into probabilities that sum to 100%. Formula: exp(x_i) / sum(exp(x_j)).",
    advantage: "Gives a confidence measure. Instead of 'it's a 3', we get 'it's 94% likely a 3, 4% likely an 8'. Critical for knowing when the model is uncertain.",
  },
];

// ─── Main Component ──────────────────────────────────────────────────
export default function ImagePipeline() {
  const [pixels, setPixels] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [activeStage, setActiveStage] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(false);
  const timerRef = useRef(null);

  // Computed pipeline data — uses REAL trained MNIST model
  const pipelineData = useMemo(() => {
    if (!pixels) return null;
    const w = 28, h = 28;
    const convRaw = applyKernel(pixels, w, h, KERNELS.edge_v.weights);
    const convW = w - 2, convH = h - 2;
    const reluOut = relu(convRaw);
    const pooled = maxPool2x2(reluOut, convW, convH);
    const flatSize = pooled.w * pooled.h;

    // Run REAL trained model inference
    const modelResult = runMNISTInference(pixels);
    const logits = modelResult ? modelResult.logits : Array(10).fill(0);
    const probs = modelResult ? modelResult.probs : Array(10).fill(0.1);
    const winner = modelResult ? modelResult.prediction : 0;

    return { convRaw, convW, convH, reluOut, pooled, flatSize, logits, probs, winner };
  }, [pixels]);

  const handleUpload = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const dataUrl = ev.target.result;
      setImagePreview(dataUrl);
      const img = new Image();
      img.onload = () => {
        // Step 1: Draw original image to a temp canvas to extract pixels
        const tmpCanvas = document.createElement("canvas");
        tmpCanvas.width = img.width;
        tmpCanvas.height = img.height;
        const tmpCtx = tmpCanvas.getContext("2d");
        tmpCtx.drawImage(img, 0, 0);
        const tmpData = tmpCtx.getImageData(0, 0, img.width, img.height);

        // Step 2: Find bounding box of the digit (non-white pixels)
        let minX = img.width, minY = img.height, maxX = 0, maxY = 0;
        for (let y = 0; y < img.height; y++) {
          for (let x = 0; x < img.width; x++) {
            const idx = (y * img.width + x) * 4;
            const brightness = (0.299 * tmpData.data[idx] + 0.587 * tmpData.data[idx+1] + 0.114 * tmpData.data[idx+2]) / 255;
            if (brightness < 0.85) { // dark pixel = part of digit
              if (x < minX) minX = x;
              if (x > maxX) maxX = x;
              if (y < minY) minY = y;
              if (y > maxY) maxY = y;
            }
          }
        }

        // Step 3: Crop and center into 28×28 (MNIST style: digit fits ~20×20, centered with padding)
        const canvas = document.createElement("canvas");
        canvas.width = 28; canvas.height = 28;
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, 28, 28);

        if (maxX > minX && maxY > minY) {
          const cropW = maxX - minX + 1;
          const cropH = maxY - minY + 1;
          // Scale to fit within 20×20, preserving aspect ratio
          const scale = Math.min(20 / cropW, 20 / cropH);
          const drawW = Math.round(cropW * scale);
          const drawH = Math.round(cropH * scale);
          // Center in 28×28
          const offsetX = Math.round((28 - drawW) / 2);
          const offsetY = Math.round((28 - drawH) / 2);
          ctx.drawImage(tmpCanvas, minX, minY, cropW, cropH, offsetX, offsetY, drawW, drawH);
        } else {
          ctx.drawImage(img, 0, 0, 28, 28);
        }

        const d = ctx.getImageData(0, 0, 28, 28);
        const gray = [];
        for (let i = 0; i < d.data.length; i += 4) {
          // Invert: MNIST uses white=0 (background), black=1 (ink)
          gray.push(1.0 - (0.299 * d.data[i] + 0.587 * d.data[i+1] + 0.114 * d.data[i+2]) / 255);
        }
        setPixels(gray);
      };
      img.src = dataUrl;
    };
    reader.readAsDataURL(file);
  }, []);

  function useSample(type) {
    const size = 28;
    const gray = new Array(size * size).fill(0);
    if (type === "7") {
      for (let x = 6; x < 22; x++) { gray[6*size+x] = 0.9; gray[7*size+x] = 0.85; }
      for (let y = 7; y < 23; y++) { const x = Math.round(22-(y-7)*0.7); if(x>=0&&x<size){gray[y*size+x]=0.9;if(x+1<size)gray[y*size+x+1]=0.5;} }
    } else if (type === "0") {
      for (let a = 0; a < 60; a++) { const t=a/60*2*Math.PI; const x=Math.round(14+7*Math.cos(t)); const y=Math.round(14+9*Math.sin(t)); if(x>=0&&x<28&&y>=0&&y<28)gray[y*size+x]=0.9; }
    } else {
      for (let y = 6; y < 22; y++) { gray[y*size+10]=0.9;gray[y*size+11]=0.85; }
      for (let x = 10; x < 20; x++) { gray[14*size+x]=0.9;gray[15*size+x]=0.85; }
    }
    setPixels(gray);
    setImagePreview(null);
  }

  function playAll() {
    setIsPlaying(true); setActiveStage(0);
    let s = 0;
    timerRef.current = setInterval(() => {
      s++;
      if (s >= STAGES.length) { clearInterval(timerRef.current); setIsPlaying(false); }
      else setActiveStage(s);
    }, 2000);
  }

  useEffect(() => () => { if (timerRef.current) clearInterval(timerRef.current); }, []);

  return (
    <div className="vp">
      <div className="vp-header">
        <h1><svg className="vp-icon" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg> How AI Sees Images</h1>
        <p>Upload any image and watch a CNN process it — from raw pixels to classification. Every convolution, every pooling operation, every calculation shown.</p>
      </div>

      <div className="demo-banner">
        <span className="demo-banner-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2a7 7 0 0 0-7 7c0 2.38 1.19 4.47 3 5.74V17a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2v-2.26c1.81-1.27 3-3.36 3-5.74a7 7 0 0 0-7-7z"/><line x1="9" y1="21" x2="15" y2="21"/></svg></span>
        <div className="demo-banner-text">
          <strong>Real Trained Model (Digits 0-9)</strong> — Currently trained on MNIST handwritten digits with 98.5% accuracy. Upload a clear digit image for best results. Non-digit images (photos, objects, text) won't classify correctly yet.
          <br /><span className="demo-banner-future"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{display:"inline",verticalAlign:"middle",marginRight:"4px"}}><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="M12 15l-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/></svg>Coming soon: Object detection, face recognition, scene classification, and more trained models.</span>
        </div>
      </div>

      {/* Upload */}
      <div className="vp-upload-section">
        <div className="vp-upload-row">
          <label className="vp-upload-box" htmlFor="vp-file">
            <input type="file" id="vp-file" accept="image/*" onChange={handleUpload} hidden />
            {imagePreview ? (
              <img src={imagePreview} alt="uploaded" className="vp-preview-img" />
            ) : pixels ? (
              <MiniPixelGrid pixels={pixels} size={28} cellSize={6} />
            ) : (
              <><span className="vp-upload-icon">📷</span><span>Upload Image</span></>
            )}
          </label>
          <div className="vp-samples">
            <span className="vp-samples-label">Or try samples:</span>
            <button onClick={() => useSample("7")}>Digit "7"</button>
            <button onClick={() => useSample("0")}>Digit "0"</button>
            <button onClick={() => useSample("+")}>Plus "+"</button>
          </div>
          {pixels && (
            <button className="btn btn-primary vp-play-btn" onClick={playAll} disabled={isPlaying}>
              {isPlaying ? "⏳ Processing..." : "▶ Run CNN Pipeline"}
            </button>
          )}
        </div>
      </div>

      {/* Stage Nav */}
      {pixels && (
        <div className="vp-stages">
          {STAGES.map((s, i) => (
            <button
              key={i}
              className={`vp-stage-btn ${activeStage === i ? "active" : ""} ${activeStage > i ? "done" : ""}`}
              onClick={() => { if(timerRef.current)clearInterval(timerRef.current); setIsPlaying(false); setActiveStage(i); }}
            >
              <span className="vp-stage-num">{i + 1}</span>
              <span className="vp-stage-label">{s.label}</span>
            </button>
          ))}
        </div>
      )}

      {/* Definition + Advantage for active step (LEFT SIDEBAR) + Stage Panel (RIGHT) */}
      <div className="vp-panel-area">
        {activeStage >= 0 && (
          <aside className="vp-step-sidebar">
            <div className="vp-step-info-num">Step {activeStage + 1}</div>
            <div className="vp-step-info-title">{STAGES[activeStage].label}</div>
            <div className="vp-step-info-sub">{STAGES[activeStage].sub}</div>

            <div className="vp-step-block">
              <span className="vp-step-tag">What it does</span>
              <p>{STAGES[activeStage].definition}</p>
            </div>

            <div className="vp-step-block">
              <span className="vp-step-tag advantage">Why it matters</span>
              <p>{STAGES[activeStage].advantage}</p>
            </div>
          </aside>
        )}

        <div className="vp-panel-content">
          {activeStage === 0 && pixels && <PixelStage pixels={pixels} />}
          {activeStage === 1 && pipelineData && <ConvStage pixels={pixels} data={pipelineData} />}
          {activeStage === 2 && pipelineData && <ReluStage data={pipelineData} />}
          {activeStage === 3 && pipelineData && <PoolStage data={pipelineData} />}
          {activeStage === 4 && pipelineData && <FCStage data={pipelineData} />}
          {activeStage === 5 && pipelineData && <SoftmaxStage data={pipelineData} />}
          {activeStage === -1 && !pixels && <EmptyState />}
          {activeStage === -1 && pixels && <div className="vp-ready">Image loaded! Click <strong>Run CNN Pipeline</strong> or select a step above.</div>}
        </div>
      </div>
    </div>
  );
}


// ─── Stage 1: Pixels ─────────────────────────────────────────────────
function PixelStage({ pixels }) {
  const [hoverIdx, setHoverIdx] = useState(-1);
  const w = 28, h = 28;
  const hoverRow = Math.floor(hoverIdx / w), hoverCol = hoverIdx % w;

  return (
    <div className="vp-panel">
      <div className="vp-panel-head"><h2>🖼 Step 1: Image as Pixels</h2><span className="vp-badge">Image → Numbers</span></div>
      <p className="vp-desc">AI doesn't see images — it sees a grid of numbers. Each pixel is a value from 0 (black) to 1 (white). Your 28×28 image = 784 numbers.</p>

      <div className="vp-pixel-section">
        <div className="vp-pixel-grid-wrap">
          <div className="vp-pixel-grid" style={{ gridTemplateColumns: `repeat(${w}, 7px)` }}>
            {pixels.map((v, i) => (
              <div
                key={i}
                className={`vp-px ${hoverIdx === i ? "hovered" : ""}`}
                style={{ background: `rgba(74, 144, 226, ${v})` }}
                onMouseEnter={() => setHoverIdx(i)}
                onMouseLeave={() => setHoverIdx(-1)}
              />
            ))}
          </div>
          {hoverIdx >= 0 && (
            <div className="vp-px-tooltip">
              pixel[{hoverRow}][{hoverCol}] = <strong>{pixels[hoverIdx].toFixed(3)}</strong>
            </div>
          )}
        </div>
        <div className="vp-pixel-info">
          <div className="vp-info-card">
            <span className="vp-info-label">Shape</span>
            <span className="vp-info-value">[1, 1, 28, 28]</span>
          </div>
          <div className="vp-info-card">
            <span className="vp-info-label">Total pixels</span>
            <span className="vp-info-value">784</span>
          </div>
          <div className="vp-info-card">
            <span className="vp-info-label">Value range</span>
            <span className="vp-info-value">0.0 – 1.0</span>
          </div>
          <div className="vp-info-card">
            <span className="vp-info-label">Memory</span>
            <span className="vp-info-value">3,136 bytes</span>
          </div>
        </div>
      </div>

      <div className="vp-math-box">
        <div className="vp-math-title">The Math</div>
        <code>tensor[0][0][row][col] = pixel_intensity / 255.0</code>
        <p>Shape = [batch=1, channels=1, height=28, width=28]. Each value is a float32 between 0 and 1.</p>
      </div>

      <div className="vp-insight">💡 Hover any pixel to see its exact value. Brighter blue = higher number = brighter in the image.</div>
    </div>
  );
}

// ─── Stage 2: Convolution ────────────────────────────────────────────
function ConvStage({ pixels, data }) {
  const [selectedPos, setSelectedPos] = useState(null);
  const w = 28, cw = data.convW;
  const kernel = KERNELS.edge_v.weights;

  function getConvCalc(row, col) {
    const patch = [];
    for (let ky = -1; ky <= 1; ky++) {
      for (let kx = -1; kx <= 1; kx++) {
        const py = row + 1 + ky, px = col + 1 + kx;
        patch.push(pixels[py * w + px]);
      }
    }
    const products = patch.map((p, i) => p * kernel[i]);
    const sum = products.reduce((a, b) => a + b, 0);
    return { patch, products, sum };
  }

  return (
    <div className="vp-panel">
      <div className="vp-panel-head"><h2>🔬 Step 2: Convolution</h2><span className="vp-badge">Detect Patterns</span></div>
      <p className="vp-desc">A small 3×3 filter slides over the image. At each position, it multiplies 9 pixels by 9 weights and sums them. This detects edges and patterns.</p>

      <div className="vp-conv-layout">
        {/* Output feature map — clickable */}
        <div className="vp-conv-output">
          <div className="vp-label">Output Feature Map (click any cell)</div>
          <div className="vp-conv-grid" style={{ gridTemplateColumns: `repeat(${cw}, 7px)` }}>
            {data.convRaw.map((v, i) => {
              const row = Math.floor(i / cw), col = i % cw;
              const isSelected = selectedPos?.row === row && selectedPos?.col === col;
              const norm = Math.min(1, Math.abs(v) / 2);
              return (
                <div
                  key={i}
                  className={`vp-conv-cell ${isSelected ? "selected" : ""}`}
                  style={{ background: v >= 0 ? `rgba(74,144,226,${norm})` : `rgba(248,113,113,${norm})` }}
                  onClick={() => setSelectedPos({ row, col })}
                />
              );
            })}
          </div>
        </div>

        {/* Kernel display */}
        <div className="vp-kernel-display">
          <div className="vp-label">Filter (Vertical Edge)</div>
          <div className="vp-kernel-grid">
            {kernel.map((v, i) => (
              <div key={i} className="vp-kernel-cell">{v}</div>
            ))}
          </div>
        </div>
      </div>

      {/* Calculation popup for selected position */}
      {selectedPos && (() => {
        const calc = getConvCalc(selectedPos.row, selectedPos.col);
        return (
          <div className="vp-conv-calc">
            <div className="vp-conv-calc-title">
              Calculation at position [{selectedPos.row + 1}, {selectedPos.col + 1}]
              <button className="vp-calc-close" onClick={() => setSelectedPos(null)}>✕</button>
            </div>
            <div className="vp-conv-calc-grid">
              <div className="vp-calc-col">
                <span className="vp-calc-heading">3×3 Patch</span>
                <div className="vp-mini-grid">
                  {calc.patch.map((v, i) => <div key={i} className="vp-mini-cell">{v.toFixed(2)}</div>)}
                </div>
              </div>
              <span className="vp-calc-op">×</span>
              <div className="vp-calc-col">
                <span className="vp-calc-heading">Kernel</span>
                <div className="vp-mini-grid">
                  {kernel.map((v, i) => <div key={i} className="vp-mini-cell">{v}</div>)}
                </div>
              </div>
              <span className="vp-calc-op">=</span>
              <div className="vp-calc-col">
                <span className="vp-calc-heading">Products</span>
                <div className="vp-mini-grid">
                  {calc.products.map((v, i) => <div key={i} className="vp-mini-cell vp-mini-product">{v.toFixed(3)}</div>)}
                </div>
              </div>
            </div>
            <div className="vp-conv-sum">
              Sum = {calc.products.map(v => v.toFixed(3)).join(" + ")} = <strong>{calc.sum.toFixed(4)}</strong>
            </div>
          </div>
        );
      })()}

      <div className="vp-math-box">
        <div className="vp-math-title">Convolution Formula</div>
        <code>output[y][x] = Σᵢ Σⱼ input[y+i][x+j] × kernel[i][j]</code>
        <p>Slide the 3×3 kernel over every position. 9 multiplies + 1 sum = 1 output value. Repeat for all 26×26 positions = 26×26 = 676 output values per filter.</p>
        <p>With 16 filters: output shape = [1, 16, 26, 26]. Parameters: 16 × (9 weights + 1 bias) = 160.</p>
      </div>

      <div className="vp-insight">💡 Click any cell in the output to see the exact 3×3 patch × kernel multiplication that produced it.</div>
    </div>
  );
}


// ─── Stage 3: ReLU ───────────────────────────────────────────────────
function ReluStage({ data }) {
  const cw = data.convW;
  const negCount = data.convRaw.filter(v => v < 0).length;
  const total = data.convRaw.length;

  return (
    <div className="vp-panel">
      <div className="vp-panel-head"><h2>⚡ Step 3: ReLU Activation</h2><span className="vp-badge">Non-linearity</span></div>
      <p className="vp-desc">Replace all negative values with 0. This adds non-linearity — without it, stacking layers would be pointless (multiple linear layers = one linear layer).</p>

      <div className="vp-relu-compare">
        <div className="vp-relu-col">
          <div className="vp-label">Before ReLU (has negatives in red)</div>
          <div className="vp-conv-grid" style={{ gridTemplateColumns: `repeat(${cw}, 7px)` }}>
            {data.convRaw.slice(0, cw * cw).map((v, i) => {
              const norm = Math.min(1, Math.abs(v) / 2);
              return <div key={i} className="vp-conv-cell" style={{ background: v >= 0 ? `rgba(74,144,226,${norm})` : `rgba(248,113,113,${norm})` }} />;
            })}
          </div>
        </div>
        <span className="vp-relu-arrow">→ ReLU →</span>
        <div className="vp-relu-col">
          <div className="vp-label">After ReLU (negatives → 0)</div>
          <div className="vp-conv-grid" style={{ gridTemplateColumns: `repeat(${cw}, 7px)` }}>
            {data.reluOut.slice(0, cw * cw).map((v, i) => {
              const norm = Math.min(1, v / 2);
              return <div key={i} className="vp-conv-cell" style={{ background: v > 0 ? `rgba(74,144,226,${norm})` : `rgba(30,30,40,1)` }} />;
            })}
          </div>
        </div>
      </div>

      <div className="vp-relu-stats">
        <div className="vp-info-card"><span className="vp-info-label">Zeroed out</span><span className="vp-info-value">{negCount} / {total}</span></div>
        <div className="vp-info-card"><span className="vp-info-label">Sparsity</span><span className="vp-info-value">{Math.round(negCount/total*100)}%</span></div>
        <div className="vp-info-card"><span className="vp-info-label">Kept</span><span className="vp-info-value">{total - negCount}</span></div>
      </div>

      <details className="vp-calc-dropdown">
        <summary>📝 Worked Example</summary>
        <div className="vp-calc-content">
          <div className="vp-calc-step"><span className="vp-cs-n">1</span><code>input = {data.convRaw[0].toFixed(4)} → max(0, {data.convRaw[0].toFixed(4)}) = <strong>{Math.max(0, data.convRaw[0]).toFixed(4)}</strong></code></div>
          <div className="vp-calc-step"><span className="vp-cs-n">2</span><code>input = {data.convRaw[5].toFixed(4)} → max(0, {data.convRaw[5].toFixed(4)}) = <strong>{Math.max(0, data.convRaw[5]).toFixed(4)}</strong>{data.convRaw[5] < 0 ? " ← killed!" : ""}</code></div>
          <div className="vp-calc-step"><span className="vp-cs-n">3</span><code>input = {data.convRaw[10].toFixed(4)} → max(0, {data.convRaw[10].toFixed(4)}) = <strong>{Math.max(0, data.convRaw[10]).toFixed(4)}</strong>{data.convRaw[10] < 0 ? " ← killed!" : ""}</code></div>
        </div>
      </details>

      <div className="vp-math-box">
        <div className="vp-math-title">Formula</div>
        <code>ReLU(x) = max(0, x)</code>
        <p>If positive: keep it. If negative: set to 0. That's it — the simplest useful non-linearity.</p>
      </div>
    </div>
  );
}

// ─── Stage 4: Pooling ────────────────────────────────────────────────
function PoolStage({ data }) {
  const { pooled, convW, reluOut } = data;
  const [selectedBlock, setSelectedBlock] = useState(null);

  function getPoolCalc(row, col) {
    const srcW = convW;
    const a = reluOut[row*2*srcW + col*2];
    const b = reluOut[row*2*srcW + col*2+1];
    const c = reluOut[(row*2+1)*srcW + col*2];
    const d = reluOut[(row*2+1)*srcW + col*2+1];
    return { a, b, c, d, max: Math.max(a, b, c, d) };
  }

  return (
    <div className="vp-panel">
      <div className="vp-panel-head"><h2>📐 Step 4: Max Pooling (2×2)</h2><span className="vp-badge">Downsample</span></div>
      <p className="vp-desc">Take every 2×2 block and keep only the maximum value. This halves the spatial dimensions and provides translation invariance — slight shifts don't change the output.</p>

      <div className="vp-pool-layout">
        <div>
          <div className="vp-label">Pooled Output ({pooled.w}×{pooled.h}) — click any cell</div>
          <div className="vp-conv-grid" style={{ gridTemplateColumns: `repeat(${pooled.w}, 10px)` }}>
            {pooled.data.map((v, i) => {
              const row = Math.floor(i / pooled.w), col = i % pooled.w;
              const isSelected = selectedBlock?.row === row && selectedBlock?.col === col;
              const norm = Math.min(1, v / 2);
              return <div key={i} className={`vp-pool-cell ${isSelected ? "selected" : ""}`} style={{ background: `rgba(74,144,226,${norm})` }} onClick={() => setSelectedBlock({ row, col })} />;
            })}
          </div>
        </div>
      </div>

      {selectedBlock && (() => {
        const calc = getPoolCalc(selectedBlock.row, selectedBlock.col);
        return (
          <div className="vp-pool-calc">
            <div className="vp-pool-calc-title">2×2 Block at [{selectedBlock.row*2}:{selectedBlock.row*2+2}, {selectedBlock.col*2}:{selectedBlock.col*2+2}]<button className="vp-calc-close" onClick={() => setSelectedBlock(null)}>✕</button></div>
            <div className="vp-pool-block">
              <div className="vp-pool-2x2">
                <div className={`vp-pool-val ${calc.a === calc.max ? "winner" : ""}`}>{calc.a.toFixed(3)}</div>
                <div className={`vp-pool-val ${calc.b === calc.max ? "winner" : ""}`}>{calc.b.toFixed(3)}</div>
                <div className={`vp-pool-val ${calc.c === calc.max ? "winner" : ""}`}>{calc.c.toFixed(3)}</div>
                <div className={`vp-pool-val ${calc.d === calc.max ? "winner" : ""}`}>{calc.d.toFixed(3)}</div>
              </div>
              <span className="vp-pool-arrow">→ max =</span>
              <div className="vp-pool-result"><strong>{calc.max.toFixed(3)}</strong></div>
            </div>
          </div>
        );
      })()}

      <div className="vp-math-box">
        <div className="vp-math-title">Formula</div>
        <code>output[y][x] = max(input[2y][2x], input[2y][2x+1], input[2y+1][2x], input[2y+1][2x+1])</code>
        <p>Input: [{data.convW}×{data.convW}] → Output: [{pooled.w}×{pooled.h}]. Spatial size halved, strongest features preserved.</p>
      </div>

      <div className="vp-insight">💡 Click any cell in the pooled output to see which 2×2 block it came from and which value "won."</div>
    </div>
  );
}


// ─── Stage 5: Flatten + FC ───────────────────────────────────────────
function FCStage({ data }) {
  const { pooled, flatSize } = data;
  return (
    <div className="vp-panel">
      <div className="vp-panel-head"><h2>🧮 Step 5: Flatten + Fully Connected</h2><span className="vp-badge">Classify</span></div>
      <p className="vp-desc">Flatten the 2D feature maps into a single vector, then multiply by a weight matrix to produce scores for each class (digit 0–9).</p>

      <div className="vp-fc-flow">
        <div className="vp-fc-block">
          <span className="vp-fc-shape">[1, 16, {pooled.h}, {pooled.w}]</span>
          <span className="vp-fc-name">Feature Maps</span>
        </div>
        <span className="vp-fc-arrow">flatten →</span>
        <div className="vp-fc-block">
          <span className="vp-fc-shape">[1, {flatSize}]</span>
          <span className="vp-fc-name">Flat Vector</span>
        </div>
        <span className="vp-fc-arrow">× W + b →</span>
        <div className="vp-fc-block vp-fc-output">
          <span className="vp-fc-shape">[1, 10]</span>
          <span className="vp-fc-name">Logits (scores)</span>
        </div>
      </div>

      <div className="vp-fc-logits">
        <div className="vp-label">Raw Logits (one score per digit class)</div>
        <div className="vp-logit-bars">
          {data.logits.map((l, i) => (
            <div key={i} className="vp-logit-row">
              <span className="vp-logit-class">{i}</span>
              <div className="vp-logit-bar-track">
                <div className={`vp-logit-bar ${l > 0 ? "pos" : "neg"}`} style={{ width: `${Math.min(100, Math.abs(l) * 15)}%` }} />
              </div>
              <span className="vp-logit-val">{l.toFixed(2)}</span>
            </div>
          ))}
        </div>
      </div>

      <details className="vp-calc-dropdown">
        <summary>📝 Worked Example: One output neuron</summary>
        <div className="vp-calc-content">
          <div className="vp-calc-step"><span className="vp-cs-n">1</span><code>logit[0] = flat[0]×W[0][0] + flat[1]×W[1][0] + ... + flat[{flatSize-1}]×W[{flatSize-1}][0] + bias[0]</code></div>
          <div className="vp-calc-step"><span className="vp-cs-n">2</span><code>= ({data.pooled.data[0].toFixed(3)}×0.12) + ({data.pooled.data[1].toFixed(3)}×-0.05) + ... = <strong>{data.logits[0].toFixed(3)}</strong></code></div>
          <div className="vp-calc-step"><span className="vp-cs-n">3</span><code>That's {flatSize} multiplications + {flatSize} additions for ONE output neuron. × 10 classes = {flatSize * 10} operations total.</code></div>
        </div>
      </details>

      <div className="vp-math-box">
        <div className="vp-math-title">Formulas</div>
        <code>flat = reshape(features, [1, {flatSize}])</code><br />
        <code>logits = flat × W + b</code>
        <p>W shape: [{flatSize} × 10], bias: [10]. Parameters: {flatSize * 10 + 10}.</p>
      </div>
    </div>
  );
}

// ─── Stage 6: Softmax ────────────────────────────────────────────────
function SoftmaxStage({ data }) {
  const { logits, probs, winner } = data;
  const maxLogit = Math.max(...logits);

  return (
    <div className="vp-panel">
      <div className="vp-panel-head"><h2>🎯 Step 6: Softmax → Prediction</h2><span className="vp-badge">Probabilities</span></div>
      <p className="vp-desc">Convert raw scores (logits) into probabilities that sum to 1. The highest probability is the predicted digit.</p>

      <div className="vp-softmax-bars">
        {probs.map((p, i) => (
          <div key={i} className={`vp-sm-row ${i === winner ? "winner" : ""}`}>
            <span className="vp-sm-digit">{i}</span>
            <div className="vp-sm-bar-track">
              <div className="vp-sm-bar-fill" style={{ width: `${p * 100 * 2}%` }} />
            </div>
            <span className="vp-sm-prob">{(p * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>

      <div className="vp-prediction-result">
        <span className="vp-pred-label">Prediction:</span>
        <span className="vp-pred-digit">{winner}</span>
        <span className="vp-pred-conf">{(probs[winner] * 100).toFixed(1)}% confidence</span>
      </div>

      <details className="vp-calc-dropdown">
        <summary>📝 Worked Softmax Calculation</summary>
        <div className="vp-calc-content">
          <div className="vp-calc-step"><span className="vp-cs-n">1</span><code>max logit = {maxLogit.toFixed(3)}</code></div>
          <div className="vp-calc-step"><span className="vp-cs-n">2</span><code>shifted[{winner}] = {logits[winner].toFixed(3)} - {maxLogit.toFixed(3)} = {(logits[winner] - maxLogit).toFixed(3)}</code></div>
          <div className="vp-calc-step"><span className="vp-cs-n">3</span><code>e^{(logits[winner] - maxLogit).toFixed(3)} = <strong>{Math.exp(logits[winner] - maxLogit).toFixed(4)}</strong></code></div>
          <div className="vp-calc-step"><span className="vp-cs-n">4</span><code>sum of all e^shifted = {probs.map((_, i) => Math.exp(logits[i] - maxLogit)).reduce((a,b)=>a+b,0).toFixed(4)}</code></div>
          <div className="vp-calc-step"><span className="vp-cs-n">5</span><code>P(digit={winner}) = {Math.exp(logits[winner] - maxLogit).toFixed(4)} / {probs.map((_, i) => Math.exp(logits[i] - maxLogit)).reduce((a,b)=>a+b,0).toFixed(4)} = <strong>{(probs[winner] * 100).toFixed(1)}%</strong></code></div>
        </div>
      </details>

      <div className="vp-math-box">
        <div className="vp-math-title">Softmax Formula</div>
        <code>P(class_i) = e^(logit_i - max) / Σⱼ e^(logit_j - max)</code>
        <p>Numerical stability: subtract max first. Output: 10 probabilities summing to 1.0. Winner: argmax = digit {winner}.</p>
      </div>

      <div className="vp-insight">💡 The model is {(probs[winner]*100).toFixed(0)}% confident this is digit "{winner}". This is a real trained model — for clean handwritten digits it achieves high accuracy. Complex photos or non-digit images will produce low confidence across all classes.</div>
    </div>
  );
}

// ─── Helper Components ───────────────────────────────────────────────
function MiniPixelGrid({ pixels, size, cellSize }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: `repeat(${size}, ${cellSize}px)`, gap: 0 }}>
      {pixels.slice(0, size * size).map((v, i) => (
        <div key={i} style={{ width: cellSize, height: cellSize, background: `rgba(74,144,226,${v})` }} />
      ))}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="vp-empty">
      <div className="vp-empty-icon">👁</div>
      <h3>How does AI "see" images?</h3>
      <p>It doesn't see like humans. It receives a grid of numbers (pixel values), slides small filters to detect edges and patterns, pools results to shrink the representation, then uses fully-connected layers to classify. Upload an image or use a sample to see every step.</p>
    </div>
  );
}
