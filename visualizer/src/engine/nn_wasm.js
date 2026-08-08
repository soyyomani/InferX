// Bridge to the C++ neural network WASM module.
// Falls back to JavaScript simulation when WASM is not available (dev mode).

let module = null;
let loadingPromise = null;
let useSimulation = false;

export async function initNNWasm() {
  if (module) return module;
  if (loadingPromise) return loadingPromise;

  loadingPromise = (async () => {
    try {
      if (!window.createInferXNNModule) {
        await new Promise((resolve, reject) => {
          const script = document.createElement("script");
          script.src = "/inferx_nn_wasm.js";
          script.onload = resolve;
          script.onerror = () => reject(new Error("NN WASM not found"));
          document.head.appendChild(script);
        });
      }

      let retries = 0;
      while (!window.createInferXNNModule && retries < 30) {
        await new Promise((r) => setTimeout(r, 100));
        retries++;
      }

      if (!window.createInferXNNModule) {
        throw new Error("createInferXNNModule not found");
      }

      module = await window.createInferXNNModule();
      module.initNN();
      return module;
    } catch (e) {
      console.warn("WASM NN module not available, using JS simulation:", e.message);
      useSimulation = true;
      return null;
    }
  })();

  return loadingPromise;
}

export function isNNReady() {
  return module !== null || useSimulation;
}

// --- JavaScript Simulation Layer ---
// This provides the same educational content when WASM isn't compiled.
// In production, WASM runs the actual C++ code.

function hashStr(str) {
  let h = 5381;
  for (let i = 0; i < str.length; i++) {
    h = ((h << 5) + h + str.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

function seededRNG(seed) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 16) / 65536;
  };
}

// Simple tokenizer simulation
function simTokenize(text) {
  const words = text.trim().split(/\s+/).filter(Boolean);
  const tokens = [];
  const ids = [];
  for (const w of words) {
    tokens.push(w);
    ids.push((hashStr(w.toLowerCase()) % 49900) + 100);
  }
  return { tokens, ids };
}

function simTraceTokenize(text) {
  const { tokens, ids } = simTokenize(text);
  return [
    {
      component: "Tokenizer",
      title: "Step 1: Character Split",
      detail: "Split input text into individual characters",
      internal: [
        `Input: "${text}"`,
        `Length: ${text.length} characters`,
        "Each character becomes a candidate for merging",
      ],
    },
    {
      component: "Tokenizer",
      title: "Step 2: Whitespace Pre-tokenization",
      detail: "Split text on whitespace boundaries into words",
      internal: [
        `Words found: ${tokens.length}`,
        `Result: [${tokens.map((t) => `"${t}"`).join(", ")}]`,
        "Each word is processed independently for BPE merges",
      ],
    },
    {
      component: "Tokenizer",
      title: "Step 3: BPE Merge Pairs",
      detail: "Repeatedly merge the most frequent adjacent pair of symbols",
      internal: tokens.map(
        (t, i) => `Merge: found "${t}" in vocabulary (ID ${ids[i]})`
      ),
    },
    {
      component: "Tokenizer",
      title: "Step 4: Vocabulary ID Lookup",
      detail: "Map each token string to its integer ID in the vocabulary table",
      internal: tokens.map((t, i) => `"${t}" → ID ${ids[i]}`),
    },
    {
      component: "Tokenizer",
      title: "Step 5: Final Token IDs",
      detail: "The sequence of integers that the neural network receives",
      internal: [
        `Token count: ${ids.length}`,
        `IDs: [${ids.join(", ")}]`,
        "Each ID indexes into an embedding matrix to get a dense vector",
        "Vocabulary size: ~50000 tokens",
      ],
    },
  ];
}

function simTraceEmbedding(tokenIds) {
  const embedDim = 32;
  const rng = seededRNG(42);
  const lookups = tokenIds.slice(0, 4).map((id) => {
    const vec = Array.from({ length: 4 }, () => (rng() * 2 - 1).toFixed(4));
    return `token_id=${id} → row[${id}] = [${vec.join(", ")}, ...]`;
  });

  return [
    {
      component: "Embedding",
      title: "Step 1: Embedding Lookup Table",
      detail: "Each token ID selects one row from the embedding matrix",
      internal: [
        `Embedding matrix shape: [50000 × ${embedDim}]`,
        `Each row is a learned ${embedDim}-dimensional vector`,
        "Token ID acts as row index into this matrix",
        "This is the FIRST learnable layer - vectors capture word meaning",
      ],
    },
    {
      component: "Embedding",
      title: "Step 2: Token → Vector Lookup",
      detail: "For each token ID, fetch its corresponding row from the weight matrix",
      internal: lookups,
    },
    {
      component: "Embedding",
      title: "Step 3: Add Positional Encoding",
      detail: "Add position-dependent signals so the model knows word ORDER",
      internal: [
        "Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d))",
        "Formula: PE(pos, 2i+1) = cos(pos / 10000^(2i/d))",
        "This encodes position information into the vector",
        `pos=0: [0.0000, 1.0000, 0.0000, 1.0000, ...] (after adding PE)`,
        `pos=1: [0.8415, 0.5403, 0.0001, 1.0000, ...] (after adding PE)`,
      ],
    },
    {
      component: "Embedding",
      title: "Step 4: Output Embedding Matrix",
      detail: "Final embedded representation ready for attention",
      internal: [
        `Output shape: [${tokenIds.length} × ${embedDim}]`,
        "Each row is now: token_meaning + position_info",
        `Total parameters: ${50000 * embedDim}`,
        `Memory: ${50000 * embedDim * 4} bytes (float32)`,
      ],
    },
  ];
}

function simTraceAttention(seqLen) {
  const embedDim = 32;
  const headDim = 16;
  const scale = Math.sqrt(headDim).toFixed(4);
  const show = Math.min(seqLen, 4);

  // Generate deterministic attention scores
  const rng = seededRNG(123);
  const scores = [];
  for (let i = 0; i < show; i++) {
    const row = [];
    for (let j = 0; j < show; j++) {
      row.push((rng() * 2 - 1).toFixed(3));
    }
    scores.push(row);
  }

  // Softmax the scores
  const weights = scores.map((row) => {
    const maxV = Math.max(...row.map(Number));
    const exps = row.map((v) => Math.exp(Number(v) - maxV));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => (e / sum).toFixed(3));
  });

  return [
    {
      component: "Attention",
      title: "Step 1: Self-Attention Overview",
      detail: "Let each token look at all other tokens to understand context",
      internal: [
        `Input shape: [${seqLen} × ${embedDim}]`,
        "Number of heads: 2",
        `Head dimension: ${headDim}`,
        "",
        "Intuition: 'Bank' means different things in:",
        "  'I went to the river bank' (nature)",
        "  'I went to the bank to deposit' (finance)",
        "Attention lets each word look at context to disambiguate.",
      ],
    },
    {
      component: "Attention",
      title: "Step 2: Q, K, V Linear Projections",
      detail: "Project input into Query, Key, Value spaces via learned matrices",
      internal: [
        "Q = Input × W_Q  (What am I looking for?)",
        "K = Input × W_K  (What do I contain?)",
        "V = Input × W_V  (What info do I give?)",
        "",
        `Shape: [${seqLen}×${embedDim}] × [${embedDim}×${embedDim}] = [${seqLen}×${embedDim}]`,
        `Total params: 3 × ${embedDim * embedDim} = ${3 * embedDim * embedDim}`,
      ],
    },
    {
      component: "Attention",
      title: "Step 3: Scaled Dot-Product Scores",
      detail: "Compute attention scores: how much each token attends to every other",
      internal: [
        "score[i][j] = (Q[i] · K[j]) / √d_k",
        `√d_k = √${headDim} = ${scale}`,
        "Scaling prevents dot products from growing too large",
        "",
        `Attention score matrix [${seqLen}×${seqLen}]:`,
        ...scores.map((row, i) => `  [${row.join(", ")}${seqLen > show ? ", ..." : ""}]`),
      ],
    },
    {
      component: "Attention",
      title: "Step 4: Softmax → Attention Weights",
      detail: "Convert scores to probabilities (how much to attend to each token)",
      internal: [
        "Apply softmax row-wise: each row becomes a probability distribution",
        "attn_weights[i][j] = softmax(scores[i]) → P(token i attends to j)",
        "",
        "Attention weight matrix (probabilities):",
        ...weights.map(
          (row, i) =>
            `  token ${i} attends to: [${row.join(", ")}${seqLen > show ? ", ..." : ""}]`
        ),
        "",
        "Each row sums to 1.0 (valid probability distribution)",
      ],
    },
    {
      component: "Attention",
      title: "Step 5: Weighted Value Aggregation",
      detail: "Multiply attention weights by values to get context-aware representations",
      internal: [
        "output[i] = Σ_j attn_weights[i][j] × V[j]",
        "Each token's output = weighted average of ALL value vectors",
        "",
        `Output shape: [${seqLen} × ${embedDim}]`,
        "",
        "Each token now contains information from ALL other tokens,",
        "weighted by how relevant they are (attention weights).",
      ],
    },
  ];
}

function simTraceSoftmax(logits) {
  const n = logits.length;
  const maxVal = Math.max(...logits);
  const shifted = logits.map((v) => v - maxVal);
  const exps = shifted.map((v) => Math.exp(v));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((v) => v / sumExp);

  return [
    {
      component: "Softmax",
      title: "Step 1: Input Logits (Raw Scores)",
      detail: "Neural network outputs unnormalized scores called logits",
      internal: [
        `Input logits (${n} values):`,
        `[${logits.slice(0, 8).map((v) => v.toFixed(3)).join(", ")}${n > 8 ? ", ..." : ""}]`,
        "Problem: e^(large number) → overflow!",
        "Solution: subtract max first (doesn't change ratios)",
      ],
    },
    {
      component: "Softmax",
      title: "Step 2: Find Maximum (Stability Trick)",
      detail: `max(logits) = ${maxVal.toFixed(4)}`,
      internal: [
        `Scan all ${n} values to find the maximum`,
        `max = ${maxVal.toFixed(4)}`,
        "",
        "WHY? Without this trick:",
        "  e^1000 = Infinity (overflow!)",
        "  But e^(1000-1000) = e^0 = 1 (safe!)",
        "",
        "Mathematical proof it's equivalent:",
        "  softmax(x)_i = e^(x_i) / Σe^(x_j)",
        "  = e^(x_i - max) / Σe^(x_j - max)",
      ],
    },
    {
      component: "Softmax",
      title: "Step 3: Subtract Max & Exponentiate",
      detail: "Shift all values down, then apply e^x (makes all values positive)",
      internal: [
        "For each logit: compute e^(logit - max)",
        ...logits.slice(0, 6).map(
          (v, i) =>
            `e^(${v.toFixed(3)} - ${maxVal.toFixed(3)}) = e^(${shifted[i].toFixed(3)}) = ${exps[i].toFixed(4)}`
        ),
        n > 6 ? `... (${n - 6} more)` : "",
      ].filter(Boolean),
    },
    {
      component: "Softmax",
      title: "Step 4: Sum of Exponentials",
      detail: "Sum all e^(shifted) values to get the normalizing constant",
      internal: [
        `Σ = ${exps.slice(0, 4).map((v) => v.toFixed(4)).join(" + ")}${n > 4 ? " + ..." : ""} = ${sumExp.toFixed(4)}`,
        "This sum becomes the denominator (normalizer)",
      ],
    },
    {
      component: "Softmax",
      title: "Step 5: Normalize → Probabilities",
      detail: "Divide each exp value by the sum to get valid probabilities (0-1, sum to 1)",
      internal: [
        "probability_i = e^(x_i - max) / Σe^(x_j - max)",
        "",
        ...probs.slice(0, 6).map(
          (p, i) =>
            `P[${i}] = ${exps[i].toFixed(4)} / ${sumExp.toFixed(4)} = ${p.toFixed(4)} (${(p * 100).toFixed(1)}%)`
        ),
        n > 6 ? "..." : "",
        "",
        `Verification: Σ probabilities = ${probs.reduce((a, b) => a + b, 0).toFixed(4)} ≈ 1.0 ✓`,
      ].filter(Boolean),
    },
    {
      component: "Softmax",
      title: "Step 6: Output Properties",
      detail: "Probability distribution analysis",
      internal: [
        "All values in [0, 1] ✓",
        `Sum ≈ 1.0 ✓`,
        `Argmax: index ${probs.indexOf(Math.max(...probs))} with P=${Math.max(...probs).toFixed(4)}`,
        "",
        "KEY INSIGHT: Softmax amplifies differences!",
        "  Small logit differences → large probability ratios",
      ],
    },
  ];
}

function simTraceMatMul(A, M, K, B, K2, N) {
  const C = new Array(M * N).fill(0);
  const traces = [];

  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
      if (traces.length < 4) {
        const parts = [];
        for (let k = 0; k < Math.min(K, 4); k++) {
          parts.push(`${A[i * K + k].toFixed(3)}×${B[k * N + j].toFixed(3)}`);
        }
        traces.push(
          `C[${i}][${j}] = ${parts.join(" + ")}${K > 4 ? " + ..." : ""} = ${sum.toFixed(4)}`
        );
      }
    }
  }

  return [
    {
      component: "MatMul",
      title: "Step 1: Matrix Dimensions",
      detail: "Verify shapes are compatible for multiplication",
      internal: [
        `Matrix A shape: [${M} × ${K}]`,
        `Matrix B shape: [${K} × ${N}]`,
        `Rule: A columns (${K}) must equal B rows (${K2}) ✓`,
        `Output C shape: [${M} × ${N}]`,
        `Total FLOPs: ${2 * M * N * K}`,
      ],
    },
    {
      component: "MatMul",
      title: "Step 2: The Algorithm",
      detail: "C[i][j] = Σ(k=0 to K-1) A[i][k] × B[k][j]",
      internal: [
        "For each element C[i][j] in the output:",
        "  1. Take row i from matrix A",
        "  2. Take column j from matrix B",
        "  3. Multiply corresponding elements pairwise",
        "  4. Sum all products → that's the dot product",
        "",
        "This is the CORE operation of neural networks.",
      ],
    },
    {
      component: "MatMul",
      title: "Step 3: Dot Product Computation",
      detail: "Computing each element of the output matrix",
      internal: [...traces, M * N > 4 ? `... (${M * N - 4} more)` : ""].filter(Boolean),
    },
    {
      component: "MatMul",
      title: "Step 4: Result Matrix",
      detail: `Output matrix C with shape [${M} × ${N}]`,
      internal: [
        `Output elements: ${M * N}`,
        `Memory: ${M * N * 4} bytes`,
      ],
    },
  ];
}

function simTraceReLU(input) {
  const output = input.map((v) => Math.max(0, v));
  const zeros = output.filter((v) => v === 0).length;

  return [
    {
      component: "ReLU",
      title: "ReLU: Rectified Linear Unit",
      detail: "f(x) = max(0, x) — The most popular activation function",
      internal: [
        "Formula: output[i] = max(0, input[i])",
        "If positive → keep it. If negative → set to 0.",
        "",
        "Why ReLU?",
        "  • Simple and fast (just a comparison)",
        "  • No vanishing gradient for positive values",
        "  • Introduces non-linearity",
      ],
    },
    {
      component: "ReLU",
      title: "ReLU Applied",
      detail: `${zeros} of ${input.length} values zeroed out`,
      internal: [
        ...input.slice(0, 6).map(
          (v, i) =>
            `max(0, ${v.toFixed(4)}) = ${output[i].toFixed(4)}${v < 0 ? "  ← zeroed" : ""}`
        ),
        input.length > 6 ? `... (${input.length - 6} more)` : "",
        "",
        `Sparsity: ${zeros}/${input.length} neurons dead (${Math.round((zeros / input.length) * 100)}%)`,
      ].filter(Boolean),
    },
  ];
}

function simTraceGELU(input) {
  const sqrt2pi = Math.sqrt(2 / Math.PI);
  const output = input.map((x) => {
    const inner = sqrt2pi * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1 + Math.tanh(inner));
  });

  return [
    {
      component: "GELU",
      title: "GELU: Gaussian Error Linear Unit",
      detail: "f(x) = x × Φ(x) — Used in GPT, BERT, modern transformers",
      internal: [
        "Formula: GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))",
        "",
        "Why GELU over ReLU?",
        "  • Smoother: no hard kink at 0",
        "  • Small negative values aren't completely zeroed",
        "  • Used in GPT-2, GPT-3, BERT",
      ],
    },
    {
      component: "GELU",
      title: "GELU Applied",
      detail: "Smooth non-linearity applied to all values",
      internal: input.slice(0, 4).map(
        (x, i) => `GELU(${x.toFixed(4)}) = ${output[i].toFixed(4)}`
      ),
    },
  ];
}

// --- Public API (same interface whether WASM or JS simulation) ---

export function traceTokenize(text) {
  if (module && !useSimulation) {
    const result = module.traceTokenize(text);
    return vecToArray(result);
  }
  return simTraceTokenize(text);
}

export function getTokenIds(text) {
  if (module && !useSimulation) {
    const vec = module.tokenize(text);
    const arr = [];
    for (let i = 0; i < vec.size(); i++) arr.push(vec.get(i));
    vec.delete();
    return arr;
  }
  return simTokenize(text).ids;
}

export function traceEmbedding(tokenIds) {
  if (module && !useSimulation) {
    const vec = new module.VectorInt();
    for (const id of tokenIds) vec.push_back(id);
    const result = module.traceEmbedding(vec);
    vec.delete();
    return vecToArray(result);
  }
  return simTraceEmbedding(tokenIds);
}

export function traceAttention(seqLen) {
  if (module && !useSimulation) {
    const input = new module.VectorFloat();
    const rng = seededRNG(77);
    for (let i = 0; i < seqLen * 32; i++) input.push_back(rng() * 2 - 1);
    const result = module.traceAttention(input, seqLen);
    input.delete();
    return vecToArray(result);
  }
  return simTraceAttention(seqLen);
}

export function traceMatMul(A, M, K, B, K2, N) {
  if (module && !useSimulation) {
    const va = new module.VectorFloat();
    const vb = new module.VectorFloat();
    for (const v of A) va.push_back(v);
    for (const v of B) vb.push_back(v);
    const result = module.traceMatMul(va, M, K, vb, K2, N);
    va.delete();
    vb.delete();
    return vecToArray(result);
  }
  return simTraceMatMul(A, M, K, B, K2, N);
}

export function traceSoftmax(logits) {
  if (module && !useSimulation) {
    const vec = new module.VectorFloat();
    for (const v of logits) vec.push_back(v);
    const result = module.traceSoftmax(vec);
    vec.delete();
    return vecToArray(result);
  }
  return simTraceSoftmax(logits);
}

export function traceReLU(input) {
  if (module && !useSimulation) {
    const vec = new module.VectorFloat();
    for (const v of input) vec.push_back(v);
    const result = module.traceReLU(vec);
    vec.delete();
    return vecToArray(result);
  }
  return simTraceReLU(input);
}

export function traceGELU(input) {
  if (module && !useSimulation) {
    const vec = new module.VectorFloat();
    for (const v of input) vec.push_back(v);
    const result = module.traceGELU(vec);
    vec.delete();
    return vecToArray(result);
  }
  return simTraceGELU(input);
}

export function traceFullPipeline(text) {
  if (module && !useSimulation) {
    const result = module.traceFullTextPipeline(text);
    return vecToArray(result);
  }
  // Combine all simulation steps
  const { ids } = simTokenize(text);
  return [
    ...simTraceTokenize(text),
    ...simTraceEmbedding(ids),
    ...simTraceAttention(ids.length),
    ...simTraceSoftmax(Array.from({ length: 10 }, (_, i) => (i - 5) * 0.5)),
  ];
}

// Helper to convert WASM vector to JS array
function vecToArray(vec) {
  if (!vec || !vec.size) return [];
  const arr = [];
  for (let i = 0; i < vec.size(); i++) {
    const step = vec.get(i);
    const internal = [];
    if (step.internal && step.internal.size) {
      for (let j = 0; j < step.internal.size(); j++) {
        internal.push(step.internal.get(j));
      }
    }
    arr.push({
      component: step.component,
      title: step.title,
      detail: step.detail,
      internal,
    });
  }
  if (vec.delete) vec.delete();
  return arr;
}
