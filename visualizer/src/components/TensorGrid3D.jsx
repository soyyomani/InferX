import { useState, useEffect, useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Text } from "@react-three/drei";
import * as THREE from "three";

// A single cell in the tensor grid
function Cell({ position, color, opacity = 0.3, highlighted = false, label = "" }) {
  const meshRef = useRef();

  useFrame(() => {
    if (meshRef.current && highlighted) {
      meshRef.current.scale.lerp(new THREE.Vector3(1.1, 1.1, 1.1), 0.1);
    } else if (meshRef.current) {
      meshRef.current.scale.lerp(new THREE.Vector3(1, 1, 1), 0.1);
    }
  });

  return (
    <group position={position}>
      <mesh ref={meshRef}>
        <boxGeometry args={[0.85, 0.85, 0.85]} />
        <meshStandardMaterial
          color={highlighted ? "#58a6ff" : color}
          transparent
          opacity={highlighted ? 0.9 : opacity}
          wireframe={!highlighted}
        />
      </mesh>
      {highlighted && label && (
        <Text position={[0, 0.6, 0]} fontSize={0.25} color="#ffffff">
          {label}
        </Text>
      )}
    </group>
  );
}

// Arrow showing stride direction
function StrideArrow({ from, to, color = "#3fb950" }) {
  const points = useMemo(() => {
    return [new THREE.Vector3(...from), new THREE.Vector3(...to)];
  }, [from, to]);

  const geometry = useMemo(() => {
    return new THREE.BufferGeometry().setFromPoints(points);
  }, [points]);

  return (
    <group>
      <line geometry={geometry}>
        <lineBasicMaterial color={color} linewidth={3} />
      </line>
      {/* Arrow head */}
      <mesh position={to}>
        <coneGeometry args={[0.1, 0.3, 8]} />
        <meshStandardMaterial color={color} />
      </mesh>
    </group>
  );
}

// The 3D tensor grid with stride animation
function TensorScene({ shape, strides, currentIndex, animating, strideStep }) {
  const dims = shape.length;
  if (dims === 0 || dims > 3) return null;

  // Normalize to 3D (pad with 1s for lower dims)
  const d = dims === 1 ? [1, 1, shape[0]] :
            dims === 2 ? [1, shape[0], shape[1]] :
            [shape[0], shape[1], shape[2]];

  const cells = [];
  const maxCells = Math.min(d[0] * d[1] * d[2], 512); // cap for performance

  // Generate cell positions
  let count = 0;
  for (let z = 0; z < d[0] && count < maxCells; z++) {
    for (let y = 0; y < d[1] && count < maxCells; y++) {
      for (let x = 0; x < d[2] && count < maxCells; x++) {
        const isHighlighted = currentIndex &&
          (dims === 1 ? x === currentIndex[0] :
           dims === 2 ? y === currentIndex[0] && x === currentIndex[1] :
           z === currentIndex[0] && y === currentIndex[1] && x === currentIndex[2]);

        const flatIdx = dims === 1 ? x :
                        dims === 2 ? y * d[2] + x :
                        z * d[1] * d[2] + y * d[2] + x;

        cells.push({
          key: `${z}-${y}-${x}`,
          position: [x * 1.1 - (d[2] - 1) * 0.55, -(y * 1.1 - (d[1] - 1) * 0.55), -(z * 1.1)],
          highlighted: isHighlighted,
          flat: flatIdx,
          color: z === 0 ? "#1f3a5f" : z === 1 ? "#3a1f5f" : "#1f5f3a",
        });
        count++;
      }
    }
  }

  // Stride arrows from current position
  let arrows = [];
  if (currentIndex && strides && strideStep !== null) {
    const cx = dims === 1 ? currentIndex[0] : dims === 2 ? currentIndex[1] : currentIndex[2];
    const cy = dims === 1 ? 0 : dims === 2 ? currentIndex[0] : currentIndex[1];
    const cz = dims <= 2 ? 0 : currentIndex[0];

    const fromPos = [cx * 1.1 - (d[2] - 1) * 0.55, -(cy * 1.1 - (d[1] - 1) * 0.55), -(cz * 1.1)];

    // Show stride for current dimension
    if (strideStep < dims) {
      let toPos;
      const stride = strides[strideStep];
      if (strideStep === dims - 1) { // last dim: move along x
        toPos = [fromPos[0] + 1.1, fromPos[1], fromPos[2]];
      } else if (strideStep === dims - 2) { // second-to-last: move along y
        toPos = [fromPos[0], fromPos[1] - 1.1, fromPos[2]];
      } else { // first dim: move along z
        toPos = [fromPos[0], fromPos[1], fromPos[2] - 1.1];
      }

      arrows.push({
        from: fromPos,
        to: toPos,
        color: strideStep === 0 ? "#ff7b72" : strideStep === 1 ? "#3fb950" : "#58a6ff",
      });
    }
  }

  // Center offset
  const centerX = 0;
  const centerY = 0;
  const centerZ = -(d[0] - 1) * 0.55;

  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <group position={[centerX, centerY, centerZ]}>
        {cells.map(cell => (
          <Cell
            key={cell.key}
            position={cell.position}
            color={cell.color}
            highlighted={cell.highlighted}
            label={cell.highlighted ? `[${cell.flat}]` : ""}
          />
        ))}
        {arrows.map((arrow, i) => (
          <StrideArrow key={i} from={arrow.from} to={arrow.to} color={arrow.color} />
        ))}
      </group>
      <OrbitControls enableDamping dampingFactor={0.05} />
    </>
  );
}

export default function TensorGrid3D({ shape, strides }) {
  const [currentIndex, setCurrentIndex] = useState(null);
  const [strideStep, setStrideStep] = useState(null);
  const [animating, setAnimating] = useState(false);
  const [walkPath, setWalkPath] = useState([]);
  const [walkPos, setWalkPos] = useState(0);

  const dims = shape.length;

  // Generate a walk path showing how iterator moves through the tensor
  function generateWalk() {
    const path = [];
    if (dims === 0) return path;

    const maxDims = shape.map(d => Math.min(d, 8)); // cap for viz
    const indices = new Array(dims).fill(0);

    const total = maxDims.reduce((a, b) => a * b, 1);
    for (let i = 0; i < Math.min(total, 32); i++) {
      path.push([...indices]);
      // Increment like a counter (last dim fastest)
      for (let d = dims - 1; d >= 0; d--) {
        indices[d]++;
        if (indices[d] < maxDims[d]) break;
        indices[d] = 0;
      }
    }
    return path;
  }

  function startWalk() {
    const path = generateWalk();
    setWalkPath(path);
    setWalkPos(0);
    setAnimating(true);
    if (path.length > 0) setCurrentIndex(path[0]);
  }

  function stepForward() {
    if (walkPos < walkPath.length - 1) {
      const next = walkPos + 1;
      setWalkPos(next);
      setCurrentIndex(walkPath[next]);

      // Determine which dimension changed for stride arrow
      const prev = walkPath[walkPos];
      const curr = walkPath[next];
      for (let d = dims - 1; d >= 0; d--) {
        if (prev[d] !== curr[d]) {
          setStrideStep(d);
          break;
        }
      }
    }
  }

  function stepBack() {
    if (walkPos > 0) {
      const prev = walkPos - 1;
      setWalkPos(prev);
      setCurrentIndex(walkPath[prev]);
      setStrideStep(null);
    }
  }

  function reset() {
    setCurrentIndex(null);
    setStrideStep(null);
    setAnimating(false);
    setWalkPath([]);
    setWalkPos(0);
  }

  function showStrideDim(d) {
    if (!currentIndex) {
      setCurrentIndex(new Array(dims).fill(0));
    }
    setStrideStep(d);
  }

  const totalElements = shape.reduce((a, b) => a * b, 1);
  const tooLarge = totalElements > 512;

  return (
    <div className="tensor-3d">
      <div className="t3d-header">
        <h4>3D Tensor View</h4>
        <span className="t3d-shape">Shape: [{shape.join(", ")}] | Strides: [{strides.join(", ")}]</span>
      </div>

      {tooLarge && (
        <div className="t3d-warning">Showing first portion (tensor too large for full 3D render)</div>
      )}

      <div className="t3d-canvas">
        <Canvas camera={{ position: [4, 3, 5], fov: 50 }}>
          <TensorScene
            shape={shape.map(d => Math.min(d, 8))}
            strides={strides}
            currentIndex={currentIndex}
            animating={animating}
            strideStep={strideStep}
          />
        </Canvas>
      </div>

      <div className="t3d-controls">
        <div className="t3d-control-group">
          <span className="control-label">Walk through tensor:</span>
          <button onClick={startWalk}>Start Walk</button>
          <button onClick={stepBack} disabled={walkPos === 0}>← Back</button>
          <button onClick={stepForward} disabled={walkPos >= walkPath.length - 1}>Step →</button>
          <button onClick={reset}>Reset</button>
          {animating && (
            <span className="walk-info">
              Position: [{currentIndex?.join(", ")}] | Flat: {walkPos} / {walkPath.length - 1}
            </span>
          )}
        </div>

        <div className="t3d-control-group">
          <span className="control-label">Show stride direction:</span>
          {shape.map((_, d) => (
            <button key={d} onClick={() => showStrideDim(d)}
              className={strideStep === d ? "active-stride" : ""}>
              Dim {d} (stride={strides[d]})
            </button>
          ))}
        </div>
      </div>

      {strideStep !== null && (
        <div className="stride-explanation-box">
          <strong>Stride[{strideStep}] = {strides[strideStep]}</strong>: Moving one step in dimension {strideStep} jumps {strides[strideStep]} element{strides[strideStep] !== 1 ? "s" : ""} in memory.
          {strideStep === dims - 1 && " (last dim = contiguous, next element in memory)"}
          {strideStep === 0 && dims === 3 && " (first dim = jump over an entire 2D plane)"}
          {strideStep === 0 && dims === 2 && " (first dim = jump over a full row)"}
        </div>
      )}
    </div>
  );
}
