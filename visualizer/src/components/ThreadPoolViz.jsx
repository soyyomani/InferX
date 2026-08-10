import { useState, useRef, useEffect, useCallback } from "react";
import "./ThreadPoolViz.css";

// Task definitions for attention demo
const ATTENTION_TASKS = [
  { id: 0, name: "Input Ready", deps: [], duration: 200, color: "#3b82f6" },
  { id: 1, name: "Compute Q", deps: [0], duration: 800, color: "#8b5cf6" },
  { id: 2, name: "Compute K", deps: [0], duration: 800, color: "#a78bfa" },
  { id: 3, name: "Compute V", deps: [0], duration: 800, color: "#c4b5fd" },
  { id: 4, name: "Attention", deps: [1, 2, 3], duration: 1200, color: "#f59e0b" },
  { id: 5, name: "FFN Layer", deps: [4], duration: 1000, color: "#22c55e" },
  { id: 6, name: "Output", deps: [5], duration: 200, color: "#06b6d4" },
];

const MLP_TASKS = [
  { id: 0, name: "Input", deps: [], duration: 100, color: "#3b82f6" },
  { id: 1, name: "MatMul₁ (rows 0-15)", deps: [0], duration: 600, color: "#8b5cf6" },
  { id: 2, name: "MatMul₁ (rows 16-31)", deps: [0], duration: 600, color: "#a78bfa" },
  { id: 3, name: "MatMul₁ (rows 32-47)", deps: [0], duration: 600, color: "#c4b5fd" },
  { id: 4, name: "MatMul₁ (rows 48-63)", deps: [0], duration: 600, color: "#ddd6fe" },
  { id: 5, name: "ReLU", deps: [1, 2, 3, 4], duration: 300, color: "#22c55e" },
  { id: 6, name: "MatMul₂", deps: [5], duration: 800, color: "#f59e0b" },
  { id: 7, name: "Softmax", deps: [6], duration: 200, color: "#ec4899" },
];

const NUM_WORKERS = 4;

export default function ThreadPoolViz() {
  const [scenario, setScenario] = useState("attention");
  const [isRunning, setIsRunning] = useState(false);
  const [workerStates, setWorkerStates] = useState(Array(NUM_WORKERS).fill(null));
  const [completedTasks, setCompletedTasks] = useState(new Set());
  const [taskTimeline, setTaskTimeline] = useState([]);
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef(null);
  const startTimeRef = useRef(0);

  const tasks = scenario === "attention" ? ATTENTION_TASKS : MLP_TASKS;

  const reset = useCallback(() => {
    setIsRunning(false);
    setWorkerStates(Array(NUM_WORKERS).fill(null));
    setCompletedTasks(new Set());
    setTaskTimeline([]);
    setElapsed(0);
    if (timerRef.current) clearInterval(timerRef.current);
  }, []);

  const runSimulation = useCallback(() => {
    reset();
    setIsRunning(true);
    startTimeRef.current = Date.now();

    const completed = new Set();
    const running = new Map(); // taskId → {worker, startTime}
    const timeline = [];
    const workers = Array(NUM_WORKERS).fill(null); // null = idle

    const tick = () => {
      const now = Date.now() - startTimeRef.current;
      setElapsed(now);

      // Check for completed tasks
      for (const [taskId, info] of running) {
        const task = tasks.find(t => t.id === taskId);
        if (now - info.startTime >= task.duration) {
          completed.add(taskId);
          workers[info.worker] = null;
          running.delete(taskId);
          timeline.push({ ...task, worker: info.worker, start: info.startTime, end: now });
        }
      }

      // Find ready tasks (all deps completed, not yet running/completed)
      const ready = tasks.filter(t =>
        !completed.has(t.id) &&
        !running.has(t.id) &&
        t.deps.every(d => completed.has(d))
      );

      // Assign ready tasks to idle workers
      for (const task of ready) {
        const idleWorker = workers.findIndex(w => w === null);
        if (idleWorker === -1) break; // No idle workers
        workers[idleWorker] = task;
        running.set(task.id, { worker: idleWorker, startTime: now });
      }

      // Update state
      setWorkerStates([...workers]);
      setCompletedTasks(new Set(completed));
      setTaskTimeline([...timeline]);

      // Check if all done
      if (completed.size === tasks.length) {
        setIsRunning(false);
        clearInterval(timerRef.current);
      }
    };

    timerRef.current = setInterval(tick, 50);
  }, [tasks, reset]);

  useEffect(() => () => { if (timerRef.current) clearInterval(timerRef.current); }, []);

  // Compute sequential time (sum of all durations)
  const seqTime = tasks.reduce((sum, t) => sum + t.duration, 0);
  // Parallel time = critical path
  const criticalPath = computeCriticalPath(tasks);
  const speedup = (seqTime / criticalPath).toFixed(1);

  return (
    <div className="tpviz">
      <div className="tpviz-header">
        <h1><span className="tpviz-icon">🔀</span> Thread Pool Visualizer</h1>
        <p>Watch tasks flow into worker threads — independent operations run in parallel</p>
      </div>

      {/* Scenario selector */}
      <div className="tpviz-controls">
        <button className={`tp-btn ${scenario === "attention" ? "active" : ""}`}
                onClick={() => { setScenario("attention"); reset(); }}>
          Attention (Q/K/V parallel)
        </button>
        <button className={`tp-btn ${scenario === "mlp" ? "active" : ""}`}
                onClick={() => { setScenario("mlp"); reset(); }}>
          MLP (row-parallel MatMul)
        </button>
        <button className="tp-btn tp-btn-play" onClick={runSimulation} disabled={isRunning}>
          {isRunning ? "Running..." : "▶ Execute"}
        </button>
        <button className="tp-btn tp-btn-reset" onClick={reset}>Reset</button>
      </div>

      {/* Worker Lanes */}
      <div className="tpviz-lanes">
        <div className="lanes-header">
          <span>Worker Threads ({NUM_WORKERS} cores)</span>
          <span className="lanes-time">{elapsed}ms</span>
        </div>
        {workerStates.map((task, i) => (
          <div key={i} className="worker-lane">
            <div className="lane-label">Thread {i}</div>
            <div className="lane-track">
              {task ? (
                <div className="lane-task active" style={{ backgroundColor: task.color }}>
                  {task.name}
                </div>
              ) : (
                <div className="lane-task idle">idle</div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Task DAG */}
      <div className="tpviz-dag">
        <div className="dag-title">Task Dependency Graph</div>
        <div className="dag-nodes">
          {tasks.map(task => {
            const isDone = completedTasks.has(task.id);
            const isActive = workerStates.some(w => w && w.id === task.id);
            return (
              <div
                key={task.id}
                className={`dag-node ${isDone ? "done" : ""} ${isActive ? "running" : ""}`}
                style={{ "--task-color": task.color }}
              >
                <div className="dag-node-name">{task.name}</div>
                <div className="dag-node-dur">{task.duration}ms</div>
                {task.deps.length > 0 && (
                  <div className="dag-node-deps">
                    deps: [{task.deps.join(", ")}]
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Timeline */}
      {taskTimeline.length > 0 && (
        <div className="tpviz-timeline">
          <div className="tl-title">Execution Timeline</div>
          <div className="tl-chart">
            {Array.from({ length: NUM_WORKERS }, (_, w) => (
              <div key={w} className="tl-row">
                <div className="tl-label">T{w}</div>
                <div className="tl-track">
                  {taskTimeline.filter(t => t.worker === w).map((t, i) => (
                    <div
                      key={i}
                      className="tl-block"
                      style={{
                        left: `${(t.start / seqTime) * 100}%`,
                        width: `${((t.end - t.start) / seqTime) * 100}%`,
                        backgroundColor: t.color,
                      }}
                      title={`${t.name}: ${t.start}-${t.end}ms`}
                    >
                      <span className="tl-block-label">{t.name}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Stats */}
      <div className="tpviz-stats">
        <div className="tps-card">
          <div className="tps-label">Sequential Time</div>
          <div className="tps-value">{seqTime}ms</div>
          <div className="tps-sub">if run on 1 thread</div>
        </div>
        <div className="tps-card">
          <div className="tps-label">Critical Path</div>
          <div className="tps-value">{criticalPath}ms</div>
          <div className="tps-sub">min possible time</div>
        </div>
        <div className="tps-card highlight">
          <div className="tps-label">Speedup</div>
          <div className="tps-value">{speedup}×</div>
          <div className="tps-sub">with {NUM_WORKERS} threads</div>
        </div>
        <div className="tps-card">
          <div className="tps-label">Workers</div>
          <div className="tps-value">{NUM_WORKERS}</div>
          <div className="tps-sub">dispatch: ~50ns</div>
        </div>
      </div>
    </div>
  );
}

// Compute critical path (longest path through DAG)
function computeCriticalPath(tasks) {
  const memo = {};
  function longest(id) {
    if (memo[id] !== undefined) return memo[id];
    const task = tasks.find(t => t.id === id);
    if (task.deps.length === 0) {
      memo[id] = task.duration;
      return task.duration;
    }
    const maxDep = Math.max(...task.deps.map(d => longest(d)));
    memo[id] = maxDep + task.duration;
    return memo[id];
  }
  return Math.max(...tasks.map(t => longest(t.id)));
}
