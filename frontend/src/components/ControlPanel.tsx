"use client";

import { SimulationParams } from "@/hooks/useGrokkingStream";
import { useState } from "react";

const OPERATIONS = [
  { value: "addition", label: "(a + b) mod 97" },
  { value: "subtraction", label: "(a − b) mod 97" },
  { value: "multiplication", label: "(a × b) mod 97" },
  { value: "polynomial", label: "(a² + ab + b²) mod 97" },
  { value: "division", label: "(a / b) mod 97" },
];

interface ControlPanelProps {
  isRunning: boolean;
  onStart: (params: SimulationParams) => void;
  onStop: () => void;
  onReset: () => void;
}

export default function ControlPanel({
  isRunning,
  onStart,
  onStop,
  onReset,
}: ControlPanelProps) {
  const [operation, setOperation] = useState("addition");
  const [fraction, setFraction] = useState(0.4);
  const [weightDecay, setWeightDecay] = useState(1.0);
  const [totalSteps, setTotalSteps] = useState(50000);
  const [lr, setLr] = useState(0.001);

  const handleStart = () => {
    onStart({ operation, fraction, weightDecay, totalSteps, lr });
  };

  return (
    <div className="glass-panel p-5 flex flex-col gap-5 h-full overflow-y-auto accent-scrollbar">
      {/* Header */}
      <div>
        <h2
          className="text-sm font-semibold tracking-wider uppercase"
          style={{ color: "var(--accent-primary)" }}
        >
          Simulation Controls
        </h2>
        <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
          Configure hyperparameters and start a live training run.
        </p>
      </div>

      {/* Operation selector */}
      <div className="flex flex-col gap-1.5">
        <label
          className="text-xs font-medium"
          style={{ color: "var(--text-secondary)" }}
        >
          Arithmetic Task
        </label>
        <select
          value={operation}
          onChange={(e) => setOperation(e.target.value)}
          disabled={isRunning}
          className="w-full px-3 py-2 rounded-lg text-sm outline-none transition-colors"
          style={{
            background: "var(--surface-600)",
            color: "var(--text-primary)",
            border: "1px solid var(--border-subtle)",
          }}
        >
          {OPERATIONS.map((op) => (
            <option key={op.value} value={op.value}>
              {op.label}
            </option>
          ))}
        </select>
      </div>

      {/* Fraction slider */}
      <div className="flex flex-col gap-1.5">
        <div className="flex justify-between items-center">
          <label
            className="text-xs font-medium"
            style={{ color: "var(--text-secondary)" }}
          >
            Dataset Fraction
          </label>
          <span
            className="text-xs font-mono"
            style={{ color: "var(--accent-secondary)" }}
          >
            {(fraction * 100).toFixed(0)}%
          </span>
        </div>
        <input
          type="range"
          min={0.05}
          max={0.95}
          step={0.05}
          value={fraction}
          onChange={(e) => setFraction(parseFloat(e.target.value))}
          disabled={isRunning}
          className="slider-track"
        />
        <div
          className="flex justify-between text-xs"
          style={{ color: "var(--text-muted)" }}
        >
          <span>5%</span>
          <span>95%</span>
        </div>
      </div>

      {/* Weight Decay slider */}
      <div className="flex flex-col gap-1.5">
        <div className="flex justify-between items-center">
          <label
            className="text-xs font-medium"
            style={{ color: "var(--text-secondary)" }}
          >
            Weight Decay
          </label>
          <span
            className="text-xs font-mono"
            style={{ color: "var(--accent-secondary)" }}
          >
            {weightDecay.toFixed(2)}
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={5}
          step={0.1}
          value={weightDecay}
          onChange={(e) => setWeightDecay(parseFloat(e.target.value))}
          disabled={isRunning}
          className="slider-track"
        />
        <div
          className="flex justify-between text-xs"
          style={{ color: "var(--text-muted)" }}
        >
          <span>0.0</span>
          <span>5.0</span>
        </div>
      </div>

      {/* Total Steps */}
      <div className="flex flex-col gap-1.5">
        <label
          className="text-xs font-medium"
          style={{ color: "var(--text-secondary)" }}
        >
          Total Steps
        </label>
        <input
          type="number"
          value={totalSteps}
          onChange={(e) =>
            setTotalSteps(Math.max(100, parseInt(e.target.value) || 100))
          }
          disabled={isRunning}
          className="w-full px-3 py-2 rounded-lg text-sm font-mono outline-none"
          style={{
            background: "var(--surface-600)",
            color: "var(--text-primary)",
            border: "1px solid var(--border-subtle)",
          }}
        />
      </div>

      {/* Learning Rate */}
      <div className="flex flex-col gap-1.5">
        <label
          className="text-xs font-medium"
          style={{ color: "var(--text-secondary)" }}
        >
          Learning Rate
        </label>
        <input
          type="number"
          value={lr}
          onChange={(e) => setLr(parseFloat(e.target.value) || 0.001)}
          step={0.0001}
          min={0.0001}
          max={0.1}
          disabled={isRunning}
          className="w-full px-3 py-2 rounded-lg text-sm font-mono outline-none"
          style={{
            background: "var(--surface-600)",
            color: "var(--text-primary)",
            border: "1px solid var(--border-subtle)",
          }}
        />
      </div>

      {/* Buttons */}
      <div className="flex flex-col gap-2 mt-auto">
        {!isRunning ? (
          <button
            onClick={handleStart}
            className="w-full py-2.5 rounded-lg text-sm font-semibold tracking-wide transition-all duration-200 cursor-pointer"
            style={{
              background: "var(--accent-primary)",
              color: "var(--surface-900)",
            }}
            onMouseEnter={(e) =>
              (e.currentTarget.style.background = "var(--accent-secondary)")
            }
            onMouseLeave={(e) =>
              (e.currentTarget.style.background = "var(--accent-primary)")
            }
          >
            Start Simulation
          </button>
        ) : (
          <button
            onClick={onStop}
            className="w-full py-2.5 rounded-lg text-sm font-semibold tracking-wide transition-all duration-200 cursor-pointer"
            style={{
              background: "var(--signal-train)",
              color: "var(--surface-900)",
            }}
          >
            Stop Simulation
          </button>
        )}
        <button
          onClick={onReset}
          disabled={isRunning}
          className="w-full py-2 rounded-lg text-xs font-medium transition-all duration-200 cursor-pointer disabled:opacity-30"
          style={{
            background: "var(--surface-600)",
            color: "var(--text-secondary)",
            border: "1px solid var(--border-subtle)",
          }}
        >
          Reset
        </button>
      </div>
    </div>
  );
}
