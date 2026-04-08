"use client";

import { EmbeddingVisualizer } from "./EmbeddingVisualizer";

interface EmbeddingSpaceProps {
  embeddings: number[][];
  grokked: boolean;
  isFullscreen: boolean;
  onToggleFullscreen: () => void;
  currentStep?: number;
  totalSteps?: number;
  isRunning?: boolean;
}

export default function EmbeddingSpace({
  embeddings,
  grokked,
  isFullscreen,
  onToggleFullscreen,
  currentStep,
  totalSteps,
  isRunning,
}: EmbeddingSpaceProps) {
  return (
    <div
      className={
        isFullscreen
          ? "fixed inset-0 z-50 bg-black overflow-hidden flex flex-col"
          : "glass-panel p-4 flex flex-col h-full overflow-hidden relative group"
      }
    >
      {/* Dashboard Mode Header */}
      {!isFullscreen && (
        <div className="flex items-center justify-between mb-2">
          <div>
            <h3
              className="text-sm font-semibold tracking-wider uppercase flex items-center gap-2"
              style={{ color: "var(--accent-primary)" }}
            >
              Token Embedding Space
              <button
                onClick={onToggleFullscreen}
                className="p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity"
                style={{
                  color: "var(--text-muted)",
                  background: "var(--surface-900)",
                  border: "1px solid var(--surface-500)",
                }}
                title="Fullscreen Mode"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="15 3 21 3 21 9"></polyline>
                  <polyline points="9 21 3 21 3 15"></polyline>
                  <line x1="21" y1="3" x2="14" y2="10"></line>
                  <line x1="3" y1="21" x2="10" y2="14"></line>
                </svg>
              </button>
            </h3>
            <p
              className="text-xs mt-0.5"
              style={{ color: "var(--text-muted)" }}
            >
              97 tokens projected via PCA — drag to rotate
            </p>
          </div>
          {grokked && (
            <div
              className="text-xs font-mono px-2 py-0.5 rounded"
              style={{
                color: "var(--signal-grok)",
                background: "rgba(110, 224, 94, 0.1)",
              }}
            >
              Ring Formed
            </div>
          )}
        </div>
      )}

      {/* Fullscreen Mode Overlays */}
      {isFullscreen && (
        <>
          <div className="absolute top-6 left-6 z-10 glass-panel px-4 py-3 flex flex-col gap-1 max-w-[300px]">
            <h1 className="text-sm font-semibold tracking-widest uppercase text-[#e8c47a]">
              3D Visualizer Mode
            </h1>
            <div className="flex items-center gap-2 mt-1">
              <div className="text-xs text-gray-400 font-mono">
                Step {currentStep?.toLocaleString()} / {totalSteps?.toLocaleString()}
              </div>
              {isRunning && (
                <span className="flex h-2 w-2 relative">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                </span>
              )}
            </div>
            
            {grokked && (
              <div className="mt-2 text-xs font-bold px-2 py-1 rounded bg-[#6ee05e]/20 text-[#6ee05e] border border-[#6ee05e]/30 inline-block text-center w-fit">
                STRUCTURE FORMED
              </div>
            )}
          </div>

          <div className="absolute top-6 right-6 z-10">
            <button 
              onClick={onToggleFullscreen}
              className="p-2 px-3 rounded glass-panel hover:bg-white/5 text-gray-300 text-xs font-semibold tracking-wider uppercase transition-colors flex items-center gap-2"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
              Close
            </button>
          </div>

          <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10 text-xs text-gray-500 tracking-wider pointer-events-none">
            Drag to rotate • Scroll to zoom
          </div>
        </>
      )}

      {/* Shared Canvas Container */}
      <div
        className={
          isFullscreen
            ? "absolute inset-0 z-0"
            : "flex-1 three-canvas-container rounded-lg overflow-hidden relative"
        }
        style={!isFullscreen ? { background: "var(--surface-900)", minHeight: 0 } : {}}
      >
        {embeddings.length === 0 ? (
          <div
            className="h-full flex items-center justify-center text-sm"
            style={{ color: "var(--text-muted)" }}
          >
            Start a simulation to see the embedding geometry.
          </div>
        ) : (
          <EmbeddingVisualizer embeddings={embeddings} />
        )}
      </div>
    </div>
  );
}
