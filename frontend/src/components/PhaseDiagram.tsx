"use client";

import React, { useMemo } from "react";

interface PhasePoint {
  fraction: number;
  weight_decay: number;
  grok_step: number;
}

interface PhaseDiagramProps {
  data: PhasePoint[];
  isLoading?: boolean;
}

export default function PhaseDiagram({ data, isLoading }: PhaseDiagramProps) {
  // Dynamically extract unique grid values from the data and sort them
  const fractions = useMemo(() => {
    if (!Array.isArray(data)) return [];
    return Array.from(new Set(data.map(p => p.fraction))).sort((a, b) => a - b);
  }, [data]);

  const decays = useMemo(() => {
    if (!Array.isArray(data)) return [];
    return Array.from(new Set(data.map(p => p.weight_decay))).sort((a, b) => a - b);
  }, [data]);

  // Map data to a grid for easy lookup
  const grid = useMemo(() => {
    const map: Record<string, number> = {};
    if (!Array.isArray(data)) return map;
    data.forEach((p) => {
      const key = `${p.fraction}-${p.weight_decay}`;
      map[key] = p.grok_step;
    });
    return map;
  }, [data]);

  // Color mapping: Dark/Gold for fast grokking, Dark for no grokking
  const getCellColor = (grokStep: number) => {
    if (grokStep === -1) return "var(--surface-900)"; // No grokking
    
    // Calculate intensity (lower is faster/brighter)
    const intensity = Math.max(0, 1 - grokStep / 50000);
    return `rgba(212, 160, 84, ${0.1 + intensity * 0.9})`; 
  };

  if (isLoading) {
    return (
      <div className="glass-panel h-full flex items-center justify-center animate-pulse">
        <span className="text-sm text-muted">Loading phase data...</span>
      </div>
    );
  }

  return (
    <div className="glass-panel p-6 h-full flex flex-col">
      <div className="mb-6">
        <h3 className="text-sm font-semibold tracking-wider uppercase text-[var(--accent-primary)]">
          Grokking Phase Diagram
        </h3>
        <p className="text-xs text-muted mt-1">
          Heatmap showing grokking step vs dataset size & weight decay
        </p>
      </div>

      <div className="flex-1 flex flex-col justify-center items-center">
        {/* Heatmap Grid */}
        <div className="relative">
          {/* Y Axis Label */}
          <div className="absolute -left-12 top-1/2 -rotate-90 origin-center text-[10px] text-muted uppercase tracking-widest whitespace-nowrap">
            Weight Decay
          </div>

          <div className="grid gap-1" style={{ 
            gridTemplateColumns: `repeat(${fractions.length}, 30px)`,
            gridTemplateRows: `repeat(${decays.length}, 30px)`
          }}>
            {decays.slice().reverse().map((decay) => (
              fractions.map((fraction) => {
                const grokStep = grid[`${fraction}-${decay}`] ?? -1;
                return (
                  <div
                    key={`${fraction}-${decay}`}
                    className="w-[30px] h-[30px] rounded-sm transition-all hover:scale-110 cursor-help border border-white/5"
                    style={{ background: getCellColor(grokStep) }}
                    title={`Fraction: ${fraction}, Decay: ${decay}, Grok Step: ${grokStep === -1 ? "Never" : grokStep}`}
                  />
                );
              })
            ))}
          </div>

          {/* X Axis Label */}
          <div className="mt-4 text-center text-[10px] text-muted uppercase tracking-widest">
            Dataset Fraction
          </div>
        </div>

        {/* Legend */}
        <div className="mt-8 flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-[var(--surface-900)] border border-white/10 rounded-sm" />
            <span className="text-[10px] text-muted">Never Grokked</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-12 h-3 rounded-sm" style={{ background: "linear-gradient(to right, rgba(212,160,84,0.1), rgba(212,160,84,1))" }} />
            <span className="text-[10px] text-muted">Slower → Faster</span>
          </div>
        </div>
      </div>
    </div>
  );
}
