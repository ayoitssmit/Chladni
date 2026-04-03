"use client";

import { useGrokkingStream } from "@/hooks/useGrokkingStream";
import ControlPanel from "@/components/ControlPanel";
import MetricsChart from "@/components/MetricsChart";
import EmbeddingSpace from "@/components/EmbeddingSpace";
import StatusBar from "@/components/StatusBar";
import PhaseDiagram from "@/components/PhaseDiagram";
import React from "react";

export default function Home() {
  const stream = useGrokkingStream();
  const [view, setView] = React.useState<"live" | "phase">("live");
  const [phaseData, setPhaseData] = React.useState<any[]>([]);
  const [isLoadingPhase, setIsLoadingPhase] = React.useState(false);

  React.useEffect(() => {
    if (view === "phase") {
      setIsLoadingPhase(true);
      fetch("http://localhost:8000/api/phase_diagram")
        .then((res) => res.json())
        .then((data) => {
          if (Array.isArray(data)) {
            setPhaseData(data);
          } else if (data && data.results && Array.isArray(data.results)) {
            setPhaseData(data.results);
          } else {
            setPhaseData([]);
          }
          setIsLoadingPhase(false);
        })
        .catch(() => setIsLoadingPhase(false));
    }
  }, [view]);

  const latestOperation =
    stream.metrics.length > 0
      ? stream.metrics[stream.metrics.length - 1].operation
      : "";

  const latestElapsed =
    stream.metrics.length > 0
      ? stream.metrics[stream.metrics.length - 1].elapsedSeconds
      : 0;

  const totalSteps =
    stream.metrics.length > 0
      ? stream.metrics[stream.metrics.length - 1].totalSteps
      : 50000;

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {/* ─── Top Bar ─── */}
      <header
        className="flex items-center justify-between px-5 py-3 shrink-0"
        style={{
          borderBottom: "1px solid var(--border-subtle)",
          background: "var(--surface-800)",
        }}
      >
        <div className="flex items-center gap-3">
          <div
            className="w-2 h-2 rounded-full"
            style={{
              background: stream.isRunning
                ? "var(--signal-grok)"
                : stream.grokked
                ? "var(--accent-primary)"
                : "var(--text-muted)",
              boxShadow: stream.isRunning
                ? "0 0 8px rgba(110, 224, 94, 0.5)"
                : "none",
            }}
          />
          <h1 className="text-base font-semibold tracking-wide">
            <span style={{ color: "var(--accent-primary)" }}>Chladni</span>
            <span
              className="ml-2 text-xs font-normal tracking-wider uppercase"
              style={{ color: "var(--text-muted)" }}
            >
              Grokking Simulation Platform
            </span>
          </h1>
        </div>

        {/* View Toggle */}
        <div className="flex bg-[var(--surface-900)] p-1 rounded-lg border border-[var(--border-subtle)]">
          <button
            onClick={() => setView("live")}
            className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-all ${
              view === "live"
                ? "bg-[var(--accent-primary)] text-black shadow-lg"
                : "text-[var(--text-secondary)] hover:text-white"
            }`}
          >
            Live Simulation
          </button>
          <button
            onClick={() => setView("phase")}
            className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-all ${
              view === "phase"
                ? "bg-[var(--accent-primary)] text-black shadow-lg"
                : "text-[var(--text-secondary)] hover:text-white"
            }`}
          >
            Phase Map
          </button>
        </div>
      </header>

      {/* ─── Main Content ─── */}
      <main className="flex-1 min-h-0 overflow-auto p-3">
        {view === "live" ? (
          <div 
            className="h-full gap-3 dashboard-grid grid grid-cols-1 md:grid-cols-[280px_1fr] grid-rows-[auto_1fr_1fr] md:grid-rows-[1fr_1fr]"
          >
            {/* Control Panel */}
            <div className="md:row-span-2 min-h-0">
              <ControlPanel
                isRunning={stream.isRunning}
                onStart={stream.start}
                onStop={stream.stop}
                onReset={stream.reset}
              />
            </div>

            {/* 3D Embedding Space */}
            <div className="min-h-[400px] md:min-h-0 flex flex-col">
              <EmbeddingSpace
                embeddings={stream.latestEmbeddings}
                grokked={stream.grokked}
              />
            </div>

            {/* Metrics Chart */}
            <div className="min-h-[300px] md:min-h-0 flex flex-col">
              <MetricsChart
                metrics={stream.metrics}
                grokked={stream.grokked}
                grokStep={stream.grokStep}
              />
            </div>
          </div>
        ) : (
          <div className="h-full">
            <PhaseDiagram data={phaseData} isLoading={isLoadingPhase} />
          </div>
        )}
      </main>

      {/* ─── Status Bar ─── */}
      <div className="shrink-0 p-3">
        <StatusBar
          currentStep={stream.currentStep}
          totalSteps={totalSteps}
          isRunning={stream.isRunning}
          grokked={stream.grokked}
          grokStep={stream.grokStep}
          elapsedSeconds={latestElapsed}
          operation={latestOperation}
          error={stream.error}
        />
      </div>
    </div>
  );
}
