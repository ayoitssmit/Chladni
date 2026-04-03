"use client";

import { useGrokkingStream } from "@/hooks/useGrokkingStream";
import ControlPanel from "@/components/ControlPanel";
import MetricsChart from "@/components/MetricsChart";
import EmbeddingSpace from "@/components/EmbeddingSpace";
import StatusBar from "@/components/StatusBar";

export default function Home() {
  const stream = useGrokkingStream();

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
      </header>

      {/* ─── Main Dashboard Grid ─── */}
      <main className="flex-1 min-h-0 p-3 gap-3 dashboard-grid"
        style={{
          display: "grid",
          gridTemplateColumns: "280px 1fr",
          gridTemplateRows: "1fr 1fr",
        }}
      >
        {/* Control Panel — spans 2 rows on desktop */}
        <div className="row-span-2 min-h-0">
          <ControlPanel
            isRunning={stream.isRunning}
            onStart={stream.start}
            onStop={stream.stop}
            onReset={stream.reset}
          />
        </div>

        {/* 3D Embedding Space */}
        <div className="min-h-0 flex flex-col">
          <EmbeddingSpace
            embeddings={stream.latestEmbeddings}
            grokked={stream.grokked}
          />
        </div>

        {/* Metrics Chart */}
        <div className="min-h-0 flex flex-col">
          <MetricsChart
            metrics={stream.metrics}
            grokked={stream.grokked}
            grokStep={stream.grokStep}
          />
        </div>
      </main>

      {/* ─── Status Bar ─── */}
      <div className="shrink-0 px-3 pb-3">
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
