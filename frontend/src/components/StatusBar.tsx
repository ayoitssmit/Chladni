"use client";

interface StatusBarProps {
  currentStep: number;
  totalSteps: number;
  isRunning: boolean;
  grokked: boolean;
  grokStep: number | null;
  elapsedSeconds: number;
  operation: string;
  error: string | null;
}

const OPERATION_LABELS: Record<string, string> = {
  addition: "(a + b) mod 97",
  subtraction: "(a − b) mod 97",
  multiplication: "(a × b) mod 97",
  polynomial: "(a² + ab + b²) mod 97",
  division: "(a / b) mod 97",
};

export default function StatusBar({
  currentStep,
  totalSteps,
  isRunning,
  grokked,
  grokStep,
  elapsedSeconds,
  operation,
  error,
}: StatusBarProps) {
  const progress =
    totalSteps > 0 ? Math.min((currentStep / totalSteps) * 100, 100) : 0;

  const formatTime = (s: number) => {
    if (s < 60) return `${s.toFixed(0)}s`;
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}m ${sec}s`;
  };

  return (
    <div
      className="glass-panel px-4 py-2.5 flex flex-wrap items-center gap-x-6 gap-y-1"
    >
      {/* Progress bar */}
      <div className="flex items-center gap-2 flex-1 min-w-[180px]">
        <div
          className="flex-1 h-1.5 rounded-full overflow-hidden"
          style={{ background: "var(--surface-500)" }}
        >
          <div
            className="h-full rounded-full transition-all duration-300 ease-out"
            style={{
              width: `${progress}%`,
              background: grokked
                ? "var(--signal-grok)"
                : "var(--accent-primary)",
            }}
          />
        </div>
        <span
          className="text-xs font-mono whitespace-nowrap"
          style={{ color: "var(--text-secondary)" }}
        >
          {currentStep.toLocaleString()} / {totalSteps.toLocaleString()}
        </span>
      </div>

      {/* Operation */}
      {operation && (
        <span
          className="text-xs font-mono"
          style={{ color: "var(--text-muted)" }}
        >
          {OPERATION_LABELS[operation] || operation}
        </span>
      )}

      {/* Elapsed */}
      {elapsedSeconds > 0 && (
        <span
          className="text-xs font-mono"
          style={{ color: "var(--text-muted)" }}
        >
          {formatTime(elapsedSeconds)}
        </span>
      )}

      {/* Status */}
      {error ? (
        <span
          className="text-xs font-semibold"
          style={{ color: "var(--signal-train)" }}
        >
          Error: {error}
        </span>
      ) : grokked ? (
        <span
          className="grok-indicator text-xs font-bold px-2 py-0.5 rounded-full"
          style={{
            background: "rgba(110, 224, 94, 0.15)",
            color: "var(--signal-grok)",
            border: "1px solid rgba(110, 224, 94, 0.3)",
          }}
        >
          GROKKED at step {grokStep?.toLocaleString()}
        </span>
      ) : isRunning ? (
        <span
          className="text-xs font-medium"
          style={{ color: "var(--accent-primary)" }}
        >
          Training...
        </span>
      ) : currentStep > 0 ? (
        <span
          className="text-xs"
          style={{ color: "var(--text-muted)" }}
        >
          Completed
        </span>
      ) : null}
    </div>
  );
}
