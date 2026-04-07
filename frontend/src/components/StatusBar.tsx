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
  predictedGrokStep?: number | null;
  fftSignal?: number;
}

const OPERATION_LABELS: Record<string, string> = {
  addition: "(a + b) mod 97",
  subtraction: "(a - b) mod 97",
  multiplication: "(a x b) mod 97",
  polynomial: "(a^2 + ab + b^2) mod 97",
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
  predictedGrokStep,
  fftSignal = 0,
}: StatusBarProps) {
  const progress =
    totalSteps > 0 ? Math.min((currentStep / totalSteps) * 100, 100) : 0;

  const formatTime = (s: number) => {
    if (s < 60) return `${s.toFixed(0)}s`;
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}m ${sec}s`;
  };

  const hasPrediction =
    predictedGrokStep != null && predictedGrokStep > 0 && !grokked;
  const stepsUntilPredicted = hasPrediction
    ? Math.max(0, predictedGrokStep! - currentStep)
    : null;

  // Post-grokking validation
  const predictionWasGiven =
    grokked && predictedGrokStep != null && predictedGrokStep > 0 && grokStep != null;
  const predictionError = predictionWasGiven
    ? Math.abs(predictedGrokStep! - grokStep!)
    : null;

  return (
    <div className="glass-panel px-4 py-2.5 flex flex-wrap items-center gap-x-6 gap-y-1">
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

      {/* FFT Signal bar */}
      {isRunning && currentStep > 0 && !grokked && (
        <div className="flex items-center gap-1.5">
          <span
            className="text-[10px] font-mono uppercase tracking-wider"
            style={{ color: "var(--text-muted)" }}
          >
            FFT {fftSignal.toFixed(1)}%
          </span>
          <div
            className="w-16 h-1.5 rounded-full overflow-hidden"
            style={{ background: "var(--surface-500)" }}
          >
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${Math.min(100, Math.max(1, fftSignal * 4))}%`,
                background: fftSignal > 10
                  ? "#f59e0b"
                  : "var(--accent-primary)",
              }}
            />
          </div>
        </div>
      )}

      {/* Prediction pill */}
      {hasPrediction && stepsUntilPredicted != null && stepsUntilPredicted > 0 && (
        <span
          className="text-xs font-semibold px-2.5 py-0.5 rounded-full"
          style={{
            background: "rgba(212, 160, 84, 0.12)",
            color: "var(--accent-primary)",
            border: "1px solid rgba(212, 160, 84, 0.35)",
          }}
        >
          Grokking predicted in ~{stepsUntilPredicted.toLocaleString()} steps
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
        <div className="flex items-center gap-2">
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
          {predictionError != null && (
            <span
              className="text-xs px-2 py-0.5 rounded-full"
              style={{
                background:
                  predictionError < 500
                    ? "rgba(110, 224, 94, 0.1)"
                    : "rgba(212, 160, 84, 0.1)",
                color:
                  predictionError < 500
                    ? "var(--signal-grok)"
                    : "var(--accent-secondary)",
                border:
                  predictionError < 500
                    ? "1px solid rgba(110, 224, 94, 0.3)"
                    : "1px solid rgba(212, 160, 84, 0.3)",
              }}
            >
              {predictionError < 500
                ? `Predicted within ${predictionError.toLocaleString()} steps`
                : `Prediction off by ${predictionError.toLocaleString()} steps`}
            </span>
          )}
        </div>
      ) : isRunning ? (
        <span
          className="text-xs font-medium"
          style={{ color: "var(--accent-primary)" }}
        >
          Training...
        </span>
      ) : currentStep > 0 ? (
        <span className="text-xs" style={{ color: "var(--text-muted)" }}>
          Completed
        </span>
      ) : null}
    </div>
  );
}
