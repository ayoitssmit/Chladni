"use client";

import { MetricSnapshot } from "@/hooks/useGrokkingStream";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

interface MetricsChartProps {
  metrics: MetricSnapshot[];
  grokked: boolean;
  grokStep: number | null;
}

export default function MetricsChart({
  metrics,
  grokked,
  grokStep,
}: MetricsChartProps) {
  const chartData = metrics.map((m) => ({
    step: m.step,
    train: +(m.trainAccuracy * 100).toFixed(1),
    test: +(m.testAccuracy * 100).toFixed(1),
  }));

  const isEmpty = chartData.length === 0;

  return (
    <div className="glass-panel p-4 flex flex-col h-full overflow-hidden relative">
      {/* Header */}
      <div className="flex items-center justify-between mb-2 shrink-0">
        <div>
          <h3
            className="text-sm font-semibold tracking-wider uppercase"
            style={{ color: "var(--accent-primary)" }}
          >
            Accuracy Curves
          </h3>
          <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
            Train vs Test accuracy over training steps
          </p>
        </div>
        {grokked && (
          <div
            className="grok-indicator px-3 py-1 rounded-full text-xs font-bold shrink-0"
            style={{
              background: "rgba(110, 224, 94, 0.15)",
              color: "var(--signal-grok)",
              border: "1px solid rgba(110, 224, 94, 0.3)",
            }}
          >
            GROKKED @ Step {grokStep?.toLocaleString()}
          </div>
        )}
      </div>

      {/* Chart */}
      <div className="flex-1 relative" style={{ minHeight: 150 }}>
        {isEmpty ? (
          <div
            className="absolute inset-0 flex items-center justify-center text-sm"
            style={{ color: "var(--text-muted)" }}
          >
            Start a simulation to see the accuracy curves.
          </div>
        ) : (
          <div className="absolute inset-0">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
              data={chartData}
              margin={{ top: 5, right: 15, left: 0, bottom: 35 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(255,255,255,0.04)"
              />
              <XAxis
                dataKey="step"
                stroke="var(--text-muted)"
                tick={{ fontSize: 10 }}
                tickFormatter={(v) =>
                  v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v
                }
              />
              <YAxis
                domain={[0, 100]}
                stroke="var(--text-muted)"
                tick={{ fontSize: 10 }}
                tickFormatter={(v) => `${v}%`}
              />
              <Tooltip
                contentStyle={{
                  background: "var(--surface-700)",
                  border: "1px solid var(--border-subtle)",
                  borderRadius: 8,
                  fontSize: 12,
                }}
                labelFormatter={(v) => `Step ${v.toLocaleString()}`}
                formatter={(value: any, name: any) => [
                  `${Number(value).toFixed(1)}%`,
                  name === "train" ? "Train Accuracy" : "Test Accuracy",
                ]}
              />
              {grokStep && (
                <ReferenceLine
                  x={grokStep}
                  stroke="var(--signal-grok)"
                  strokeDasharray="4 4"
                  strokeWidth={1.5}
                />
              )}
              <Line
                type="linear"
                dataKey="train"
                stroke="var(--signal-train)"
                strokeWidth={2}
                dot={false}
                name="train"
                isAnimationActive={false}
              />
              <Line
                type="linear"
                dataKey="test"
                stroke="var(--signal-test)"
                strokeWidth={2}
                dot={false}
                name="test"
                isAnimationActive={false}
              />
            </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="flex gap-5 mt-1 justify-center shrink-0">
        <div className="flex items-center gap-1.5">
          <div
            className="w-3 h-0.5 rounded-full"
            style={{ background: "var(--signal-train)" }}
          />
          <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
            Train
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <div
            className="w-3 h-0.5 rounded-full"
            style={{ background: "var(--signal-test)" }}
          />
          <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
            Test
          </span>
        </div>
      </div>
    </div>
  );
}
