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
  // Transform data for Recharts
  const chartData = metrics.map((m) => ({
    step: m.step,
    train: +(m.trainAccuracy * 100).toFixed(1),
    test: +(m.testAccuracy * 100).toFixed(1),
    loss: m.trainLoss,
  }));

  const isEmpty = chartData.length === 0;

  return (
    <div className="glass-panel p-4 flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
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
            className="grok-indicator px-3 py-1 rounded-full text-xs font-bold"
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

      {/* Chart Container */}
      <div className="flex-1 min-h-0 relative w-full">
        {isEmpty ? (
          <div
            className="absolute inset-0 flex items-center justify-center text-sm"
            style={{ color: "var(--text-muted)" }}
          >
            Start a simulation to see the accuracy curves.
          </div>
        ) : (
          <ResponsiveContainer width="99%" height="99%">
            <LineChart
              data={chartData}
              margin={{ top: 5, right: 20, left: 0, bottom: 5 }}
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
                type="monotone"
                dataKey="train"
                stroke="var(--signal-train)"
                strokeWidth={2}
                dot={false}
                name="train"
              />
              <Line
                type="monotone"
                dataKey="test"
                stroke="var(--signal-test)"
                strokeWidth={2}
                dot={false}
                name="test"
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Legend */}
      <div className="flex gap-5 mt-2 justify-center">
        <div className="flex items-center gap-1.5">
          <div
            className="w-3 h-0.5 rounded-full"
            style={{ background: "var(--signal-train)" }}
          />
          <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
            Train Accuracy
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <div
            className="w-3 h-0.5 rounded-full"
            style={{ background: "var(--signal-test)" }}
          />
          <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
            Test Accuracy
          </span>
        </div>
      </div>
    </div>
  );
}
