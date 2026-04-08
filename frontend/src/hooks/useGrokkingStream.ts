"use client";

import { useState, useRef, useCallback } from "react";

/* ─── Types ─── */
export interface SimulationParams {
  operation: string;
  fraction: number;
  weightDecay: number;
  totalSteps: number;
  lr: number;
}

export interface MetricSnapshot {
  step: number;
  totalSteps: number;
  operation: string;
  trainLoss: number;
  trainAccuracy: number;
  testAccuracy: number;
  pcaEmbeddings: number[][];
  grokked: boolean;
  elapsedSeconds: number;
  finished?: boolean;
  stopped?: boolean;
  fftSignal?: number;
  predictedGrokStep?: number;
  interventionTriggered?: boolean;
}

export interface StreamState {
  isRunning: boolean;
  metrics: MetricSnapshot[];
  latestEmbeddings: number[][];
  grokked: boolean;
  grokStep: number | null;
  currentStep: number;
  error: string | null;
  predictedGrokStep: number | null;
  fftSignal: number;
  interventionTriggered: boolean;
}

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

const INITIAL_STATE: StreamState = {
  isRunning: false,
  metrics: [],
  latestEmbeddings: [],
  grokked: false,
  grokStep: null,
  currentStep: 0,
  error: null,
  predictedGrokStep: null,
  fftSignal: 0,
  interventionTriggered: false,
};

/* ─── Hook ─── */
export function useGrokkingStream() {
  const [state, setState] = useState<StreamState>({ ...INITIAL_STATE });

  const wsRef = useRef<WebSocket | null>(null);

  const start = useCallback((params: SimulationParams) => {
    // Close any existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    // Reset state
    setState({ ...INITIAL_STATE, isRunning: true });

    // Build WebSocket URL with query params
    const queryStr = new URLSearchParams({
      operation: params.operation,
      fraction: params.fraction.toString(),
      weight_decay: params.weightDecay.toString(),
      total_steps: params.totalSteps.toString(),
      lr: params.lr.toString(),
    }).toString();

    const url = `${WS_BASE}/ws/simulate?${queryStr}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("[ws] Connected:", url);
    };

    ws.onmessage = (event) => {
      try {
        const raw = JSON.parse(event.data);

        // Check for server error
        if (raw.error) {
          setState((prev) => ({
            ...prev,
            isRunning: false,
            error: raw.error,
          }));
          ws.close();
          return;
        }

        // Check for stop acknowledgement
        if (raw.stopped) {
          setState((prev) => ({ ...prev, isRunning: false }));
          return;
        }

        // Parse the snapshot
        const snapshot: MetricSnapshot = {
          step: raw.step,
          totalSteps: raw.total_steps,
          operation: raw.operation,
          trainLoss: raw.train_loss,
          trainAccuracy: raw.train_accuracy,
          testAccuracy: raw.test_accuracy,
          pcaEmbeddings: raw.pca_embeddings,
          grokked: raw.grokked,
          elapsedSeconds: raw.elapsed_seconds,
          finished: raw.finished,
          fftSignal: raw.fft_signal ?? 0,
          predictedGrokStep: raw.predicted_grok_step ?? -1,
          interventionTriggered: raw.intervention_triggered ?? false,
        };

        setState((prev) => {
          const newMetrics = [...prev.metrics, snapshot];
          return {
            ...prev,
            metrics: newMetrics,
            latestEmbeddings: snapshot.pcaEmbeddings,
            grokked: snapshot.grokked || prev.grokked,
            grokStep:
              snapshot.grokked && !prev.grokked
                ? snapshot.step
                : prev.grokStep,
            currentStep: snapshot.step,
            isRunning: !snapshot.finished,
            predictedGrokStep:
              snapshot.predictedGrokStep && snapshot.predictedGrokStep > 0
                ? snapshot.predictedGrokStep
                : prev.predictedGrokStep,
            fftSignal: snapshot.fftSignal ?? prev.fftSignal,
            interventionTriggered: snapshot.interventionTriggered || prev.interventionTriggered,
          };
        });
      } catch (err) {
        console.error("[ws] Failed to parse message:", err);
      }
    };

    ws.onerror = (event) => {
      console.error("[ws] Error:", event);
      setState((prev) => ({
        ...prev,
        isRunning: false,
        error: "WebSocket connection failed. Is the backend running?",
      }));
    };

    ws.onclose = () => {
      console.log("[ws] Connection closed");
      setState((prev) => ({ ...prev, isRunning: false }));
    };
  }, []);

  const stop = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send("stop");
    }
  }, []);

  const reset = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    setState({ ...INITIAL_STATE });
  }, []);

  const intervene = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send("intervene");
      setState((prev) => ({ ...prev, interventionTriggered: true }));
    }
  }, []);

  return { ...state, start, stop, reset, intervene };
}
