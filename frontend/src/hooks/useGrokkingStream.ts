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
}

export interface StreamState {
  isRunning: boolean;
  metrics: MetricSnapshot[];
  latestEmbeddings: number[][];
  grokked: boolean;
  grokStep: number | null;
  currentStep: number;
  error: string | null;
}

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

/* ─── Hook ─── */
export function useGrokkingStream() {
  const [state, setState] = useState<StreamState>({
    isRunning: false,
    metrics: [],
    latestEmbeddings: [],
    grokked: false,
    grokStep: null,
    currentStep: 0,
    error: null,
  });

  const wsRef = useRef<WebSocket | null>(null);

  const start = useCallback((params: SimulationParams) => {
    // Close any existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    // Reset state
    setState({
      isRunning: true,
      metrics: [],
      latestEmbeddings: [],
      grokked: false,
      grokStep: null,
      currentStep: 0,
      error: null,
    });

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
    setState({
      isRunning: false,
      metrics: [],
      latestEmbeddings: [],
      grokked: false,
      grokStep: null,
      currentStep: 0,
      error: null,
    });
  }, []);

  return { ...state, start, stop, reset };
}
