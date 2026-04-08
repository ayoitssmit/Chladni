"""
FastAPI server with WebSocket streaming for live grokking simulation.

Endpoints:
  - GET  /                          → Health check
  - GET  /api/phase_diagram         → Serve pre-computed phase diagram data
  - WS   /ws/simulate               → Start a training run and stream metrics
"""

import json
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from train import train_generator
from dataset import OPERATION_LABELS

app = FastAPI(
    title="Grokking Simulation API",
    description="Real-time neural network grokking visualization backend",
    version="1.0.0",
)

# CORS — allow the React frontend to connect from any origin during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "grokking-simulation-api"}


@app.get("/api/operations")
async def get_operations():
    """Return the list of supported arithmetic operations and their labels."""
    return {
        "operations": [
            {"value": key, "label": label}
            for key, label in OPERATION_LABELS.items()
        ]
    }


@app.websocket("/ws/simulate")
async def websocket_simulate(
    websocket: WebSocket,
    fraction: float = Query(0.5, ge=0.05, le=0.95),
    weight_decay: float = Query(1.0, ge=0.0, le=10.0),
    total_steps: int = Query(50000, ge=100, le=150000),
    lr: float = Query(1e-3, gt=0, le=0.1),
    operation: str = Query("addition"),
):
    """
    WebSocket endpoint that streams training metrics in real-time.

    Query parameters:
      - fraction:     Dataset fraction for training (0.05 - 0.95)
      - weight_decay: L2 regularization (0.0 - 10.0)
      - total_steps:  Max training steps (1000 - 150000)
      - lr:           Learning rate
      - operation:    Arithmetic task (addition, subtraction, multiplication,
                      polynomial, division)

    The server sends a JSON message every ~100 steps containing:
      { step, total_steps, operation, train_loss, train_accuracy, test_accuracy,
        pca_embeddings, grokked, elapsed_seconds }
    """
    await websocket.accept()

    # Validate operation
    if operation not in OPERATION_LABELS:
        await websocket.send_json({
            "error": f"Unknown operation '{operation}'. "
                     f"Choose from: {list(OPERATION_LABELS.keys())}"
        })
        await websocket.close()
        return

    print(f"[ws] Client connected: op={operation}, fraction={fraction}, "
          f"wd={weight_decay}, steps={total_steps}, lr={lr}")

    last_payload = None
    grok_step = -1
    shared_state = {"intervene": False}
    
    try:
        # Run training in a thread to avoid blocking the event loop
        generator = train_generator(
            fraction=fraction,
            weight_decay=weight_decay,
            lr=lr,
            total_steps=total_steps,
            log_every=100,
            operation=operation,
            shared_state=shared_state,
        )

        for payload in generator:
            last_payload = payload
            if payload.get("grokked") and grok_step == -1:
                grok_step = payload.get("step", -1)

            # Send the snapshot as JSON
            await websocket.send_json(payload)

            # Small yield to keep the event loop responsive
            await asyncio.sleep(0.01)

            # Check if client sent a "stop" command
            try:
                msg = await asyncio.wait_for(
                    websocket.receive_text(), timeout=0.001
                )
                if msg == "stop":
                    print(f"[ws] Client requested stop at step {payload['step']}")
                    await websocket.send_json({
                        "stopped": True,
                        "step": payload["step"],
                    })
                    break
                elif msg == "intervene":
                    print(f"[ws] Client triggered early EWS intervention")
                    shared_state["intervene"] = True
            except asyncio.TimeoutError:
                pass  # No message from client, continue training

        print("[ws] Training complete, closing connection")

    except WebSocketDisconnect:
        print("[ws] Client disconnected")
    except Exception as e:
        print(f"[ws] Error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
        pass


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
