"""
WebSocket test client for the Grokking Simulation API.

Tests that the server correctly accepts the `operation` parameter 
and streams training snapshots for each supported arithmetic task.

Usage:
    python test_ws.py                         # Test all operations (1 snapshot each)
    python test_ws.py addition 3              # Test addition with 3 snapshots
    python test_ws.py multiplication 5        # Test multiplication with 5 snapshots
"""

import asyncio
import websockets
import json
import sys

BASE_URL = "ws://localhost:8000/ws/simulate"

OPERATIONS = [
    "addition",
    "subtraction",
    "multiplication",
    "polynomial",
    "division",
]


async def test_operation(operation: str, n_snapshots: int = 1) -> bool:
    """
    Connect to the WebSocket server, request a training run for `operation`,
    receive `n_snapshots` payloads, and return True if successful.
    """
    url = f"{BASE_URL}?operation={operation}&fraction=0.4&total_steps=500&lr=0.001"
    print(f"\n  Testing '{operation}' -> {url.split('?')[1]}")

    try:
        async with websockets.connect(url) as ws:
            # Check for an error message (e.g. invalid operation)
            first_msg = await asyncio.wait_for(ws.recv(), timeout=10)
            data = json.loads(first_msg)

            if "error" in data:
                print(f"  [FAIL] Server returned error: {data['error']}")
                return False

            # Print the first snapshot
            print(f"  [OK] Step: {data['step']} | "
                  f"Train: {data['train_accuracy']:.2%} | "
                  f"Test: {data['test_accuracy']:.2%} | "
                  f"PCA points: {len(data['pca_embeddings'])} | "
                  f"Operation in payload: {data.get('operation', 'N/A')}")

            # Receive additional snapshots if requested
            for i in range(1, n_snapshots):
                msg = await asyncio.wait_for(ws.recv(), timeout=15)
                data = json.loads(msg)
                print(f"  [OK] Step: {data['step']} | "
                      f"Train: {data['train_accuracy']:.2%} | "
                      f"Test: {data['test_accuracy']:.2%}")

        return True

    except asyncio.TimeoutError:
        print(f"  [FAIL] Timed out waiting for a response from the server.")
        return False
    except ConnectionRefusedError:
        print(f"  [FAIL] Could not connect. Is the server running? (python server.py)")
        return False
    except Exception as e:
        print(f"  [FAIL] Unexpected error: {e}")
        return False


async def main():
    operation = sys.argv[1] if len(sys.argv) > 1 else None
    n_snapshots = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    print("=" * 60)
    print("Grokking API — WebSocket Test Client")
    print("=" * 60)

    if operation:
        # Test a single specific operation
        ops_to_test = [operation]
    else:
        # Test all operations
        ops_to_test = OPERATIONS
        print(f"Testing all {len(ops_to_test)} operations (1 snapshot each)...")

    results = {}
    for op in ops_to_test:
        results[op] = await test_operation(op, n_snapshots)

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results.values())
    for op, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {op}")

    print(f"\n{passed}/{len(results)} operations passed.")


if __name__ == "__main__":
    asyncio.run(main())
