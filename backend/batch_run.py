"""
Batch runner for Phase Diagram generation.

Trains the model across a grid of (dataset_fraction, weight_decay) combinations
and records when (if) grokking occurs. Outputs phase_diagram.json for the frontend.

Usage:
    python batch_run.py                      # Run with defaults
    python batch_run.py --steps 30000        # Custom max steps
    python batch_run.py --fractions 10 --decays 10  # 10x10 = 100 runs
"""

import json
import argparse
import time
import numpy as np
from pathlib import Path

from train import train_single_run


def run_phase_diagram(
    n_fractions: int = 15,
    n_decays: int = 15,
    total_steps: int = 40000,
    log_every: int = 500,
    output_path: str = "phase_diagram.json",
):
    """
    Execute a grid search over dataset fractions and weight decays.

    Args:
        n_fractions: Number of fraction values to test
        n_decays:    Number of weight decay values to test
        total_steps: Max steps per training run
        log_every:   Logging frequency
        output_path: Where to save results
    """
    # Define the hyperparameter grid
    fractions = np.linspace(0.1, 0.9, n_fractions).tolist()
    weight_decays = np.logspace(-2, 1, n_decays).tolist()  # 0.01 to 10.0 log-scale

    total_runs = n_fractions * n_decays
    results = []

    print("=" * 70)
    print(f"PHASE DIAGRAM BATCH RUN")
    print(f"Grid: {n_fractions} fractions × {n_decays} weight_decays = {total_runs} runs")
    print(f"Max steps per run: {total_steps}")
    print("=" * 70)

    start_time = time.time()

    for i, frac in enumerate(fractions):
        for j, wd in enumerate(weight_decays):
            run_idx = i * n_decays + j + 1
            print(f"\n[{run_idx}/{total_runs}] fraction={frac:.3f}, weight_decay={wd:.4f}")

            try:
                result = train_single_run(
                    fraction=frac,
                    weight_decay=wd,
                    total_steps=total_steps,
                    log_every=log_every,
                )

                entry = {
                    "fraction": round(frac, 4),
                    "weight_decay": round(wd, 6),
                    "grok_step": result.get("grok_step", -1),
                    "final_train_accuracy": result.get("train_accuracy", 0),
                    "final_test_accuracy": result.get("test_accuracy", 0),
                    "grokked": result.get("grokked", False),
                }
                results.append(entry)

                status = f"GROKKED at step {entry['grok_step']}" if entry["grokked"] else "NO GROK"
                print(f"  → {status} | Train: {entry['final_train_accuracy']:.2%} "
                      f"| Test: {entry['final_test_accuracy']:.2%}")

            except Exception as e:
                print(f"  → ERROR: {e}")
                results.append({
                    "fraction": round(frac, 4),
                    "weight_decay": round(wd, 6),
                    "grok_step": -1,
                    "final_train_accuracy": 0,
                    "final_test_accuracy": 0,
                    "grokked": False,
                    "error": str(e),
                })

            # Save intermediate results after each run
            _save_results(results, fractions, weight_decays, output_path)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"BATCH RUN COMPLETE — {total_runs} runs in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 70}")

    return results


def _save_results(results, fractions, weight_decays, output_path):
    """Save current results to JSON."""
    output = {
        "metadata": {
            "prime": 97,
            "fractions": [round(f, 4) for f in fractions],
            "weight_decays": [round(w, 6) for w in weight_decays],
            "total_runs": len(results),
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate grokking phase diagram")
    parser.add_argument("--fractions", type=int, default=15, help="Number of fraction grid points")
    parser.add_argument("--decays", type=int, default=15, help="Number of weight_decay grid points")
    parser.add_argument("--steps", type=int, default=40000, help="Max training steps per run")
    parser.add_argument("--log-every", type=int, default=500, help="Logging frequency")
    parser.add_argument("--output", type=str, default="phase_diagram.json", help="Output file path")

    args = parser.parse_args()

    run_phase_diagram(
        n_fractions=args.fractions,
        n_decays=args.decays,
        total_steps=args.steps,
        log_every=args.log_every,
        output_path=args.output,
    )
