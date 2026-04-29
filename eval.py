"""
eval.py — Benchmark Evaluation

Runs the navigation agent on a set of instructions across multiple scenes
and computes Success Rate (SR) and SPL (Success weighted by Path Length).
"""

import json
import math
import os
import sys
import time
from typing import List, Dict, Tuple

# Benchmark: 24 instructions across FloorPlan1–FloorPlan5 (kitchens)
# Mix of easy (common visible objects), medium (need rotation), hard (need exploration)
BENCHMARK = [
    # FloorPlan1 — easy
    {"scene": "FloorPlan1", "instruction": "find the microwave",       "target": "Microwave"},
    {"scene": "FloorPlan1", "instruction": "go to the fridge",         "target": "Fridge"},
    {"scene": "FloorPlan1", "instruction": "find the toaster",         "target": "Toaster"},
    {"scene": "FloorPlan1", "instruction": "find the sink",            "target": "Sink"},
    {"scene": "FloorPlan1", "instruction": "go to the stove burner",   "target": "StoveBurner"},
    # FloorPlan2 — mixed
    {"scene": "FloorPlan2", "instruction": "find the coffee machine",  "target": "CoffeeMachine"},
    {"scene": "FloorPlan2", "instruction": "go to the garbage can",    "target": "GarbageCan"},
    {"scene": "FloorPlan2", "instruction": "find the apple",           "target": "Apple"},
    {"scene": "FloorPlan2", "instruction": "find the bowl",            "target": "Bowl"},
    {"scene": "FloorPlan2", "instruction": "go to the knife",          "target": "Knife"},
    # FloorPlan3
    {"scene": "FloorPlan3", "instruction": "find the lettuce",         "target": "Lettuce"},
    {"scene": "FloorPlan3", "instruction": "go to the pot",            "target": "Pot"},
    {"scene": "FloorPlan3", "instruction": "find the plate",           "target": "Plate"},
    {"scene": "FloorPlan3", "instruction": "find the mug",             "target": "Mug"},
    {"scene": "FloorPlan3", "instruction": "go to the cup",            "target": "Cup"},
    # FloorPlan4
    {"scene": "FloorPlan4", "instruction": "find the bread",           "target": "Bread"},
    {"scene": "FloorPlan4", "instruction": "go to the pan",            "target": "Pan"},
    {"scene": "FloorPlan4", "instruction": "find the spatula",         "target": "Spatula"},
    {"scene": "FloorPlan4", "instruction": "find the egg",             "target": "Egg"},
    # FloorPlan5
    {"scene": "FloorPlan5", "instruction": "find the tomato",          "target": "Tomato"},
    {"scene": "FloorPlan5", "instruction": "go to the dish sponge",    "target": "DishSponge"},
    {"scene": "FloorPlan5", "instruction": "find the salt shaker",     "target": "SaltShaker"},
    {"scene": "FloorPlan5", "instruction": "find the pepper shaker",   "target": "PepperShaker"},
    {"scene": "FloorPlan5", "instruction": "go to the wine bottle",    "target": "WineBottle"},
]


def compute_oracle_distance(env, target_type: str) -> float:
    """
    Compute the shortest straight-line distance from the agent's starting
    position to the nearest instance of target_type using scene metadata.
    """
    objects = env.get_object_positions()
    agent_pos = env.controller.last_event.metadata["agent"]["position"]

    min_dist = float("inf")
    target_lower = target_type.lower()
    for obj in objects:
        if target_lower in obj["objectType"].lower():
            d = math.sqrt(
                (agent_pos["x"] - obj["position"]["x"]) ** 2
                + (agent_pos["z"] - obj["position"]["z"]) ** 2
            )
            min_dist = min(min_dist, d)

    return min_dist


def run_benchmark(
    max_steps: int = 250,
    detection_threshold: float = 0.15,
    approach_distance: float = 1.0,
    subset: List[int] = None,
):
    """
    Run the full benchmark and report metrics.

    Args:
        max_steps: step budget per episode.
        detection_threshold: grounding confidence threshold.
        approach_distance: distance (m) for success.
        subset: optional list of benchmark indices to run (for debugging).
    """
    from env import ThorEnv
    from grounding import VLGrounder
    from mapping import OccupancyMap
    from agent import NavigationAgent

    print("=" * 70)
    print("SEMANTIC NAVIGATION BENCHMARK")
    print("=" * 70)

    # Initialize shared components
    print("\n[eval] Loading models...")
    env = ThorEnv(headless=True)
    grounder = VLGrounder(detection_threshold=detection_threshold)

    entries = BENCHMARK if subset is None else [BENCHMARK[i] for i in subset]
    results = []

    eval_dir = os.path.join("runs", f"eval_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(eval_dir, exist_ok=True)

    for idx, entry in enumerate(entries):
        scene = entry["scene"]
        instruction = entry["instruction"]
        target = entry["target"]

        print(f"\n[{idx+1}/{len(entries)}] Scene={scene}, "
              f"Instruction='{instruction}', Target={target}")

        # Fresh mapper per episode
        mapper = OccupancyMap()
        log_dir = os.path.join(eval_dir, f"ep_{idx:03d}")

        agent = NavigationAgent(
            env=env,
            grounder=grounder,
            mapper=mapper,
            detection_threshold=detection_threshold,
            approach_distance=approach_distance,
            max_steps=max_steps,
            log_dir=log_dir,
        )

        # Get oracle distance before running
        env.reset(scene)
        oracle_dist = compute_oracle_distance(env, target)

        # Run episode
        result = agent.run(instruction, scene=scene, max_steps=max_steps)

        # Compute SPL for this episode
        if result.success:
            spl = oracle_dist / max(result.path_length, oracle_dist) if oracle_dist > 0 else 1.0
        else:
            spl = 0.0

        entry_result = {
            "scene": scene,
            "instruction": instruction,
            "target": target,
            "success": result.success,
            "steps": result.steps,
            "path_length": round(result.path_length, 3),
            "oracle_distance": round(oracle_dist, 3),
            "spl": round(spl, 4),
            "final_distance": round(result.final_distance, 3),
        }
        results.append(entry_result)

        status = "SUCCESS" if result.success else "FAIL"
        print(f"  → {status} | steps={result.steps} | "
              f"path={result.path_length:.2f}m | "
              f"dist={result.final_distance:.2f}m | "
              f"spl={spl:.3f}")

    # Compute aggregate metrics
    n = len(results)
    successes = sum(1 for r in results if r["success"])
    sr = successes / n if n > 0 else 0.0
    avg_spl = sum(r["spl"] for r in results) / n if n > 0 else 0.0
    avg_steps = sum(r["steps"] for r in results) / n if n > 0 else 0.0
    avg_path = sum(r["path_length"] for r in results) / n if n > 0 else 0.0

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Episodes       : {n}")
    print(f"  Success Rate   : {sr:.1%} ({successes}/{n})")
    print(f"  SPL            : {avg_spl:.4f}")
    print(f"  Avg Steps      : {avg_steps:.1f}")
    print(f"  Avg Path Length : {avg_path:.2f} m")
    print("=" * 70)

    # Per-scene breakdown
    scenes = sorted(set(r["scene"] for r in results))
    print("\nPer-Scene Breakdown:")
    for scene in scenes:
        scene_results = [r for r in results if r["scene"] == scene]
        s_n = len(scene_results)
        s_sr = sum(1 for r in scene_results if r["success"]) / s_n
        s_spl = sum(r["spl"] for r in scene_results) / s_n
        print(f"  {scene}: SR={s_sr:.1%}, SPL={s_spl:.4f} (n={s_n})")

    # Save results
    results_path = os.path.join(eval_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "metrics": {
                "success_rate": round(sr, 4),
                "spl": round(avg_spl, 4),
                "avg_steps": round(avg_steps, 1),
                "avg_path_length": round(avg_path, 3),
            },
            "episodes": results,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

    env.close()
    return sr, avg_spl


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run navigation benchmark")
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--subset", type=str, default=None,
                       help="Comma-separated episode indices (e.g., 0,1,2)")
    args = parser.parse_args()

    subset = None
    if args.subset:
        subset = [int(x) for x in args.subset.split(",")]

    sr, spl = run_benchmark(
        max_steps=args.max_steps,
        detection_threshold=args.threshold,
        approach_distance=args.distance,
        subset=subset,
    )

    sys.exit(0 if sr >= 0.5 else 1)
