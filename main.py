"""
main.py — CLI Entry Point for Semantic Navigation Agent

Usage:
    python main.py --instruction "find the microwave" --scene FloorPlan1 --visualize
"""

import argparse
import os
import sys
import json


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Navigation Agent — navigate to objects via language"
    )
    parser.add_argument(
        "--instruction", "-i",
        type=str,
        required=True,
        help='Navigation instruction, e.g. "find the microwave"',
    )
    parser.add_argument(
        "--scene", "-s",
        type=str,
        default="FloorPlan1",
        help="AI2-THOR scene name (default: FloorPlan1)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=250,
        help="Maximum steps per episode (default: 250)",
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Show live visualization window",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for trace logs (default: runs/<timestamp>)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run AI2-THOR in headless mode (default: True)",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.15,
        help="Minimum detection score to trigger grounding (default: 0.15)",
    )
    parser.add_argument(
        "--approach-distance",
        type=float,
        default=1.0,
        help="Distance (m) to target for success (default: 1.0)",
    )

    args = parser.parse_args()

    # Lazy imports so --help is fast
    from env import ThorEnv
    from grounding import VLGrounder
    from mapping import OccupancyMap
    from agent import NavigationAgent

    print("=" * 60)
    print("Semantic Navigation Agent")
    print("=" * 60)
    print(f"  Instruction : {args.instruction}")
    print(f"  Scene       : {args.scene}")
    print(f"  Max steps   : {args.max_steps}")
    print(f"  Visualize   : {args.visualize}")
    print("=" * 60)

    # Initialize components
    print("\n[init] Setting up environment...")
    env = ThorEnv(scene=args.scene, headless=args.headless)

    print("[init] Loading vision-language models...")
    grounder = VLGrounder(detection_threshold=args.detection_threshold)

    mapper = OccupancyMap()

    agent = NavigationAgent(
        env=env,
        grounder=grounder,
        mapper=mapper,
        detection_threshold=args.detection_threshold,
        approach_distance=args.approach_distance,
        max_steps=args.max_steps,
        log_dir=args.log_dir,
    )

    # Optionally set up visualization
    viz = None
    if args.visualize:
        try:
            from viz import NavigationVisualizer
            viz = NavigationVisualizer(mapper)
            # Hook into agent to update viz each step
            _orig_run = agent.run

            def run_with_viz(instruction, scene=None, max_steps=None):
                # We'll patch the step logic — simpler to just call run
                # and visualize from the trace after
                result = _orig_run(instruction, scene, max_steps)
                return result

            agent.run = run_with_viz
        except ImportError:
            print("[warn] viz.py not available, running without visualization")

    # Run the agent
    print("\n[run] Starting navigation episode...\n")
    result = agent.run(args.instruction, scene=args.scene, max_steps=args.max_steps)

    # Print results
    print("\n" + "=" * 60)
    print("EPISODE RESULT")
    print("=" * 60)
    print(f"  Success        : {result.success}")
    print(f"  Steps          : {result.steps}")
    print(f"  Path length    : {result.path_length:.2f} m")
    print(f"  Final distance : {result.final_distance:.2f} m")
    print(f"  Log directory  : {agent.log_dir}")
    print("=" * 60)

    # Save summary
    summary_path = os.path.join(agent.log_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "instruction": args.instruction,
            "scene": args.scene,
            "success": result.success,
            "steps": result.steps,
            "path_length": result.path_length,
            "final_distance": result.final_distance,
        }, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Post-episode visualization
    if args.visualize and viz is not None:
        print("[viz] Generating visualization from trajectory...")
        viz.replay_trajectory(result.trajectory, agent.mapper)

    env.close()
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
