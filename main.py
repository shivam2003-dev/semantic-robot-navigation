"""
main.py — CLI Entry Point for Semantic Navigation Agent

Usage:
    python main.py --instruction "find the microwave" --scene FloorPlan1 --visualize
"""

import argparse
import os
import sys
import json
import numpy as np


def _show_annotated_frame(env, agent, result):
    """Display the final RGB frame with detection bounding boxes annotated."""
    import matplotlib
    matplotlib.use("TkAgg")  # interactive backend so the window stays open
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Grab the current frame from the live controller
    frame = env.controller.last_event.frame  # (H, W, 3) uint8

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.imshow(frame)

    # Draw detections
    for det in agent.last_detections[:5]:
        x1, y1, x2, y2 = det.bbox
        score = float(det.score)
        label = det.label

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, edgecolor="lime", facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 8,
            f"{label}  {score:.2f}",
            color="white", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.85),
        )

    status = "FOUND" if result.success else "NOT REACHED"
    ax.set_title(
        f"[{status}] \"{result.instruction}\" — {result.steps} steps, "
        f"{result.final_distance:.2f}m away",
        fontsize=13, fontweight="bold",
    )
    ax.axis("off")
    plt.tight_layout()

    # Save annotated frame
    out_path = os.path.join(agent.log_dir, "annotated_final.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[viz] Annotated frame saved → {out_path}")

    # Show interactively (non-blocking so terminal prompt still works)
    plt.show(block=False)
    plt.pause(0.5)


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

    # Show annotated final frame (found or not)
    if agent.last_detections or result.success:
        _show_annotated_frame(env, agent, result)
    else:
        # Run grounding one last time on the final frame for annotation
        final_frame = env.controller.last_event.frame
        final_dets = grounder.score_frame(final_frame, agent.query)
        agent.last_detections = final_dets
        _show_annotated_frame(env, agent, result)

    # Keep scene open for manual inspection — wait for user
    print("\n[info] AI2-THOR scene is still open.")
    input("[info] Press Enter to close the scene and exit...")
    env.close()
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
