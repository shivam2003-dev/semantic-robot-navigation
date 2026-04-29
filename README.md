# Semantic Navigation Agent using Vision-Language Models in Simulation

An AI agent that interprets natural language commands (e.g., *"go to the red mug on the kitchen counter"*) and navigates a 3D simulated environment to locate target objects using vision-language grounding and path planning.

![Architecture](https://img.shields.io/badge/Architecture-State_Machine-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![Simulator](https://img.shields.io/badge/Simulator-AI2--THOR-orange)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (CLI)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌─────────────┐   ┌───────────┐   ┌─────────┐ │
│  │ language  │   │  grounding  │   │  mapping   │   │  env    │ │
│  │  .py      │   │    .py      │   │   .py      │   │  .py    │ │
│  │           │   │             │   │            │   │         │ │
│  │ spaCy NLP │   │ OWL-ViT +  │   │ Occupancy  │   │AI2-THOR│ │
│  │ parser    │   │ CLIP re-rank│   │ Grid + A*  │   │wrapper  │ │
│  └─────┬─────┘   └──────┬──────┘   └─────┬──────┘   └────┬────┘ │
│        │                │                │               │      │
│        └────────────────┴────────────────┴───────────────┘      │
│                              │                                   │
│                    ┌─────────▼──────────┐                       │
│                    │     agent.py       │                       │
│                    │  State Machine     │                       │
│                    │                    │                       │
│                    │ EXPLORE → SEARCH   │                       │
│                    │ → GROUND → APPROACH│                       │
│                    │ → STOP             │                       │
│                    └────────────────────┘                       │
│                                                                 │
│  ┌──────────┐   ┌──────────┐                                   │
│  │ eval.py  │   │ viz.py   │                                   │
│  │Benchmark │   │Visualize │                                   │
│  └──────────┘   └──────────┘                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Modules

| Module | Purpose |
|--------|---------|
| `env.py` | AI2-THOR wrapper — `reset()`, `step()`, `observe()` returning RGB-D + pose + intrinsics |
| `language.py` | Instruction parser — spaCy NLP extracting target, attributes, room hint, spatial relation |
| `grounding.py` | Vision-language grounding — OWL-ViT open-vocabulary detection + CLIP re-ranking |
| `mapping.py` | Occupancy grid from depth back-projection + A* path planning with 8-connectivity |
| `agent.py` | State machine controller — EXPLORE → SEARCH_FRONTIER → GROUND → APPROACH → STOP |
| `main.py` | CLI entry point |
| `eval.py` | Benchmark evaluation — Success Rate (SR) & SPL across 24 instructions |
| `viz.py` | Matplotlib visualization — RGB + bbox overlay, top-down map + path + pose |

## Setup

### Prerequisites
- Python 3.10+
- Ubuntu 22.04 (recommended) or macOS
- GPU recommended (8 GB VRAM); CPU-only mode supported
- ~2 GB disk for model weights + ~500 MB for AI2-THOR assets

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/shivam2003-dev/semantic-robot-navigation.git
cd semantic-robot-navigation

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install PyTorch (pick your CUDA version, or CPU)
# GPU (CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CPU only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Download spaCy language model
python -m spacy download en_core_web_sm

# 6. Verify installation
python -c "from env import ThorEnv; from language import parse; print('OK')"
```

> **Note:** AI2-THOR will download ~500 MB of scene assets on first run to `~/.ai2thor/`.

## Quick Start

### Single Episode
```bash
python main.py --instruction "find the microwave" --scene FloorPlan1

# With visualization
python main.py --instruction "find the microwave" --scene FloorPlan1 --visualize

# Custom settings
python main.py \
    --instruction "go to the red mug" \
    --scene FloorPlan3 \
    --max-steps 300 \
    --detection-threshold 0.1 \
    --approach-distance 1.5
```

### Run Benchmark
```bash
# Full benchmark (24 episodes across FloorPlan1–5)
python eval.py

# Quick test (first 3 episodes)
python eval.py --subset 0,1,2

# Custom thresholds
python eval.py --threshold 0.1 --distance 1.5 --max-steps 300
```

### Run Tests
```bash
pytest -q tests/
```

### Generate Visualization GIF
```bash
# After running an episode with trace logging:
python viz.py runs/<timestamp>/trace.jsonl demo.gif
```

## Agent State Machine

```
     ┌──────────┐
     │  START    │
     └────┬─────┘
          │ reset env
          ▼
     ┌──────────┐     rotate 360° to build initial map
     │ EXPLORE   │────────────────────────────────────┐
     └────┬─────┘                                      │
          │ 8 rotations done                           │
          ▼                                            │
     ┌──────────────┐  detection found                 │
     │SEARCH_FRONTIER│─────────────────┐               │
     └────┬─────────┘                  │               │
          │ no frontier left           ▼               │
          │                     ┌───────────┐          │
          │                     │  GROUND    │◄─────────┘
          │                     │back-project│   detection during explore
          │                     │to 3D goal  │
          │                     └────┬──────┘
          │                          │ path planned
          │                          ▼
          │                     ┌───────────┐
          │                     │ APPROACH   │
          │                     │follow path │
          │                     └────┬──────┘
          │                          │ within 1m + visible
          │                          ▼
          │                     ┌───────────┐
          └────────────────────►│   STOP     │
                                │  Done()   │
                                └───────────┘
```

### Key Behaviors
- **Grounding frequency**: Every 5 steps during frontier search; every step during approach
- **Fallback**: If no detection for 50+ steps during approach, revert to frontier exploration
- **Step budget**: 250 steps (configurable)
- **Success**: Agent calls `Done` within 1.0 m of target AND target visible

## Output

### Trace Logs
Each episode saves a per-step trace to `runs/<timestamp>/trace.jsonl`:
```json
{
  "step": 42,
  "state": "APPROACH",
  "action": "MoveAhead",
  "pose": {"x": 1.25, "z": -0.5, "yaw": 90.0},
  "query": "red mug",
  "detections_top3": [
    {"bbox": [120, 80, 200, 160], "score": 0.72, "label": "red mug"}
  ],
  "target_world": [2.1, -0.3],
  "path_length": 3.45
}
```

### Evaluation Metrics
- **Success Rate (SR)**: Fraction of episodes where agent stops within 1.0 m of target
- **SPL**: Success weighted by Path Length — penalizes inefficient paths
  - SPL = (1/N) Σ S_i · (l_i / max(p_i, l_i))
  - where S_i = success, l_i = oracle shortest path, p_i = actual path

## Troubleshooting

| Issue | Solution |
|-------|----------|
| AI2-THOR won't start | Ensure X11 or use `--headless` mode. On Linux, install `xvfb`: `sudo apt install xvfb && xvfb-run python main.py ...` |
| CUDA out of memory | Use CPU mode: models auto-detect device. Or reduce image resolution in `ThorEnv` constructor |
| OWL-ViT slow on CPU | Expected (~2-3s per frame). Reduce `ground_every_n` in agent config |
| spaCy model missing | Run `python -m spacy download en_core_web_sm` |
| AI2-THOR download hangs | Delete `~/.ai2thor` and retry. Check internet connection |
| "No module named clip" | Install CLIP: `pip install git+https://github.com/openai/CLIP.git` |

## Project Structure

```
semantic-robot-navigation/
├── agent.py              # State machine controller
├── env.py                # AI2-THOR environment wrapper
├── eval.py               # Benchmark evaluation (SR, SPL)
├── grounding.py          # OWL-ViT + CLIP grounding
├── language.py           # spaCy instruction parser
├── main.py               # CLI entry point
├── mapping.py            # Occupancy grid + A* planner
├── viz.py                # Matplotlib visualization
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── .gitignore
├── tests/
│   ├── __init__.py
│   ├── test_env.py       # Environment wrapper tests (mocked)
│   ├── test_grounding.py # IoU, NMS, detection tests
│   ├── test_language.py  # Instruction parser tests
│   └── test_mapping.py   # Occupancy grid + A* tests
└── runs/                 # Episode logs (auto-created)
    └── <timestamp>/
        ├── trace.jsonl   # Per-step log
        └── summary.json  # Episode summary
```

## Known Limitations

1. **Discrete action space**: 45° rotation steps and 0.25 m movement quanta limit fine-grained positioning. Objects requiring precise approach angles may be missed.
2. **Open-vocabulary detection noise**: OWL-ViT can produce false positives for visually similar objects (e.g., "mug" vs "cup"). CLIP re-ranking mitigates but doesn't eliminate this.
3. **No memory of previously seen objects**: The agent doesn't maintain a semantic map of where objects were seen. If it passes a target while exploring, it must re-detect it.
4. **Single-floor only**: No stair navigation or multi-floor planning.
5. **Static scenes**: AI2-THOR objects are not moved by the agent; no pick-and-place manipulation.

## Future Improvements

1. **Learned navigation policy (DD-PPO)**: Replace the rule-based state machine with a reinforcement learning policy trained with Decentralized Distributed PPO for more robust navigation in unseen environments.
2. **Semantic memory map**: Maintain a persistent spatial map of detected objects with their semantic labels, allowing the agent to navigate to previously seen objects without re-detection.
3. **LLM-based replanning**: Use a local LLM (e.g., Llama-3) to decompose complex instructions ("find the mug, then bring it to the table") into sub-goals and dynamically replan when the initial strategy fails.

## Tech Stack

- **Simulator**: [AI2-THOR](https://ai2thor.allenai.org/) (iTHOR)
- **Vision-Language Models**: [OWL-ViT](https://huggingface.co/google/owlvit-base-patch32) + [CLIP](https://github.com/openai/CLIP)
- **NLP**: [spaCy](https://spacy.io/)
- **Path Planning**: A* on depth-derived occupancy grid
- **Framework**: PyTorch

## License

MIT License

## Citation

If you use this work, please cite:
```bibtex
@misc{semantic-nav-2026,
  title={Semantic Navigation Agent using Vision-Language Models in Simulation},
  author={Shivam Kumar},
  year={2026},
  url={https://github.com/shivam2003-dev/semantic-robot-navigation}
}
```