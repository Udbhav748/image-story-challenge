# Image → Story V2

**A professional, modular, reproducible multimodal ML system for grounded creative storytelling.**

## Architecture Overview

```
IMAGE(S)
   ↓
VISUAL PERCEPTION (Florence-2 + GroundingDINO + OCR)
   ↓
GROUNDED EVIDENCE (structured, provenance-tracked)
   ↓
WORLD / ENTITY STATE (persistent across frames)
   ↓
FAISS RETRIEVAL / MEMORY (semantic + narrative)
   ↓
EVIDENCE RANKING (reliability-aware)
   ↓
RELIABILITY-AWARE CONTEXT (Hard Facts / Soft Inferences / Creative Space)
   ↓
CREATIVE STORY ENGINE (Character, Conflict, Humor, Surprise, Open Loops)
   ↓
STORY BEAT PLANNER (7-beat structure)
   ↓
QWEN WRITER (local LLM)
   ↓
CLAIM EXTRACTION
   ↓
CLAIM / VISUAL VERIFICATION
   ↓
CONTINUITY + CONTRADICTION + QUALITY EVALUATION
   ↓
FINAL GROUNDED CREATIVE STORY
```

## Key Principles

> **The system is free to invent the story, but never free to invent the evidence.**

- **Hard Facts**: Directly supported by visual evidence (locked)
- **Soft Inferences**: Reasonable interpretations (qualified)
- **Creative Space**: Narrative invention allowed (personalities, motivations, dialogue, humor, twists)

## Pipeline Modes

| Mode | Vision | Memory | Creative | Verification | Use Case |
|------|--------|--------|----------|--------------|----------|
| **Fast** | Florence-2 only | ❌ | ❌ | ❌ | CPU demos, quick tests |
| **Standard** | Florence-2 + GroundingDINO | FAISS | ✅ | ✅ | Primary development |
| **Full** | All + OCR | FAISS + narrative memory | ✅ | ✅ | Research, showcase |

## Installation

```bash
# Clone and install
pip install -e .

# Or with dependencies only
pip install -r requirements_v2.txt

# Download models (requires internet once)
python scripts/download_models.py

# Install spaCy model for claim extraction
python -m spacy download en_core_web_sm
```

## Usage

```bash
# Single image, standard mode
python main_v2.py images/thumb-chihiro001.png

# Multi-image sequence
python main_v2.py images/ --multi

# Fast mode (no GroundingDINO, no FAISS)
python main_v2.py images/ --mode fast

# Full mode (all features)
python main_v2.py images/ --mode full

# Custom config
python main_v2.py images/ --config configs/custom.yaml

# Skip evaluation
python main_v2.py images/ --no-eval

# Output directory
python main_v2.py images/ --output-dir my_results
```

## Project Structure

```
image_story_v2/
├── src/
│   └── image_story/
│       ├── domain/          # Schemas, enums, exceptions
│       ├── vision/          # Florence-2, GroundingDINO, OCR, Verifier
│       ├── memory/          # Embeddings, FAISS, Retrieval
│       ├── context/         # World State, Ranker, Context Builder
│       ├── narrative/       # Character, Conflict, Humor, Surprise, Planner
│       ├── generation/      # Qwen adapter, Generation pipeline
│       ├── evaluation/      # Grounding, Continuity, Narrative, Claims
│       ├── pipeline/        # Orchestrator
│       └── config/          # Settings
├── configs/                 # YAML configs (fast, standard, full, baseline)
├── evals/
│   ├── golden/              # Regression test cases
│   └── fixtures/
├── tests/
│   ├── unit/                # Unit tests (no models)
│   ├── integration/         # Integration tests
│   └── regression/          # Golden set regression
├── scripts/                 # Utility scripts
├── artifacts/               # Generated outputs
├── notebooks/               # Analysis notebooks
└── main_v2.py              # CLI entry point
```

## Running Tests

```bash
# Unit tests (no model inference)
pytest tests/unit -v

# With model-dependent tests
pytest tests/unit --models

# Integration tests (requires downloaded models)
pytest tests/integration -v
```

## Evaluation Metrics

The system provides comprehensive evaluation:

- **Grounding**: CLIP image-story similarity, NLI contradiction, attribute conflicts
- **Claims**: Claim extraction, visual verification, claim grounding score
- **Continuity**: Entity consistency, transition markers, adjacent similarity
- **Narrative Quality**: Character depth, plot structure, emotional arc, humor, callbacks
- **Runtime**: Separate generation vs evaluation timing

## Golden Set Regression

Fixed test cases in `evals/golden/` ensure reproducibility:

```json
{
  "case_id": "golden_001",
  "image_path": "images/thumb-chihiro001.png",
  "expected_entities": ["girl", "car", "flowers"],
  "known_visual_facts": ["girl in car", "holding flowers"],
  "known_contradictions": ["boy", "dog"],
  "min_grounding_score": 0.6
}
```

## Ablation Framework

Track experiment manifests automatically:

```json
{
  "git_commit": "...",
  "vision_model": "florence2",
  "language_model": "qwen2.5-0.5b",
  "embedding_model": "minilm-l6",
  "dataset": "chihiro",
  "seed": 0,
  "config": {...}
}
```

Artifacts saved per run for full reproducibility.

## Citation

If you use this system, please cite the original Image → Story challenge and this V2 implementation.

## License

MIT License