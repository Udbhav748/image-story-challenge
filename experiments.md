# Experiments: Image → Story Challenge

All numbers come from one run of `python main.py "D:\Downloads\images" --both --output results_new.csv`
(see `results_new.csv`, `results_new_vision.json`, `combined_story.json`). Everything ran offline (`HF_HUB_OFFLINE=1`) on a
CPU-only laptop (Python 3.13, torch 2.13, transformers 5.15). 8 images from the *Spirited Away* frame set.

## Problem

The baseline pipeline uses a single BLIP caption (one sentence) as the only visual information passed to the story model.
This creates a severe bottleneck: most visual detail (objects, characters, actions, spatial layout, text) is lost before
story generation. The story model hallucinates details to fill the gaps, producing generic or contradictory narratives.

## Baseline

**Pipeline A (baseline):** `Image → BLIP (Salesforce/blip-image-captioning-base) → 1-sentence caption → Qwen2.5-0.5B-Instruct → story`

- BLIP generates one short caption (e.g., "a girl sitting in the back of a car with a bunch of flowers")
- Same Qwen2.5-0.5B-Instruct model used for both pipelines (greedy, seed=0, repetition_penalty=1.05)
- Prompt asks for 80–120 word story; no length enforcement in baseline run
- Instructor's original baseline code was not available; this is a stand-in matching the spec

**Pipeline B (improved):** `Image → Florence-2 (detailed caption + OD + dense regions) → structured JSON → context builder → Qwen2.5-0.5B-Instruct → story`

- Module 7 change: replace BLIP with Florence-2-base for richer visual extraction (plus supporting pieces added in the same pass: dense region captions, a structured JSON and a deterministic context builder; see "Change" below)
- Structured output: scene, description, objects, characters, actions, relationships, OCR, spatial, style/mood, region descriptions
- Deterministic context builder converts structured JSON → concise LLM prompt
- Same Qwen model, same prompt template, same decoding settings

## Observed Failures (Baseline)

From reading the 8 baseline stories in `results_new.csv`:

1. **Wrong length — 0/8 stories in 80–120 range** (30–76 words). The baseline's only check is word count.
2. **Invented visual details with no basis in the image.** `chihiro003.jpg`: caption "a man walking down the street" → story adds "jeans and a t-shirt that shows off his muscular build". `thumb-chihiro008.png`: caption "a man and a dog" → story calls it "a golden retriever".
3. **Generic stories from underspecified captions.** `thumb-chihiro007.png`: caption "a restaurant with a table and chairs" → story invents coffee, bread, waiter, conversation — none grounded in the image.
4. **No continuity across frames.** Each image treated independently; multi-image story (`combined_story.json`) invents names (Chihiro, Takahashi) and events not in the visual evidence.

## Hypothesis

Replacing the single BLIP sentence with a structured Florence-2 description (detailed caption + object detection + dense region captions)
will:
- Provide the story model with concrete visual facts (objects, characters, actions, spatial layout)
- Reduce hallucination by giving the model something to ground to
- Improve CLIP image-story similarity and reduce NLI contradiction
- Enable continuity by making recurring entities explicit in the structured context

## Change (Module 7)

**Scope note.** The core change is to the seeing stage: Florence-2-base replaces BLIP. In the same pass, three supporting
pieces were added: dense region captions, a structured JSON, and a deterministic context builder. They were **not ablated**
(never run one at a time), so the measured gain below cannot be attributed to any single one of them.

**Replace BLIP captioner with Florence-2-base using three tasks:**
1. `<MORE_DETAILED_CAPTION>` — paragraph-level scene description
2. `<OD>` — object detection labels
3. `<DENSE_REGION_CAPTION>` — region-level descriptions with bounding boxes

**Structured output schema (populated only from model outputs, no hallucination):**
```json
{
  "image_id": "...",
  "scene": "...",           // first sentence of detailed caption
  "description": "...",     // full detailed caption
  "objects": [...],         // OD labels + dense region object descriptions
  "characters": [...],      // character types from dense regions + detailed caption (person, man, girl, pig, bird...)
  "actions": [...],         // action verbs from dense regions + detailed caption (walking, sitting, holding...)
  "relationships": [...],   // spatial phrases (next to, beside, on, in...)
  "ocr_text": "",           // disabled (junk on anime frames)
  "spatial_relations": [...], // same as relationships
  "region_descriptions": [...], // full dense region captions (most informative)
  "style_or_mood": "..."    // keyword from detailed caption (whimsical, festive, peaceful...)
}
```

**CPU budget:** ~15–20 s/image (vs ~2 s for BLIP). Dropped `<OCR>` (junk) and `<CAPTION>` (redundant). Greedy decoding, KV cache on. Not ablated.

**Context builder (`context_builder.py`):** deterministic, no ML. For single image: concatenates scene, characters, objects, actions, top 3 region descriptions, style. For multi-image: builds sequence context with continuity notes (recurring characters, locations, objects) and explicit instruction to maintain identity and connect events.

## Evaluation Method (Module 10)

Every `(image, caption/context, story)` tuple scored by `metric.py`:

| Metric | What it measures | How computed | Pass threshold |
|--------|------------------|--------------|----------------|
| **Grounding score** | Composite: visual alignment + consistency + attribute accuracy | `0.4*clip_n + 0.4*(1-nli_contra_mean) + 0.2*(1-attr_conflict)` | `≥0.60` AND no attr conflict AND length valid |
| `clip_n` | CLIP ViT-B/32 image–story cosine, normalized | `clip((mean_cos - 0.15)/0.15, 0, 1)` | — |
| `nli_contra_mean` | NLI contradiction prob (caption vs story sentences) | `cross-encoder/nli-MiniLM2-L6-H768` mean P(contradiction) | lower better |
| `attribute_conflict` | Color/material word in story not in caption | Set overlap per group (color, material) | 0 |
| **Length valid** | Story in 80–120 words | `80 ≤ words ≤ 120` | hard check |
| **Repetition rate** | Fraction of repeated trigrams | `repeated_trigrams / total_trigrams` | lower better (report only) |
| **Entity consistency** | Fraction of recurring visual entities mentioned in story | Entities appearing in 2+ frames ∩ story words / recurring entities | higher better (report only) |
| **Transition markers** | Count of temporal/transition words in story | Lexicon: then, next, after, later, suddenly, meanwhile... | report only |
| **Adjacent similarity** | Entity Jaccard between adjacent frame contexts | Mean over adjacent pairs | report only |
| **Runtime** | Monotonic timers | `seeing_s`, `story_s`, `total_s` | report only |

`grounding_pass` = all three: score ≥ 0.60, no attribute conflict, length valid.

## Baseline Results (8 images)

| Metric | Value |
|--------|-------|
| Mean grounding score | 0.783 |
| Mean CLIP image–story | 0.233 |
| Mean NLI contradiction | 0.098 |
| Mean repetition rate | 0.0034 |
| Length valid (80–120 words) | 0/8 |
| Grounding pass | 0/8 |
| Mean seeing time | 4.3 s |
| Mean story time | 9.1 s |
| Mean total time | 13.4 s |

Per-image grounding (baseline):  
chihiro003.jpg: 0.740 | thumb-chihiro001.png: 0.856 | thumb-chihiro002.png: 0.644 | thumb-chihiro004.png: 0.895 | thumb-chihiro005.png: 0.722 | thumb-chihiro006.png: 0.878 | thumb-chihiro007.png: 0.787 | thumb-chihiro008.png: 0.746

## Improved Results (8 images)

| Metric | Value |
|--------|-------|
| Mean grounding score | 0.866 |
| Mean CLIP image–story | 0.258 |
| Mean NLI contradiction | 0.055 |
| Mean repetition rate | 0.0073 |
| Length valid (80–120 words) | 2/8 |
| Grounding pass | 2/8 |
| Mean seeing time | 18.1 s |
| Mean story time | 8.8 s |
| Mean total time | 27.0 s |

Per-image grounding (improved):  
chihiro003.jpg: 0.805 | thumb-chihiro001.png: 0.817 | thumb-chihiro002.png: 0.883 | thumb-chihiro004.png: 0.834 | thumb-chihiro005.png: 0.843 | thumb-chihiro006.png: 0.864 | thumb-chihiro007.png: 0.907 | thumb-chihiro008.png: 0.975

## Before vs After

| METRIC | BASELINE | IMPROVED | DELTA |
|--------|----------|----------|-------|
| Mean grounding score | 0.783 | 0.866 | **+0.083** |
| Mean CLIP similarity | 0.233 | 0.258 | **+0.025** |
| Mean NLI contradiction | 0.098 | 0.055 | **−0.043** |
| Mean repetition rate | 0.0034 | 0.0073 | +0.0039 |
| Length valid (80–120) | 0/8 | 2/8 | +2 |
| Grounding pass | 0/8 | 2/8 | +2 |
| Mean seeing time (s) | 4.3 | 18.1 | +13.8 |
| Mean story time (s) | 9.1 | 8.8 | −0.3 |
| Mean total time (s) | 13.4 | 27.0 | +13.6 |

**Grounding score improved on 5 of 8 images.** Largest gains: thumb-chihiro002.png (+0.239), thumb-chihiro008.png (+0.230), thumb-chihiro005.png (+0.121), thumb-chihiro007.png (+0.120), chihiro003.jpg (+0.065).  
**Regressed on 3 images:** thumb-chihiro001.png (−0.039), thumb-chihiro004.png (−0.061), thumb-chihiro006.png (−0.014).  
(Counts checked against `results_new.csv`.)

**The pass-rate gain is mostly a length effect.** Grounding pass requires 80–120 words, so "0/8 → 2/8" says little about
grounding; see the length-equalised check below.

## Length-equalised check (earlier run, simpler pipeline)

Run before the structured-context version existed (Florence-2 detailed caption + object labels only, no dense regions or context
builder), with one length/cut-off fix applied to the shared Qwen stage of **both** pipelines (`results_fixlen.csv`):

| | Baseline | Improved |
|---|---|---|
| Mean grounding score | 0.756 | 0.837 |
| Mean NLI contradiction | 0.207 | 0.103 |
| Length valid | 6/8 | 5/8 |
| Grounding pass | 5/8 | 5/8 |

With length equalised the **pass counts tie (5/8 vs 5/8)**, while the grounding-score gain remains (+0.08). This check has **not**
been repeated for the current structured-context pipeline, so treat the "+2 passes" above as a length effect until it is.

## Runtime

| Stage | Baseline | Improved |
|-------|----------|----------|
| Vision/seeing | 4.3 s | 18.1 s |
| Story generation | 9.1 s | 8.8 s |
| **Total** | **13.4 s** | **27.0 s** |

Seeing is ~4.2× slower with Florence-2 (3 tasks vs 1). Story time similar. Total ~2× slower. Measured after model loads (load times excluded).

## Per-Image Analysis

| Image | Baseline story | Improved story | Why improved / regressed |
|-------|----------------|----------------|--------------------------|
| chihiro003.jpg | 71w, generic man, invented jeans/muscles | 61w, mentions children, Christmas lights, European city | **+0.065**: dense regions gave "man + two children walking", "colorful buildings with lanterns" |
| thumb-chihiro001.png | 45w, girl in car with flowers | 66w, girl+boy, flowers, car interior | **−0.039**: dense regions confused girl/boy; context became noisier. NLI contradiction rose (0.008→0.041). |
| thumb-chihiro002.png | 47w, woman on rock (wrong) | 75w, girl + monster statue in forest | **+0.239**: OD+dense corrected "woman on rock" → "girl + monster statue". CLIP 0.209→0.260, NLI 0.287→0.029. |
| thumb-chihiro004.png | 30w, coffee/bread (hallucinated) | 80w, family meal, hot dog, fish | **−0.061**: improved story longer (80w, length valid) but NLI contradiction jumped (0.018→0.205) — dense regions added "dog" (false) and "hot dog" (uncertain). |
| thumb-chihiro005.png | 50w, man in suit on ledge | 53w, boy with green hair on balcony | **+0.121**: detailed caption corrected "man in suit" → "boy with green hair". CLIP 0.203→0.249, NLI 0.050→0.052. |
| thumb-chihiro006.png | 58w, pig painting man | 47w, pig in kitchen with food | **−0.014**: both reasonable; improved slightly shorter. NLI improved (0.057→0.010) but CLIP dipped (0.263→0.251). |
| thumb-chihiro007.png | 76w, waiter/coffee (hallucinated) | 118w, Chinese restaurant, lanterns, bird | **+0.120**: dense regions gave "lantern, stool, bowl, bird". CLIP 0.249→0.267, NLI 0.191→0.012. Length valid! |
| thumb-chihiro008.png | 76w, golden retriever (hallucinated) | 36w, video game screenshot, sword | **+0.230**: detailed caption corrected "town + dog" → "video game screenshot". CLIP 0.213→0.300, NLI 0.059→0.062. But story too short (36w). |

**Key pattern:** Improvements come from Florence-2 correcting BLIP's errors (wrong characters, wrong scene). Regressions come from Florence-2's own errors (false "dog" in 004, girl/boy confusion in 001) or from longer contexts confusing the small Qwen model.

## Limitations

1. **Length control still broken.** 6/8 improved stories outside 80–120 (36, 47, 53, 61, 66, 75 words; only 80 and 118 are valid). Qwen 0.5B cannot reliably hit word count. `--fix-length` retries help but multiply runtime.
2. **NLI compares story to caption/context, not image.** Favors improved pipeline (its context is richer). Only CLIP is image-aware; gain is small (+0.025).
3. **Florence-2 makes its own errors.** `thumb-chihiro001.png`: dense regions say both "girl" and "boy". `thumb-chihiro004.png`: dense regions add "dog" not in image. `thumb-chihiro008.png`: calls frame "video game screenshot" and adds sword.
4. **Continuity metrics not discriminative.** Entity consistency = 1.0 for both (recurring entities: person, building — trivial). Transition markers = 0 (Qwen doesn't use them). Adjacent similarity = 1.0 (entity sets overlap heavily). Need better continuity signal.
5. **Thresholds untuned.** 0.60 grounding pass, 0.15–0.30 CLIP range chosen without data. 8 images, 1 run, greedy seed=0.
6. **No fluency/narrative quality metric.** Stories can be repetitive, incoherent, or grammatically odd and still score well.
7. **Repetition rate near zero** for both — distinct3 ~1.0. Qwen 0.5B with repetition_penalty=1.05 doesn't repeat trigrams.
8. **Multi-image story quality low.** `combined_story.json` invents names/events; 0.5B model cannot handle long sequence prompt.

## Finding

**Florence-2's structured visual extraction (detailed caption + OD + dense regions) raises grounding score by +0.083 and halves NLI contradiction, but 3/8 images regress (001, 004, 006), at least two of them due to Florence-2's own mispredictions (false "dog", gender confusion). The single-sentence BLIP bottleneck is real; the fix works on net, but the small Qwen model remains the weak link for length control and multi-frame coherence.**

## Reproducibility

```powershell
# One-time setup (needs internet)
pip install -r requirements.txt
python download_models.py

# Offline runs (PowerShell)
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

# Single image, improved pipeline
python main.py "D:\Downloads\images\chihiro003.jpg" --improved

# Full comparison (baseline vs improved) on all 8 images
python main.py "D:\Downloads\images" --both --output results_new.csv

# Multi-image story (sequence)
python main.py "D:\Downloads\images" --multi

# Self-test (synthetic strings)
python metric.py
```

**Models (cached in `HF_HOME=D:\AI-Models\huggingface`):**
- `Salesforce/blip-image-captioning-base` (baseline captioner)
- `Qwen/Qwen2.5-0.5B-Instruct` (story generator, both pipelines)
- `florence-community/Florence-2-base` (improved seeing stage)
- `openai/clip-vit-base-patch32` (metric: CLIP)
- `cross-encoder/nli-MiniLM2-L6-H768` (metric: NLI)

**Environment:** Windows 11, Python 3.13.9, torch 2.13.0 (CPU), transformers 5.15.0. Greedy decoding, fixed seed=0 → deterministic stories on same machine. Runtimes vary ±20%.