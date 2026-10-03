# Image -> Story: BLIP baseline vs Florence-2 structured seeing

Challenge: Image -> Story (Modules 7 and 10). A local pipeline turns images into stories, and an automatic evaluation
measures how well the story matches the image. Everything runs locally and offline on CPU; no hosted inference API is used.

**Final submission artifacts** (produced by the commands in [Reproducing the results](#reproducing-the-results)):

| Artifact | Contents |
|---|---|
| `results_final.csv`, `vision_final.json` | Normal comparison, 8 images x 2 pipelines |
| `results_final_fixlen.csv`, `vision_final_fixlen.json` | Length-controlled comparison (`--fix-length`) |
| `combined_story.json` | One story across all 8 images (`--multi`) with the continuity proxy |
| `selftest.csv` | Output of `python metric.py` (synthetic strings) |
| `experiments.md` | Experiment write-up |

All other result files are historical or diagnostic runs, kept as evidence; see [Result files](#result-files).

## Problem

The baseline compresses each image into **one sentence** before the story model sees it. The story model never sees the image,
so details are lost and the model fills the gaps with invented ones. Failures observed in the baseline output:

1. **Wrong length.** 0/8 baseline stories are 80-120 words (range 30-76). The baseline's only built-in check is word count.
2. **Details with no basis in the input.** `chihiro003.jpg`: the caption is "a painting of a street scene with a man walking down the street" and the story adds "jeans and a t-shirt that shows off his muscular build". `thumb-chihiro008.png`: the caption mentions "a man and a dog" and the story calls the dog "a golden retriever".
3. **Generic stories from a one-sentence caption.** `thumb-chihiro007.png`: the caption is "a restaurant with a table and chairs" and the story adds coffee, bread and a waiter that nothing in the input supports.

## Baseline

`Image -> BLIP (Salesforce/blip-image-captioning-base) -> one sentence -> Qwen2.5-0.5B-Instruct -> story`

> The original instructor baseline code was unavailable, so this repository uses a **stand-in baseline** that matches the documented
> architecture (BLIP caption -> Qwen2.5-0.5B-Instruct, 80-120 word story). It is not the exact official baseline, and its absolute
> numbers may differ from the official one.

## Improved pipeline

```text
IMAGE
  |
  +--> BLIP baseline                          (A: one sentence)
  |
  +--> Florence-2 seeing (B)                  detailed caption + object detection + dense region captions
          |
          v
       STRUCTURED VISUAL CONTEXT (JSON)       scene, characters, objects, actions, regions, mood
          |
          v
       CONTEXT BUILDER (deterministic)        JSON -> short prompt text
  |
  v
QWEN2.5-0.5B-INSTRUCT (same model, seed, decoding and prompt for A and B)
  |
  v
STORY --> EVALUATION (CLIP, NLI, attribute check, repetition, length, runtime)
```

**Module 7 claim:** replace BLIP's single-sentence image understanding with Florence-2 structured visual extraction. The detailed caption,
object detection, dense region captions, the structured JSON and the context builder are implementation components of this richer
seeing stage. **They were not ablated separately**, so any gain cannot be attributed to one of them.

- `seeing.py` runs Florence-2-base (`florence-community/Florence-2-base`, greedy decoding, CPU) with three tasks: `<MORE_DETAILED_CAPTION>`,
  `<OD>` and `<DENSE_REGION_CAPTION>`, then derives `characters`, `actions`, `spatial_relations` and `style_or_mood` from that text with
  fixed keyword lists (closed vocabularies; see Limitations). `<OCR>` is disabled (it returned junk on anime frames).
- `context_builder.py` is deterministic and has no ML and no image-specific vocabulary. It keeps the detected object labels in order,
  adds up to 5 further object descriptions, up to 3 region descriptions, and trims to 180 words.
- Multi-image mode (`--multi`) builds a sequence context in filename order. Its continuity notes list only entities that occur in at least
  two different frames. File names are not shown to the story model.

### Structured vision JSON

```json
{
  "image_id": "...", "scene": "...", "description": "...",
  "objects": ["..."], "od_labels": ["..."], "characters": ["..."], "actions": ["..."],
  "relationships": [], "spatial_relations": [], "region_descriptions": ["..."], "style_or_mood": "...",
  "ocr_text": "", "runtime_s": 0.0, "model_load_s": 0.0
}
```

Fields are filled only from model output. `od_labels` are the object-detection labels used for entity matching across frames.

## Experimental fairness

Identical for baseline and improved: the Qwen2.5-0.5B-Instruct model, seed 0, greedy decoding, `repetition_penalty=1.05`, the story prompt
template, the evaluation formula and threshold. `--fix-length` applies the same mechanism to both. The only intended difference is the
seeing stage (and the text built from it). Model loading is excluded from every per-image runtime.

**Length-controlled** (`--fix-length`): up to 3 greedy attempts with progressively stricter length prompts. Each attempt is cut to whole sentences
(a cut-off last sentence is dropped) and to at most 120 words. The first attempt with 80-120 words is returned; otherwise the attempt closest to
100 words. This does **not** produce equal word counts for the two pipelines.

## Evaluation

| Metric | Meaning |
|---|---|
| `grounding_score` | `0.4*clip_n + 0.4*(1 - nli_contra_mean) + 0.2*(0 if attribute_conflict else 1)` |
| CLIP (`clip_image_story_mean`) | Cosine similarity between the **image** and each story sentence (ViT-B/32), averaged; `clip_n = clip((cos - 0.15)/0.15, 0, 1)`. The only signal that compares the story with the image itself. |
| NLI (`nli_contra_mean`) | Mean contradiction probability between the supplied context (caption or structured context) and each story sentence. It measures **textual consistency with the extracted visual context**; it is not independent verification of the image. |
| `attribute_conflict` | Deterministic check of colour and material words: a different colour or material than the context names, or a material whose usual colour the context's colour excludes (for example "white cabinets" vs "oak cabinet"). Heuristic; can raise false alarms. |
| `length_valid` | `80 <= words <= 120`. A benchmark requirement, **not** a quality measure: a story can be well aligned but too short, or the right length and visually wrong. |
| `grounding_pass` | `grounding_score >= 0.60` AND no attribute conflict AND `length_valid` |
| Repetition | Fraction of repeated word trigrams (reported only) |
| Runtime | `see_s` + `story_s` = `generation_s`; evaluation cost `eval_clip_s`, `eval_nli_s`, `eval_rules_s`, `eval_total_s` is reported separately (`time.perf_counter`) |

`grounding_score` is a composite automatic proxy, not ground truth. The 0.60 threshold and the CLIP range are untuned.

**Continuity** (`--multi` only) is a lightweight structural proxy based on entity overlap and transition words. It does not fully evaluate narrative
coherence. For ordinary single-image evaluation these metrics are not computed or reported.

## Results

Source of truth: `results_final.csv` (normal) and `results_final_fixlen.csv` (length-controlled), 8 images, one run, 16/16 evaluations
succeeded in each. The two comparisons use the same images.

| Metric | Baseline (normal) | Improved (normal) | Delta | Baseline (length-controlled) | Improved (length-controlled) | Delta |
|---|---|---|---|---|---|---|
| Mean grounding score | 0.783 | 0.856 | +0.072 | 0.756 | 0.827 | +0.070 |
| Mean CLIP image-story similarity | 0.233 | 0.257 | +0.024 | 0.240 | 0.254 | +0.014 |
| Mean NLI contradiction (lower is better) | 0.098 | 0.075 | -0.023 | 0.207 | 0.061 | -0.145 |
| Mean repetition rate (lower is better) | 0.0034 | 0.0187 | +0.0154 | 0.0068 | 0.0034 | -0.0034 |
| Length valid (80-120 words) | 0/8 | 1/8 | | 6/8 | 6/8 | |
| Grounding pass | 0/8 | 1/8 | | 5/8 | 5/8 | |
| Mean words per story | 56.6 | 62.6 | | 92.8 | 93.1 | |
| Images improved / regressed (grounding score) | | 5 / 3 | | | 5 / 3 | |

Per image, normal mode:

| Image | Baseline grounding | Improved grounding | Delta | Delta CLIP | Delta NLI | Words (baseline / improved) |
|---|---|---|---|---|---|---|
| chihiro003.jpg | 0.740 | 0.856 | +0.116 | +0.032 | -0.075 | 71 / 99 |
| thumb-chihiro001.png | 0.856 | 0.817 | -0.039 | -0.010 | +0.033 | 45 / 66 |
| thumb-chihiro002.png | 0.644 | 0.911 | +0.267 | +0.068 | -0.218 | 47 / 65 |
| thumb-chihiro004.png | 0.895 | 0.823 | -0.073 | -0.020 | +0.050 | 30 / 59 |
| thumb-chihiro005.png | 0.722 | 0.841 | +0.119 | +0.051 | +0.042 | 50 / 41 |
| thumb-chihiro006.png | 0.878 | 0.975 | +0.097 | +0.029 | -0.047 | 58 / 66 |
| thumb-chihiro007.png | 0.787 | 0.750 | -0.037 | -0.011 | +0.019 | 76 / 52 |
| thumb-chihiro008.png | 0.746 | 0.873 | +0.127 | +0.050 | +0.013 | 76 / 53 |

Per image, length-controlled:

| Image | Baseline grounding | Improved grounding | Delta | Delta CLIP | Delta NLI | Words (baseline / improved) |
|---|---|---|---|---|---|---|
| chihiro003.jpg | 0.800 | 0.887 | +0.087 | +0.033 | +0.001 | 107 / 110 |
| thumb-chihiro001.png | 0.811 | 0.783 | -0.027 | +0.009 | +0.129 | 66 / 114 |
| thumb-chihiro002.png | 0.691 | 0.712 | +0.021 | +0.049 | -0.224 | 86 / 116 |
| thumb-chihiro004.png | 0.902 | 0.827 | -0.075 | -0.020 | +0.054 | 52 / 59 |
| thumb-chihiro005.png | 0.677 | 0.821 | +0.145 | +0.019 | -0.236 | 112 / 61 |
| thumb-chihiro006.png | 0.905 | 0.851 | -0.054 | -0.033 | -0.086 | 82 / 106 |
| thumb-chihiro007.png | 0.777 | 0.911 | +0.133 | +0.031 | -0.127 | 117 / 92 |
| thumb-chihiro008.png | 0.488 | 0.820 | +0.333 | +0.024 | -0.675 | 120 / 87 |

### Runtime

| Stage (mean seconds per image, model loading excluded) | Baseline (normal) | Improved (normal) | Baseline (length-controlled) | Improved (length-controlled) |
|---|---|---|---|---|
| Seeing / vision | 1.4 | 15.5 | 1.5 | 16.5 |
| Story generation | 6.3 | 7.1 | 22.2 | 21.9 |
| **Generation total** | **7.7** | **22.6** | **23.7** | **38.5** |
| Evaluation (CLIP + NLI + rules; not part of generation) | 0.92 | 1.22 | 1.17 | 1.94 |

Seeing is slower with Florence-2 (three tasks instead of one BLIP call). Length control multiplies story time because of retries.
Per-image evaluation time varies; the median `eval_total_s` is 0.93 s (normal) and 1.27 s (length-controlled).

### What the results do and do not show

- In both comparisons the improved pipeline has a higher mean grounding score, higher CLIP similarity and lower NLI contradiction.
  Both were measured on 8 images in a single run, with an untuned threshold, so the sizes are indicative only.
- The normal-mode pass rate (0/8 -> 1/8) mostly reflects length: only 1/8 improved stories are 80-120 words.
- Under length control, the pass rate does not change (5/8 vs 5/8), and 3 of 8 images regress.
- Repetition is higher for the improved pipeline in the normal run (0.0034 -> 0.0187) and lower in the length-controlled run (0.0068 -> 0.0034).
  Repeated trigrams are rare in absolute terms, so this is not a finding.
- The gain depends on how the context is formatted: the same Florence-2 model gave different gains in earlier versions of the context
  builder (table below). With 8 images and one run this cannot be separated from run-to-run variation.

### Result files

| Result file | Mean grounding (baseline -> improved) | Grounding pass | Length valid |
|---|---|---|---|
| `results_final.csv` (FINAL, normal) | 0.783 -> 0.856 (+0.072) | 0/8 -> 1/8 | 0/8 -> 1/8 |
| `results_final_fixlen.csv` (FINAL, length-controlled) | 0.756 -> 0.827 (+0.070) | 5/8 -> 5/8 | 6/8 -> 6/8 |
| `results_new.csv` (historical: earlier structured version, normal) | 0.783 -> 0.866 (+0.083) | 0/8 -> 2/8 | 0/8 -> 2/8 |
| `results_fixlen_new.csv` (historical: earlier structured version, length-controlled) | 0.756 -> 0.789 (+0.032) | 5/8 -> 4/8 | 6/8 -> 6/8 |
| `results.csv` (historical: earlier simple pipeline, normal) | 0.783 -> 0.845 (+0.062) | 0/8 -> 3/8 | 0/8 -> 3/8 |
| `results_fixlen.csv` (historical: earlier simple pipeline, length-controlled) | 0.756 -> 0.837 (+0.080) | 5/8 -> 5/8 | 6/8 -> 5/8 |

Historical rows come from earlier code (an earlier simple context, and a context builder that used an object whitelist) and from an earlier
metric version, so they are not comparable with the final rows and are kept only as evidence of how the earlier numbers were produced.

## Failure cases and regressions

- **`thumb-chihiro001.png` (regressed in both modes):** the structured context contains both "girl" and "boy" for the same figure
  (`Characters: girl, boy, human`).
- **`thumb-chihiro004.png` (regressed in both modes):** the characters list contains "dog". It comes from the region caption "man eating hot dog":
  the keyword matcher in `seeing.py` matched the word "dog" inside "hot dog". This is an extraction artifact, not a detected dog.
- **`thumb-chihiro006.png` (regressed only in length-controlled mode)** and **`thumb-chihiro007.png` (regressed only in normal mode)**:
  see the per-image tables above.
- **Attribute conflicts:** the attribute check fired on 1 of 32 scored rows (`thumb-chihiro002.png`, improved, length-controlled: "red").
  It contributes almost nothing to these results.
- **Length:** Qwen2.5-0.5B does not reliably hit 80-120 words; 1/8 (normal) and 6/8 (length-controlled) improved stories are in range.

## Limitations

- The baseline is a stand-in, not the official one.
- 8 images, one run, greedy decoding with seed 0; the threshold (0.60) and CLIP range are untuned.
- CLIP is the only image-aware signal; NLI and the attribute check compare the story with generated text, so errors made by the seeing stage
  propagate into the story and are not penalised by them.
- Closed vocabularies: character and action keywords in `seeing.py`, colour/material words in `metric.py`, transition words in `metric.py`.
  Words outside these lists are not recognised, and the "hot dog" artifact above comes from this approach.
- The attribute check is a word-list heuristic with a typical-colour rule for wood; it can raise false alarms and misses most conflicts.
- Continuity is an overlap-and-lexicon proxy, not a measure of narrative coherence. Generic recurring labels (for example "person") count like any other.
- Components of the improved seeing stage were not ablated.
- No fluency or narrative-quality metric.

## Multi-image story (`--multi`)

All 8 images in filename order, 333 words, frames used: 8, cut-off final sentence removed: True.
Continuity proxy: recurring entities (11): boy, chair, child, footwear, girl, human, human face, man, person, window, woman; entity_consistency 0.4545;
transition_markers 0; adjacent_similarity 0.2285.
These numbers describe lexical overlap only. The story itself is in `combined_story.json`. It still invents a named character ("Leo") and relatives that no frame shows,
and covers the frames unevenly; a 0.5B model cannot hold a long sequence prompt, so treat it as a demonstration, not as evidence of coherence.

## Installation

Python 3.10+ (tested on Python 3.13, Windows 11, CPU only, about 8 GB RAM for the model weights).

```powershell
pip install -r requirements.txt
python download_models.py        # once, needs internet
```

Models: `Salesforce/blip-image-captioning-base`, `Qwen/Qwen2.5-0.5B-Instruct`, `florence-community/Florence-2-base`,
`openai/clip-vit-base-patch32`, `cross-encoder/nli-MiniLM2-L6-H768`.

### Offline mode and cache location

The code sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; nothing needs the internet at inference time. It respects an existing
`HF_HOME`; otherwise Hugging Face's normal cache location is used. A custom cache is optional, for example:

```powershell
$env:HF_HOME = "D:\AI-Models\huggingface"   # optional; not required
```

## Usage

```powershell
python main.py images/ --both --output results_final.csv --vision-json vision_final.json     # normal comparison
python main.py images/ --both --fix-length --output results_final_fixlen.csv --vision-json vision_final_fixlen.json
python main.py images/ --multi                      # one story across all images -> combined_story.json
python main.py images/ --baseline | --improved      # one pipeline only, printed
python metric.py                                    # self-test with synthetic strings -> selftest.csv
```

Defaults: `--output results.csv`, `--vision-json vision.json`, `--multi-output combined_story.json`. **`results.csv` is also a historical result file in this
repository, so always pass `--output` when you do not want to overwrite it.** A failing image is recorded with `status=error` and an error message,
gets no metric values, and is excluded from the means; the run prints successful and failed evaluation counts.

| Flag | Description |
|---|---|
| `--baseline`, `--improved`, `--both` | Pipelines to run (`--both` is the default) |
| `--multi` | One story across all images in filename order, with the continuity proxy |
| `--fix-length` | Length-controlled generation (not used by `--multi`) |
| `--output FILE`, `--vision-json FILE`, `--multi-output FILE` | Output paths |

## Testing

```powershell
python test_pipeline.py            # 24 unit tests, no model inference
python test_pipeline.py --models   # plus 2 model-dependent tests (CLIP/NLI scoring, Florence-2 vision)
pytest -q                          # 24 passed, 2 skipped (model-dependent tests skipped unless --models or RUN_MODEL_TESTS=1)
```

Importing `metric` loads CLIP and the NLI model from the local cache, so the cache must exist. The unit tests call production code:
the grounding formula helper, the grounding-pass rule, the 79/80/120/121 word-count boundaries, the attribute check (including the "white cabinets" vs "oak cabinet"
challenge example), repetition, continuity (including "recurring means at least two frames" and "not applicable for one image"), the context builder, image discovery,
CSV writing, error rows, summaries, `--vision-json` handling, and configuration.

## Reproducing the results

```powershell
$env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"
python main.py images/ --both --output results_final.csv --vision-json vision_final.json
python main.py images/ --both --fix-length --output results_final_fixlen.csv --vision-json vision_final_fixlen.json
python main.py images/ --multi
python metric.py
python test_pipeline.py --models
```

Greedy decoding with a fixed seed makes the stories repeatable on the same machine; runtimes vary between runs.

## Project structure

```text
main.py              CLI: --baseline / --improved / --both / --multi / --fix-length
baseline.py          BLIP baseline and the shared Qwen story generator (length rule, length-controlled mode)
seeing.py            Florence-2 seeing stage -> structured vision JSON
context_builder.py   deterministic context and sequence-context builder, entity helpers
metric.py            grounding score, repetition, attribute check, continuity proxy, runtime
test_pipeline.py     unit and optional model-dependent tests
download_models.py   one-time model download
experiments.md       experiment write-up
images/              the 8 evaluation frames
results_final*.csv, vision_final*.json, combined_story.json, selftest.csv   final artifacts
results*.csv, results_*_vision.json, stories.json                           historical runs
```

## Visual examples

Stories below are copied programmatically from `results_final.csv` (normal run) and are not edited.

### 1. chihiro003.jpg

<p align="center"><img src="images/chihiro003.jpg" alt="chihiro003.jpg" width="360"/></p>

| Pipeline | Grounding | Words | Story |
|---|---|---|---|
| Baseline (BLIP) | 0.740 | 71 | A man steps out onto the bustling street, his footsteps echoing through the concrete. He's dressed in a casual yet stylish outfit, a pair of jeans and a t-shirt that shows off his muscular build. The sun is shining brightly, casting a warm glow over the cityscape below. The man's eyes are fixed on the sidewalk, looking for something to grab. He's walking slowly, taking his time to enjoy the moment. |
| Improved (Florence-2) | 0.856 | 99 | In the heart of a bustling European city, a vibrant street scene unfolded. Children played joyfully under the twinkling Christmas lights, their laughter echoing through the air. A man, dressed in a festive attire, walked down the street, his eyes sparkling with excitement. He stopped at a colorful building with its windows adorned with festive decorations, which he admired with a sense of pride. As he continued, he passed by another building with its windows filled with red lanterns, creating a festive atmosphere. The street was alive with the sounds of children's laughter and the cheerful chatter of neighbors. |

### 2. thumb-chihiro001.png

<p align="center"><img src="images/thumb-chihiro001.png" alt="thumb-chihiro001.png" width="360"/></p>

| Pipeline | Grounding | Words | Story |
|---|---|---|---|
| Baseline (BLIP) | 0.856 | 45 | A girl sat in the back of a car, surrounded by colorful flowers. The sun shone through the window, casting a warm glow on her face. She was lost in thought as she gazed at the vibrant blooms, feeling grateful for the beauty around her. |
| Improved (Florence-2) | 0.817 | 66 | In the heart of a bustling city, a young girl sat in the back seat of a sleek, modern car, her eyes fixed on the vibrant blooms around her. She held the bouquet in one hand, her fingers tracing the delicate petals with practiced ease. The scene was captured by a camera lens, capturing the essence of a moment that would soon be shared with others. |

### 3. thumb-chihiro002.png

<p align="center"><img src="images/thumb-chihiro002.png" alt="thumb-chihiro002.png" width="360"/></p>

| Pipeline | Grounding | Words | Story |
|---|---|---|---|
| Baseline (BLIP) | 0.644 | 47 | A woman sits alone on a rocky outcropping, gazing out at the car that drives by. The sun sets over the horizon, casting long shadows across the landscape. She takes a deep breath, feeling the cool breeze on her face as she watches the world pass by. |
| Improved (Florence-2) | 0.911 | 65 | In the heart of the dense forest, a young girl stood beside a large green monster statue, her eyes sparkling with laughter as she gazed at the towering figure. She wore a simple green dress and a pair of sturdy boots, her face a mix of mischief and innocence. The car, its headlights casting long shadows, passed by, its occupants oblivious to the child's presence. |

### 4. thumb-chihiro004.png

<p align="center"><img src="images/thumb-chihiro004.png" alt="thumb-chihiro004.png" width="360"/></p>

| Pipeline | Grounding | Words | Story |
|---|---|---|---|
| Baseline (BLIP) | 0.895 | 30 | A man and woman sit at a table, sipping on coffee and savoring their meal. The aroma of freshly baked bread fills the air as they chat about their day. |
| Improved (Florence-2) | 0.823 | 59 | In the bustling restaurant, the family sat at a table, their faces all focused on their meal. The man, with his golden fur and expressive eyes, was savoring his hot dog. The woman, with her long hair and a mischievous grin, was munching on a red fish. The child, with a curious look, watched them both with wide eyes. |

### 5. thumb-chihiro005.png

<p align="center"><img src="images/thumb-chihiro005.png" alt="thumb-chihiro005.png" width="360"/></p>

| Pipeline | Grounding | Words | Story |
|---|---|---|---|
| Baseline (BLIP) | 0.722 | 50 | A man in a formal suit and tie stands on a ledge, looking out over a cityscape. The sun glints off his polished leather jacket, casting a warm glow on his face. He takes a deep breath, feeling the tension in his muscles as he prepares to make a statement. |
| Improved (Florence-2) | 0.841 | 41 | In a serene garden, a young man with green hair stands on a red railing, his eyes fixed on the distant balcony. He holds a small, delicate bird in his hand, its wings fluttering gently as it watches the world below. |

### 6. thumb-chihiro006.png

<p align="center"><img src="images/thumb-chihiro006.png" alt="thumb-chihiro006.png" width="360"/></p>

| Pipeline | Grounding | Words | Story |
|---|---|---|---|
| Baseline (BLIP) | 0.878 | 58 | A pig, with its long snout and bushy tail, was painting a picture of a man in a field. The pig's eyes were fixed on the figure, his brush strokes soft and gentle. The man's face was serene, his expression calm and composed. The pig's artistry was evident in every stroke, capturing the essence of the man's presence. |
| Improved (Florence-2) | 0.975 | 66 | In a vibrant world, a pig with a blue shirt stood proudly, its eyes twinkling with mischief. Its green leafy vegetable in its mouth was a sight to behold, a testament to its hearty appetite. The scene was captured by a vibrant cartoon pig with a blue shirt and a green leafy vegetable in its mouth, adding a touch of whimsy to the otherwise mundane scene. |

### 7. thumb-chihiro007.png

<p align="center"><img src="images/thumb-chihiro007.png" alt="thumb-chihiro007.png" width="360"/></p>

| Pipeline | Grounding | Words | Story |
|---|---|---|---|
| Baseline (BLIP) | 0.787 | 76 | In the dimly lit restaurant, the soft glow of the overhead light bathed the tables in a warm, inviting light. The air was filled with the scent of freshly baked bread and the aroma of steaming coffee. A friendly waiter greeted each patron with a warm smile, and the conversation flowed effortlessly as they sipped their coffee and ate their meals. The atmosphere was cozy and welcoming, perfect for a special occasion or a casual dinner. |
| Improved (Florence-2) | 0.750 | 52 | In the dimly lit alleyway, a bird perched on a stool, its feathers shimmering under the lantern's soft glow. The bowl sat precariously on the chair, its contents a mix of rice and vegetables, while the stool was occupied by a curious passerby who watched with a mix of curiosity and amusement. |

### 8. thumb-chihiro008.png

<p align="center"><img src="images/thumb-chihiro008.png" alt="thumb-chihiro008.png" width="360"/></p>

| Pipeline | Grounding | Words | Story |
|---|---|---|---|
| Baseline (BLIP) | 0.746 | 76 | In the heart of a bustling town, a man sat on a bench, his eyes fixed on a distant dog. The dog, a golden retriever, wagged its tail in greeting. The man's face lit up as he watched the dog play with a toy, its tail swishing back and forth. The dog, in turn, wagged its tail back and forth, its eyes sparkling with joy. The man smiled, feeling grateful for the simple bond between them. |
| Improved (Florence-2) | 0.873 | 53 | In a dimly lit room, a woman stood before a small red building with a green door and red lights. She held a figure in her arms, which was a man. The scene was set against a dark background, with shadows playing on the walls and the figure's face obscured by a hood. |

