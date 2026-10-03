# Experiments: Image → Story Challenge

All numbers come from two runs of `python main.py images/ --both` on the 8 images in `images/` (frames from *Spirited Away*). Everything ran offline (`HF_HUB_OFFLINE=1`) on a CPU-only laptop (Python 3.13, torch 2.13, transformers 5.15). 

Run 1 (normal): `python main.py images/ --both --output results_new.csv`
Run 2 (length-equalised): `python main.py images/ --both --fix-length --output results_fixlen_new.csv`

## Problem

The baseline pipeline uses a single BLIP caption (one sentence) as the only visual information passed to the story model. This creates a severe bottleneck: most visual detail (objects, characters, actions, spatial layout, text) is lost before story generation. The story model hallucinates details to fill the gaps, producing generic or contradictory narratives.

## Baseline

**Pipeline A (baseline):** `Image → BLIP (Salesforce/blip-image-captioning-base) → 1-sentence caption → Qwen2.5-0.5B-Instruct → story`

- BLIP generates one short caption (e.g., "a girl sitting in the back of a car with a bunch of flowers")
- Same Qwen2.5-0.5B-Instruct model used for both pipelines (greedy, seed=0, repetition_penalty=1.05)
- Prompt asks for 80–120 word story; no length enforcement in normal run
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

Replacing the single BLIP sentence with a structured Florence-2 description (detailed caption + object detection + dense region captions) will:
- Provide the story model with concrete visual facts (objects, characters, actions, spatial layout)
- Reduce hallucination by giving the model something to ground to
- Improve CLIP image-story similarity and reduce NLI contradiction
- Enable continuity by making recurring entities explicit in the structured context

## Change (Module 7)

**Scope note.** The core change is to the seeing stage: Florence-2-base replaces BLIP. In the same pass, three supporting pieces were added: dense region captions, a structured JSON, and a deterministic context builder. They were **not ablated** (never run one at a time), so the measured gain below cannot be attributed to any single one of them.

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
| `nli_contra_mean` | NLI contradiction prob (context vs story sentences) | `cross-encoder/nli-MiniLM2-L6-H768` mean P(contradiction) | lower better |
| `attribute_conflict` | Color/material word in story not in context | Set overlap per group (color, material) | 0 |
| **Length valid** | Story in 80–120 words | `80 ≤ words ≤ 120` | hard check |
| **Repetition rate** | Fraction of repeated trigrams | `repeated_trigrams / total_trigrams` | lower better (report only) |
| **Entity consistency** | Fraction of recurring visual entities mentioned in story | Entities appearing in 2+ frames ∩ story words / recurring entities | higher better (report only) |
| **Transition markers** | Count of temporal/transition words in story | Lexicon: then, next, after, later, suddenly, meanwhile... | report only |
| **Adjacent similarity** | Entity Jaccard between adjacent frame contexts | Mean over adjacent pairs | report only |
| **Runtime** | Monotonic timers | `seeing_s`, `story_s`, `total_s` | report only |

**Critical note on NLI:** The NLI model evaluates contradiction between the **story sentences and the generated caption/context** — not against the image directly. It measures *consistency with the extracted visual context*, not independent image verification. Only CLIP directly compares image to story.

`grounding_pass` = all three: score ≥ 0.60, no attribute conflict, length valid.

### Evaluation Interpretation Table

| Evaluation Signal | Normal Baseline | Normal Improved | Fixlen Baseline | Fixlen Improved | Interpretation |
|-------------------|----------------|----------------|-----------------|----------------|----------------|
| Grounding Score   | 0.783          | **0.866**      | 0.756           | **0.789**      | Higher = stronger alignment |
| CLIP Similarity   | 0.233          | **0.258**      | 0.240           | **0.253**      | Higher = stronger image-story semantic similarity |
| NLI Contradiction | 0.098          | **0.055**      | 0.207           | **0.088**      | Lower = fewer textual contradictions with context |
| Repetition Rate   | 0.0034         | 0.0073         | 0.0068          | 0.0118         | Lower = less repeated phrasing |
| Entity Consistency| 1.000          | 1.000          | 1.000           | 1.000          | Higher = stronger cross-frame consistency |
| Length Valid      | 0/8            | **2/8**        | 6/8             | 6/8            | Requirement compliance |
| Grounding Pass    | 0/8            | **2/8**        | **5/8**         | 4/8            | Composite threshold |
| Runtime (total)   | 13.4 s         | 27.0 s         | 29.4 s          | 41.9 s         | Lower = more efficient |

> **Note:** NLI evaluates contradiction between the **story and the generated context/caption** — it measures *consistency with the extracted visual context*, not independent image verification. Only CLIP directly compares image to story. These are automatic proxies — not perfect human-quality measures.

## Normal Evaluation (no length control)

| Metric | Baseline (BLIP) | Improved (Florence-2) | Delta |
|--------|-----------------|----------------------|-------|
| Mean grounding score | 0.783 | **0.866** | **+0.083** |
| Mean CLIP image–story | 0.233 | **0.258** | **+0.025** |
| Mean NLI contradiction | 0.098 | **0.055** | **−0.043** |
| Mean repetition rate | 0.0034 | 0.0073 | +0.0039 |
| Length valid (80–120 words) | 0/8 | **2/8** | +2 |
| Grounding pass | 0/8 | **2/8** | +2 |
| Mean seeing time | 4.3 s | 18.1 s | +13.8 s |
| Mean story time | 9.1 s | 8.8 s | −0.3 s |
| Mean total time | 13.4 s | 27.0 s | +13.6 s |

Per-image grounding (baseline):  
chihiro003.jpg: 0.740 | thumb-chihiro001.png: 0.856 | thumb-chihiro002.png: 0.644 | thumb-chihiro004.png: 0.895 | thumb-chihiro005.png: 0.722 | thumb-chihiro006.png: 0.878 | thumb-chihiro007.png: 0.787 | thumb-chihiro008.png: 0.746

Per-image grounding (improved):  
chihiro003.jpg: 0.805 | thumb-chihiro001.png: 0.817 | thumb-chihiro002.png: 0.883 | thumb-chihiro004.png: 0.834 | thumb-chihiro005.png: 0.843 | thumb-chihiro006.png: 0.864 | thumb-chihiro007.png: 0.907 | thumb-chihiro008.png: 0.975

### Normal: Before vs After

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

**The pass-rate gain is largely a length artifact.** Grounding pass requires 80–120 words; baseline produced 0 valid stories, improved produced 2. See length-equalised check below.

## Length-Equalised Evaluation (--fix-length)

Run: `python main.py images/ --both --fix-length --output results_fixlen_new.csv`

With `--fix-length`, both pipelines retry generation up to 3 times with stricter length prompts until the story falls in 80–120 words (or return the best attempt). This controls the length confound.

| Metric | Baseline (BLIP) | Improved (Florence-2) | Delta |
|--------|-----------------|----------------------|-------|
| Mean grounding score | 0.756 | **0.789** | **+0.033** |
| Mean CLIP image–story | 0.240 | **0.253** | **+0.013** |
| Mean NLI contradiction | 0.207 | **0.088** | **−0.119** |
| Mean repetition rate | 0.0068 | 0.0118 | +0.0050 |
| Length valid (80–120 words) | 6/8 | 6/8 | 0 |
| Grounding pass | **5/8** | 4/8 | −1 |
| Mean seeing time | 2.4 s | 17.4 s | +15.0 s |
| Mean story time | 27.0 s | 24.5 s | −2.5 s |
| Mean total time | 29.4 s | 41.9 s | +12.5 s |

Per-image grounding (baseline fixlen):  
chihiro003.jpg: 0.800 | thumb-chihiro001.png: 0.811 | thumb-chihiro002.png: 0.691 | thumb-chihiro004.png: 0.902 | thumb-chihiro005.png: 0.677 | thumb-chihiro006.png: 0.905 | thumb-chihiro007.png: 0.777 | thumb-chihiro008.png: 0.488

Per-image grounding (improved fixlen):  
chihiro003.jpg: 0.607 | thumb-chihiro001.png: 0.783 | thumb-chihiro002.png: 0.870 | thumb-chihiro004.png: 0.858 | thumb-chihiro005.png: 0.790 | thumb-chihiro006.png: 0.870 | thumb-chihiro007.png: 0.911 | thumb-chihiro008.png: 0.621

### Length-Equalised: Before vs After

| METRIC | BASELINE | IMPROVED | DELTA |
|--------|----------|----------|-------|
| Mean grounding score | 0.756 | 0.789 | **+0.033** |
| Mean CLIP similarity | 0.240 | 0.253 | **+0.013** |
| Mean NLI contradiction | 0.207 | 0.088 | **−0.119** |
| Mean repetition rate | 0.0068 | 0.0118 | +0.0050 |
| Length valid (80–120) | 6/8 | 6/8 | 0 |
| Grounding pass | **5/8** | 4/8 | −1 |
| Mean seeing time (s) | 2.4 | 17.4 | +15.0 |
| Mean story time (s) | 27.0 | 24.5 | −2.5 |
| Mean total time (s) | 29.4 | 41.9 | +12.5 |

**With length equalised, the grounding gain shrinks from +0.083 to +0.033**, and **baseline wins on grounding pass (5/8 vs 4/8)** because both produce 6 valid-length stories (length-valid tied at 6/8 each) but baseline scores higher on the ones that pass. The large normal-mode pass gain (+2 passes) was largely a length artifact.

Per-image grounding deltas (fixlen):
- chihiro003.jpg: A=0.800 → B=0.607 (Δ=−0.193) **regressed**
- thumb-chihiro001.png: A=0.811 → B=0.783 (Δ=−0.027) **regressed**
- thumb-chihiro002.png: A=0.691 → B=0.870 (Δ=+0.179) **improved**
- thumb-chihiro004.png: A=0.902 → B=0.858 (Δ=−0.044) **regressed**
- thumb-chihiro005.png: A=0.677 → B=0.790 (Δ=+0.113) **improved**
- thumb-chihiro006.png: A=0.905 → B=0.870 (Δ=−0.035) **regressed**
- thumb-chihiro007.png: A=0.777 → B=0.911 (Δ=+0.133) **improved**
- thumb-chihiro008.png: A=0.488 → B=0.621 (Δ=+0.133) **improved**

**Fixlen summary:** Improved wins on 4/8 images, baseline wins on 4/8. The grounding gain is real but smaller (+0.033 vs +0.083). NLI contradiction is substantially reduced (−0.119). Baseline wins on grounding pass (5/8 vs 4/8) when length is controlled, even though length-valid is tied at 6/8 each.

## Runtime

| Stage | Baseline | Improved |
|-------|----------|----------|
| Vision/seeing (normal) | 4.3 s | 18.1 s |
| Vision/seeing (fixlen) | 2.4 s | 17.4 s |
| Story generation (normal) | 9.1 s | 8.8 s |
| Story generation (fixlen) | 27.0 s | 24.5 s |
| **Total (normal)** | **13.4 s** | **27.0 s** |
| **Total (fixlen)** | **29.4 s** | **41.9 s** |

Seeing is ~4–7× slower with Florence-2 (3 tasks vs 1). Story time similar in normal mode; fixlen multiplies story time ~3× due to retries. Measured after model loads (load times excluded).

## Per-Image Analysis

| Image | Normal Δ | Fixlen Δ | Normal Assessment | Fixlen Assessment |
|-------|----------|----------|-------------------|-------------------|
| chihiro003.jpg | +0.065 | −0.193 | Improved: dense regions gave "man + two children walking", "colorful buildings with lanterns" | **Regressed**: longer context confused Qwen; story less coherent |
| thumb-chihiro001.png | −0.039 | −0.027 | Regressed: dense regions confused girl/boy; context noisier | Regressed: gender confusion in dense regions |
| thumb-chihiro002.png | +0.239 | +0.179 | **Improved**: OD+dense corrected "woman on rock" → "girl + monster statue" | **Improved**: correction persists |
| thumb-chihiro004.png | −0.061 | −0.044 | Regressed: dense regions added false "dog" and "hot dog" | Regressed: false "dog" detection persists |
| thumb-chihiro005.png | +0.121 | +0.113 | **Improved**: detailed caption corrected "man in suit" → "boy with green hair" | **Improved**: correction persists |
| thumb-chihiro006.png | −0.014 | −0.035 | Slight regress: both reasonable | Slight regress |
| thumb-chihiro007.png | +0.120 | +0.133 | **Improved**: dense regions gave "lantern, stool, bowl, bird" | **Improved**: consistent gain |
| thumb-chihiro008.png | +0.230 | +0.133 | **Improved**: detailed caption corrected "town + dog" → "video game screenshot" | **Improved**: gain persists but smaller |

**Pattern:** Improvements come from Florence-2 correcting BLIP's errors (wrong characters, wrong scene). Regressions come from Florence-2's own errors (false "dog" in 004, gender confusion in 001) or from longer contexts confusing the small Qwen model (003 fixlen). The fix works on net but is not universal.

## Failure Cases & Limitations

| Failure Mode | Observed In | Impact |
|--------------|-------------|--------|
| **Gender confusion** | thumb-chihiro001.png | Dense regions output both "girl" and "boy" for same figure |
| **False object detection** | thumb-chihiro004.png | Dense regions add "dog" not present in image |
| **Incorrect scene interpretation** | thumb-chihiro008.png | Detailed caption calls frame "video game screenshot" and adds sword |
| **Length control failure** | 6/8 improved stories | Qwen 0.5B cannot reliably hit 80–120 words (range: 36–118 normal; 75–120 fixlen) |
| **Weak sequence coherence** | Multi-image story | 0.5B model cannot maintain long-range narrative continuity |
| **Increased vision latency** | All improved runs | 3 Florence-2 tasks vs 1 BLIP call |
| **Context confusion (fixlen)** | chihiro003.jpg fixlen | Longer structured context confused small Qwen model |

**A richer context is only useful if the underlying vision information is accurate.**

## Concrete Baseline Failures (Evidence)

### FAILURE 1: Hallucinated Attribute (chihiro003.jpg)
- **Image:** chihiro003.jpg (European street scene, man + two children walking)
- **Baseline caption:** "a painting of a street scene with a man walking down the street"
- **Baseline story claim:** "He's dressed in a casual yet stylish outfit, a pair of jeans and a t-shirt that shows off his muscular build"
- **Observed issue:** The caption mentions only "a man" — no jeans, no t-shirt, no muscular build. The story invents specific clothing and physique.
- **Metric signal:** Grounding 0.740 (moderate), CLIP 0.219 (low), NLI 0.111 (moderate contradiction)
- **Did metric detect it:** Partially — low CLIP and elevated NLI flag inconsistency, but grounding still 0.740 due to attribute_conflict=0 (no color/material conflict detected).

### FAILURE 2: Hallucinated Object Class (thumb-chihiro008.png)
- **Image:** thumb-chihiro008.png (night town, video game style, figure with sword)
- **Baseline caption:** "a scene of a town with a man and a dog"
- **Baseline story claim:** "The dog, a golden retriever, wagged its tail in greeting"
- **Observed issue:** The caption says "a dog" — the story invents the specific breed "golden retriever". The image shows a video game scene with a figure holding a sword; no dog is clearly visible.
- **Metric signal:** Grounding 0.746, CLIP 0.213, NLI 0.059, attribute_conflict=0
- **Did metric detect it:** Weakly — low CLIP suggests poor image-story alignment, but no attribute conflict triggered (breed not in color/material groups).

### FAILURE 3: Generic Hallucination from Underspecified Caption (thumb-chihiro007.png)
- **Image:** thumb-chihiro007.png (Chinese restaurant entrance, lanterns, stools, bird mural)
- **Baseline caption:** "a restaurant with a table and chairs"
- **Baseline story claims:** coffee, bread, waiter, conversation, cozy atmosphere
- **Observed issue:** The caption gives only "restaurant + table + chairs". The story invents an entire dining scene (coffee, bread, waiter, conversation) with no visual basis.
- **Metric signal:** Grounding 0.787, CLIP 0.249, NLI 0.191 (high contradiction), length 76 words
- **Did metric detect it:** Partially — NLI contradiction is high (0.191) because story claims contradict the minimal caption. CLIP is moderate. Grounding still 0.787.

### Why These Matter
These failures show the bottleneck: when the vision stage provides only a single sentence, the story model *must* hallucinate to produce a narrative. The metrics partially catch this (low CLIP, high NLI), but the composite grounding score can still be moderate because it weights multiple factors.

## Regression Analysis (Images Where Improved < Baseline)

### 1. thumb-chihiro001.png (Normal: −0.039, Fixlen: −0.027)
- **Cause:** Dense region captions output both "girl" and "boy" for the same figure. The context builder includes both, confusing the Qwen model.
- **Evidence:** Dense regions: "A young girl sitting on the back seat..." AND "A young boy sitting on a couch..."
- **Likely cause:** Florence-2's dense region captioning is ambiguous on this frame.
- **Effect:** Noisy context → higher NLI contradiction (0.008→0.041 normal; 0.181→0.192 fixlen).

### 2. thumb-chihiro004.png (Normal: −0.061, Fixlen: −0.044)
- **Cause:** Dense regions add false "dog" detection and uncertain "hot dog".
- **Evidence:** Dense regions: "man eating hot dog in restaurant", "woman eating red fish in restaurant"; characters list includes "dog".
- **Likely cause:** Florence-2 misclassifies a food item or background element as a dog.
- **Effect:** False entity in context → higher NLI contradiction (0.018→0.205 normal; 0.048→0.189 fixlen).

### 3. thumb-chihiro006.png (Normal: −0.014, Fixlen: −0.035)
- **Cause:** Both pipelines produce reasonable stories; improved is slightly shorter.
- **Evidence:** Baseline story describes pig painting (creative but wrong); improved describes pig in kitchen with food (accurate per dense regions).
- **Note:** CLIP dips slightly (0.263→0.251 normal; 0.257→0.242 fixlen) because improved story mentions "kitchen" which may not be strongly visually grounded.

### 4. chihiro003.jpg (Fixlen only: −0.193)
- **Cause:** Longer structured context confused the small Qwen model under length constraint.
- **Evidence:** Fixlen baseline grounding 0.800 → improved 0.607. Story words: baseline 107, improved 101.
- **Likely cause:** The richer context (3 region descriptions + characters + objects + actions) exceeds what the 0.5B model can reliably integrate under strict length constraints.

## Improvement Analysis (Strongest Gains)

### 1. thumb-chihiro002.png (Normal: +0.239, Fixlen: +0.179)
- **Baseline error:** Caption "a woman sitting on a rock next to a car" — completely wrong character (woman vs girl) and action (sitting on rock vs standing by monster).
- **Improved correction:** Detailed caption "young girl standing next to a large green monster statue in a forest", OD "person", dense "girl standing next to car with green monster".
- **Effect:** CLIP 0.209→0.260 (normal), NLI 0.287→0.029 (normal). Massive correction of baseline's fundamental scene error.

### 2. thumb-chihiro008.png (Normal: +0.230, Fixlen: +0.133)
- **Baseline error:** Caption "a scene of a town with a man and a dog" — wrong scene (town vs video game), wrong entities (dog not present).
- **Improved correction:** Detailed caption "screenshot from a video game... person holding a sword", no dog in dense regions.
- **Effect:** CLIP 0.213→0.300 (normal), NLI 0.059→0.062 (similar). Major scene correction.

### 3. thumb-chihiro007.png (Normal: +0.120, Fixlen: +0.133)
- **Baseline error:** Caption "a restaurant with a table and chairs" → story invents coffee, bread, waiter.
- **Improved correction:** Dense regions: "lantern", "stool", "bowl", "bird"; detailed caption "Chinese restaurant entrance".
- **Effect:** CLIP 0.249→0.267, NLI 0.191→0.012. Accurate visual entities enable grounded story.

### 4. thumb-chihiro005.png (Normal: +0.121, Fixlen: +0.113)
- **Baseline error:** Caption "a man in a suit and tie is standing on a ledge" — wrong character (man vs boy), wrong setting (ledge vs balcony).
- **Improved correction:** Detailed caption "young boy with green hair... standing on a balcony".
- **Effect:** CLIP 0.203→0.249, NLI similar. Character and setting correction.

## Limitations (Explicit)

1. **Length control unreliable.** 6/8 improved stories outside 80–120 normal; 2/8 outside fixlen. Qwen 0.5B cannot reliably hit word count. `--fix-length` retries help but multiply runtime ~3×.
2. **NLI evaluates context consistency, not image truth.** NLI premise = generated caption/context. Favors improved pipeline (richer context). Only CLIP directly measures image-story alignment.
3. **Florence-2 makes its own errors.** False "dog" (004), gender confusion (001), "video game screenshot" + sword (008).
4. **Continuity metrics not discriminative.** Entity consistency = 1.0 for both (trivial recurring: person, building). Transition markers = 0 (Qwen doesn't use them). Adjacent similarity = 1.0 (entity sets overlap heavily).
5. **Thresholds untuned.** 0.60 grounding pass, 0.15–0.30 CLIP range chosen without data. 8 images, 1 run, greedy seed=0.
6. **No fluency/narrative quality metric.** Stories can be repetitive, incoherent, or grammatically odd and still score well.
7. **Repetition rate near zero** for both — distinct3 ~1.0. Qwen 0.5B with repetition_penalty=1.05 doesn't repeat trigrams.
8. **Multi-image story quality low.** `combined_story.json` invents names/events; 0.5B model cannot handle long sequence prompt.
9. **No component ablation.** Cannot attribute gain to any one Florence-2 task (detailed caption vs OD vs dense regions).
10. **Small evaluation set, single run.** 8 images, 1 seed, greedy decoding. Results are indicative, not statistically robust.

## Finding

**Florence-2's structured visual extraction (detailed caption + OD + dense regions) raises grounding score by +0.083 normally and +0.033 when length is equalised, while substantially reducing NLI contradiction (−0.043 normal, −0.119 fixlen). In normal mode, grounding improved on 5/8 images and regressed on 3/8; in length-controlled mode, improved and baseline each win on 4/8 images, and baseline wins on grounding pass (5/8 vs 4/8) because both produce 6 valid-length stories. The single-sentence BLIP bottleneck is real and the richer visual representation helps on average, but Florence-2 introduces its own errors (false "dog", gender confusion) and the small Qwen model struggles to integrate longer contexts under length constraints.**

## Reproducibility

```powershell
# One-time setup (needs internet)
pip install -r requirements.txt
python download_models.py

# Offline runs (PowerShell)
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

# Single image, improved pipeline
python main.py images/chihiro003.jpg --improved

# Full comparison (baseline vs improved) on all 8 images
python main.py images/ --both --output results_new.csv

# Length-equalised comparison
python main.py images/ --both --fix-length --output results_fixlen_new.csv

# Multi-image story (sequence)
python main.py images/ --multi

# Self-test (synthetic strings)
python metric.py
python test_pipeline.py
```

**Models (cached in `HF_HOME=D:\AI-Models\huggingface`):**
- `Salesforce/blip-image-captioning-base` (baseline captioner)
- `Qwen/Qwen2.5-0.5B-Instruct` (story generator, both pipelines)
- `florence-community/Florence-2-base` (improved seeing stage)
- `openai/clip-vit-base-patch32` (metric: CLIP)
- `cross-encoder/nli-MiniLM2-L6-H768` (metric: NLI)

**Environment:** Windows 11, Python 3.13.9, torch 2.13.0 (CPU), transformers 5.15.0. Greedy decoding, fixed seed=0 → deterministic stories on same machine. Runtimes vary ±20%.