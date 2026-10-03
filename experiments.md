# Experiments: Image -> Story

All numbers come from one run of `compare.py` over the 8 images in `D:\Downloads\images`
(see `results.csv`, `stories.json`, `compare.log`). Everything ran offline (`HF_HUB_OFFLINE=1`) on a
CPU-only laptop (Python 3.13.9, torch 2.13.0, transformers 5.15.0).

## Important caveat about the baseline

The instructor's baseline code was not available to me, so `baseline.py` is a **stand-in** I wrote from the
spec: BLIP (`Salesforce/blip-image-captioning-base`) -> one sentence -> Qwen2.5-0.5B-Instruct
(greedy, `max_new_tokens=200`, `repetition_penalty=1.05`) asked for an 80-120 word story. Its prompt and
settings differ from the instructor's, so the baseline numbers below are **not** comparable to the
instructor's baseline. The method (same images, same metric, one change) carries over; re-run `compare.py`
against the real baseline before trusting any gain.

## Failures observed in the baseline (from `results.csv` / `stories.json`)

The three failures below come from reading the stories; confirm each against the actual image before submitting.

1. **Wrong length passes nothing, and the baseline never reaches 80 words.** 0 of 8 baseline stories are
   80-120 words (range 30-76). The baseline's only built-in check is word count.
2. **Invented details with no basis in the input.** `chihiro003`: the caption is "a man walking down the
   street" and the story adds "jeans and a t-shirt that shows off his muscular build". `chihiro008`: the caption
   has "a man and a dog" and the story calls it "a golden retriever".
3. **The story model sees one sentence**, so stories are generic (`chihiro007`: "a restaurant with a table
   and chairs" -> a story about coffee, bread and a waiter that nothing in the input supports).

## Module 10: measurement

`metric.py` scores `(image, caption, story)` and records runtime.

    grounding_score = 0.4*clip_n + 0.4*(1 - nli_contra_mean) + 0.2*(1 - attribute_conflict)
    clip_n = clip((clip_image_story_mean - 0.15) / 0.15, 0, 1)
    grounding_pass = grounding_score >= 0.60 AND no attribute conflict AND 80 <= words <= 120

- CLIP (`openai/clip-vit-base-patch32`): similarity between the image and each story sentence.
- NLI (`cross-encoder/nli-MiniLM2-L6-H768`): probability that each story sentence contradicts the caption.
- Attribute conflict: a colour/material word in the story that conflicts with the caption's.

Self-test (`selftest.csv`, synthetic strings on one local image): consistent story 0.928 (pass),
contradicting story 0.158 (fail), while both pass the baseline's word-count check.

## Module 7: the one change to the seeing stage

Replace BLIP's single sentence with a Florence-2 description: `<MORE_DETAILED_CAPTION>` plus the object labels from
`<OD>`, passed to the **same** Qwen story model with the same prompt (`seeing.py`). Model:
`florence-community/Florence-2-base` (the native transformers conversion of `microsoft/Florence-2-base`;
the original checkpoint's custom code failed to load on transformers 5.15).

Deviations made to fit CPU time (88 s -> about 15 s per image), all applied before the reported run:
dropped `<OCR>` (returned junk, e.g. "E") and `<CAPTION>` (redundant), greedy decoding instead of 3 beams,
KV cache on. Not ablated, so I cannot say how much each affected quality.

## Results (8 images, same photos, same metric)

| | Baseline | Improved |
|---|---|---|
| Mean grounding score | 0.783 | 0.845 |
| Mean CLIP image-vs-story | 0.233 | 0.255 |
| Length valid (80-120 words) | 0/8 | 3/8 |
| Grounding pass | 0/8 | 3/8 |
| Mean words per story | 56.6 | 93.6 |
| Mean seeing time (s) | 2.2 | 15.4 |
| Mean story time (s) | 8.7 | 15.1 |

Per image, grounding score (baseline -> improved): 003: 0.740 -> 0.887; 001: 0.856 -> 0.864;
002: 0.644 -> 0.776; 004: 0.895 -> 0.786; 005: 0.722 -> 0.804; 006: 0.878 -> 0.844;
007: 0.787 -> 0.870; 008: 0.746 -> 0.929.

Runtime: about 11 s -> about 30 s per image end to end (roughly 2.8x), measured with other work idle during
the final run.

Reproducibility: the whole run was repeated (`compare_rerun.log`). All 8 stories' scores, word counts and pass/fail
were identical (greedy decoding, fixed seed); only timings varied (mean baseline story time 8.7 s vs 13.3 s,
improved seeing time 15.4 s vs 15.9 s), so quote runtimes as approximate. `stories.json` holds the first run's timings.

## Second run: same length/cut-off fix applied to BOTH pipelines (`--fix-length`)

The first run's pass counts were driven by story length, so `compare.py --fix-length` applies one fix to the shared
story stage of both pipelines (drop a cut-off last sentence, stop at a sentence boundary under 120 words, retry
with a stricter length prompt up to 3 times). The seeing stage is still the only difference. Output:
`results_fixlen.csv`, `compare_fixlen.log`. Original run unchanged (`results.csv`).

| | Baseline | Improved |
|---|---|---|
| Mean grounding score | 0.756 | 0.837 |
| Mean CLIP image-vs-story | 0.240 | 0.254 |
| Mean NLI contradiction (lower better) | 0.207 | 0.103 |
| Length valid | 6/8 | 5/8 |
| Grounding pass | 5/8 | 5/8 |
| Truncated stories | 0 | 0 |
| Mean story time (s) | 28.6 | 19.8 |

- With length equalised, **pass counts tie (5/8 vs 5/8)**: the 0/8 -> 3/8 gain in the first run was a length artifact.
- The **grounding-score gain remains** (+0.08) and NLI contradiction halves; improved wins on 6 of 8 images, most on
  `008` (0.488 -> 0.929). It still loses on `004` (0.902 -> 0.675) and `006` (0.905 -> 0.844).
- Improved stories are still too short on `005` (78 words), `006` (71) and `008` (63).
- Caveat (unchanged): NLI compares the story with the caption it was built from, which favours the improved pipeline;
  CLIP is the only image-aware signal and its gain is small (+0.014). Treat the grounding gain as suggestive, not proven.
- New reported fields in `metric.py`: `truncated` and `distinct3` (repeated-trigram score; both pipelines ~0.99, so
  repetition is not a problem here). They are not part of `grounding_score`.

## What the data supports

- Grounding score rose on 6 of 8 images and CLIP image similarity on 7 of 8; both gains are small (+0.06 and +0.02).
- Improved stories are visibly more specific and tied to what Florence-2 described.

## What it does not support / honest limitations

- **The pass-rate gain is mostly length.** `grounding_pass` requires 80-120 words, and baseline stories were all
  shorter. 0/8 -> 3/8 says little about grounding.
- **Two images got worse:** `004` (0.895 -> 0.786, NLI contradiction 0.018 -> 0.327) and `006` (0.878 -> 0.844).
- **Length is still wrong for 5/8 improved stories** (68, 61, 71, 171, 63 words). Qwen 0.5B does not control length.
- **The metric cannot catch the seeing model's own errors.** NLI compares the story to the caption the story was
  built from, so it favours the improved pipeline. Only CLIP looks at the image and it is weak (values compressed
  into about 0.2-0.28). Florence-2 errors seen on reading: `001` calls Chihiro a "young boy"; `004` sees "hot dogs";
  `007` says "Chinese restaurant"; `008` calls the frame "a video game screenshot" and adds a sword.
- **The attribute-conflict rule never fired** on any of the 8 images, so it contributed nothing here.
- **Threshold and CLIP range (0.60, 0.15-0.30) are untuned**, chosen without data. 8 images is a small sample and
  there is one run, no seeds varied (greedy decoding, seed 0).
- Not measured: fluency, repetition, narrative quality.

## Reproduce

See `README.md`. `python compare.py "<images folder>"` writes `results.csv`; `python export_json.py` writes `stories.json`.
