"""Grounding metric for Image -> Story (Module 10: evaluation + runtime logging).

grounding_score = 0.4*clip_n + 0.4*(1 - nli_contra_mean) + 0.2*(1 - conflict)
  clip_n    = clip((clip_image_story_mean - 0.15) / (0.30 - 0.15), 0, 1)
              (CLIP ViT-B/32 cosine for matching text is ~0.25-0.32, unrelated ~0.10-0.18)
  nli_contra_mean = mean P(contradiction | premise=caption, hypothesis=story sentence)
  conflict  = 1 if a color/material word in the story contradicts the caption, else 0
grounding_pass = grounding_score >= 0.60 AND no attribute conflict AND length_valid.
"""
import os, re, time, csv
from contextlib import ContextDecorator
from collections import Counter

# HF_HOME: use environment variable if set, otherwise let Hugging Face use its default cache
# os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModelForSequenceClassification

THRESHOLD = 0.60
TIMINGS = {}  # stage name -> list of seconds


class Timer(ContextDecorator):
    """Use as `with Timer('stage'):` or `@Timer('stage')`; appends seconds to TIMINGS."""
    def __init__(self, name): self.name = name
    def __enter__(self): self.t = time.perf_counter(); return self
    def __exit__(self, *a): TIMINGS.setdefault(self.name, []).append(time.perf_counter() - self.t)


with Timer("load_models"):
    _clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    _clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _nli_name = "cross-encoder/nli-MiniLM2-L6-H768"
    _nli_tok = AutoTokenizer.from_pretrained(_nli_name)
    _nli = AutoModelForSequenceClassification.from_pretrained(_nli_name).eval()
_CONTRA = [i for i, l in _nli.config.id2label.items() if l.lower() == "contradiction"][0]

GROUPS = {"color": {"white", "black", "red", "blue", "green", "yellow", "brown", "grey", "gray", "pink", "orange", "purple"},
          "material": {"wood", "wooden", "oak", "glass", "metal", "steel", "plastic", "stone", "marble", "brick", "leather"}}


def split_sentences(text): return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
def words(text): return set(re.findall(r"[a-z]+", text.lower()))


def attribute_conflict(caption, story):
    """Per group: if caption names words of the group and story names words of it not in caption -> offending."""
    cw, sw, bad = words(caption), words(story), []
    for g in GROUPS.values():
        if cw & g: bad += sorted((sw & g) - cw)
    return bad


@torch.no_grad()
def _clip_sims(image, texts):
    inp = _clip_proc(text=texts, images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
    out = _clip(**inp)
    return out.logits_per_image[0] / _clip.logit_scale.exp()  # cosine similarities


@torch.no_grad()
def _contradiction_probs(caption, sentences):
    inp = _nli_tok([caption] * len(sentences), sentences, return_tensors="pt", padding=True, truncation=True)
    return _nli(**inp).logits.softmax(-1)[:, _CONTRA]


def repetition_score(story: str) -> dict:
    """
    Lightweight deterministic repetition metrics.
    Returns dict with:
      - repeated_sentences: count of duplicate sentences
      - repeated_bigrams: count of repeated word bigrams
      - repeated_trigrams: count of repeated word trigrams
      - repetition_rate: repeated_units / total_units (using trigrams)
      - distinct3: 1 - repetition_rate (compat with existing)
    """
    sents = split_sentences(story)
    toks = re.findall(r"[a-z']+", story.lower())
    
    # Sentence-level repetition
    sent_counts = Counter(sents)
    repeated_sentences = sum(c - 1 for c in sent_counts.values() if c > 1)
    
    # Bigram repetition
    bigrams = list(zip(toks, toks[1:])) if len(toks) > 1 else []
    bigram_counts = Counter(bigrams)
    repeated_bigrams = sum(c - 1 for c in bigram_counts.values() if c > 1)
    
    # Trigram repetition (existing distinct3)
    trigrams = list(zip(toks, toks[1:], toks[2:])) if len(toks) > 2 else []
    trigram_counts = Counter(trigrams)
    repeated_trigrams = sum(c - 1 for c in trigram_counts.values() if c > 1)
    
    total_trigrams = len(trigrams)
    repetition_rate = repeated_trigrams / total_trigrams if total_trigrams > 0 else 0.0
    distinct3 = 1.0 - repetition_rate if total_trigrams > 0 else 1.0
    
    return {
        "repeated_sentences": repeated_sentences,
        "repeated_bigrams": repeated_bigrams,
        "repeated_trigrams": repeated_trigrams,
        "repetition_rate": round(repetition_rate, 4),
        "distinct3": round(distinct3, 4),
    }


def continuity_score(image_contexts: list, story: str) -> dict:
    """
    Sequence coherence/continuity metric for multi-image stories.
    Compares adjacent image contexts and story content.
    Returns dict with:
      - entity_consistency: fraction of recurring entities mentioned in story
      - transition_markers: count of temporal/transition words
      - adjacent_similarity: placeholder for CLIP-based adjacent frame consistency
    """
    if not image_contexts or len(image_contexts) < 2:
        return {
            "entity_consistency": 1.0,
            "transition_markers": 0,
            "adjacent_similarity": 1.0,
        }
    
    # Extract key entities from each image context (characters + objects)
    context_entities = []
    for ctx in image_contexts:
        entities = set()
        # Characters
        for c in ctx.get("characters", []):
            entities.add(c.lower())
        # Objects (simple ones only)
        for o in ctx.get("objects", []):
            if len(o.split()) <= 2:  # prefer OD-style labels
                entities.add(o.lower())
        # Region descriptions - extract nouns
        for r in ctx.get("region_descriptions", [])[:3]:
            nouns = re.findall(r"\b(?:building|person|man|woman|girl|boy|child|car|tree|flower|pig|bird|chair|table|bowl|lantern|stool|window|house|street|city|forest|restaurant|kitchen|balcony|town)\b", r.lower())
            entities.update(nouns)
        context_entities.append(entities)
    
    # Find recurring entities (appear in 2+ contexts)
    all_entities = set()
    for e_set in context_entities:
        all_entities.update(e_set)
    
    recurring = set()
    for ent in all_entities:
        count = sum(1 for e_set in context_entities if ent in e_set)
        if count >= 2:
            recurring.add(ent)
    
    # Check how many recurring entities appear in the story
    story_words = set(re.findall(r"[a-z]+", story.lower()))
    mentioned_recurring = sum(1 for ent in recurring if ent in story_words)
    entity_consistency = mentioned_recurring / len(recurring) if recurring else 1.0
    
    # Transition markers
    transition_words = {"then", "next", "after", "later", "suddenly", "meanwhile", "as", "when", "while",
                        "before", "following", "subsequently", "continues", "continues to", "moves to",
                        "goes to", "walks to", "runs to", "arrives", "reaches", "enters", "leaves"}
    story_lower = story.lower()
    transition_count = sum(1 for tw in transition_words if tw in story_lower)
    
    # Adjacent similarity (simplified: entity overlap between adjacent contexts)
    adjacent_overlaps = []
    for i in range(len(context_entities) - 1):
        overlap = len(context_entities[i] & context_entities[i + 1])
        union = len(context_entities[i] | context_entities[i + 1])
        adjacent_overlaps.append(overlap / union if union > 0 else 1.0)
    adjacent_similarity = sum(adjacent_overlaps) / len(adjacent_overlaps) if adjacent_overlaps else 1.0
    
    return {
        "entity_consistency": round(entity_consistency, 4),
        "transition_markers": transition_count,
        "adjacent_similarity": round(adjacent_similarity, 4),
    }


def score(image_path, caption, story, image_contexts: list = None):
    """
    Score a single image-story pair.
    If image_contexts provided (list of all contexts in sequence), also computes continuity.
    """
    t0 = time.perf_counter()
    sents = split_sentences(story) or [story]
    wc = len(story.split())
    image = Image.open(image_path).convert("RGB")
    with Timer("clip"):
        s = _clip_sims(image, sents + [caption])
    sent_sims, cap_sim = s[:-1], float(s[-1])
    with Timer("nli"):
        c = _contradiction_probs(caption, sents)
    bad = attribute_conflict(caption, story)
    clip_mean = float(sent_sims.mean())
    clip_n = min(max((clip_mean - 0.15) / 0.15, 0.0), 1.0)
    g = 0.4 * clip_n + 0.4 * (1 - float(c.mean())) + 0.2 * (0 if bad else 1)
    ok_len = 80 <= wc <= 120
    
    # Repetition metrics
    rep = repetition_score(story)
    
    # Continuity metrics (if contexts provided)
    cont = continuity_score(image_contexts or [], story)
    
    result = {
        "word_count": wc, "length_valid": ok_len,
        "truncated": not story.strip().rstrip('"\'').endswith((".", "!", "?")),
        "clip_image_story_mean": clip_mean, "clip_image_story_min": float(sent_sims.min()),
        "clip_image_caption": cap_sim,
        "nli_contra_mean": float(c.mean()), "nli_contra_max": float(c.max()),
        "attribute_conflict": bad, "grounding_score": g,
        "grounding_pass": bool(g >= THRESHOLD and not bad and ok_len),
        "runtime_s": time.perf_counter() - t0,
        **rep,
        **cont,
    }
    return result


def evaluate_run(rows, out_csv, image_contexts_by_image: dict = None):
    """
    rows: list of {image, caption, story}.
    image_contexts_by_image: dict image_id -> full context dict (for continuity).
    Writes CSV, prints means and failure counts.
    """
    res = []
    for r in rows:
        img_id = os.path.basename(r["image"])
        contexts = image_contexts_by_image.get(img_id, []) if image_contexts_by_image else None
        # For multi-image, pass all contexts in sequence order
        if image_contexts_by_image:
            # Get all contexts in order
            all_contexts = list(image_contexts_by_image.values())
            m = score(r["image"], r["caption"], r["story"], all_contexts)
        else:
            m = score(r["image"], r["caption"], r["story"])
        res.append({**r, **m})
    
    for r in res: r["attribute_conflict"] = " ".join(r["attribute_conflict"])
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    
    num = ["clip_image_story_mean", "clip_image_story_min", "nli_contra_mean", "nli_contra_max", 
           "grounding_score", "repetition_rate", "entity_consistency", "transition_markers", 
           "adjacent_similarity", "runtime_s"]
    print("n =", len(res), {k: round(sum(r[k] for r in res) / len(res), 3) for k in num})
    print("fail: length", sum(not r["length_valid"] for r in res), "| attr_conflict", sum(bool(r["attribute_conflict"]) for r in res),
          "| grounding", sum(not r["grounding_pass"] for r in res))
    return res


def aggregate_results(results: list) -> dict:
    """Compute aggregate metrics across all results."""
    if not results:
        return {}
    n = len(results)
    def mean(key): return sum(r.get(key, 0) for r in results) / n
    def pct(key): return sum(bool(r.get(key, 0)) for r in results) / n * 100
    
    return {
        "num_images": n,
        "mean_grounding_score": round(mean("grounding_score"), 3),
        "mean_clip_similarity": round(mean("clip_image_story_mean"), 3),
        "mean_nli_contradiction": round(mean("nli_contra_mean"), 3),
        "mean_repetition_rate": round(mean("repetition_rate"), 4),
        "mean_entity_consistency": round(mean("entity_consistency"), 3),
        "mean_transition_markers": round(mean("transition_markers"), 1),
        "mean_adjacent_similarity": round(mean("adjacent_similarity"), 3),
        "pct_length_valid": round(pct("length_valid"), 1),
        "pct_grounding_pass": round(pct("grounding_pass"), 1),
        "mean_seeing_time": round(mean("see_s") if any("see_s" in r for r in results) else 0, 2),
        "mean_story_time": round(mean("story_s") if any("story_s" in r for r in results) else 0, 2),
        "mean_total_time": round(mean("see_s") + mean("story_s") if any("see_s" in r for r in results) else mean("runtime_s"), 2),
    }


if __name__ == "__main__":  # self-test with SYNTHETIC strings (test only)
    img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selftest_image.jpg")  # ginger cat in a blue bag
    print("Image:", img, "| load_models s:", round(TIMINGS["load_models"][0], 2))
    cap = "a brown cat sitting inside a blue bag"
    good = ("The little brown cat curled up inside the blue bag and watched the room with wide amber eyes. "
            "Every time footsteps passed, her ears twitched and she peeked out from the plastic folds. "
            "The bag crinkled softly whenever she shifted her paws, and the sound made her tilt her head. "
            "It was warm and safe in there, her favorite hiding place in the whole house. "
            "Soon the afternoon sun slid across the floor, and the cat settled down to nap, "
            "purring quietly inside her cozy blue shelter, content and completely unbothered.")
    bad = ("The old oak cabinet creaked in the dark kitchen of the grandmother's farmhouse. "
           "Inside it she kept a black iron key, hidden under red velvet cloth, and nobody had touched it for years. "
           "A storm roared over the mountain while the wolves howled outside the castle gates. "
           "She lit a candle and read the ancient map by its flickering light. "
           "Nobody knew that the treasure lay beneath the frozen lake, waiting for the brave sailor "
           "who dared to cross the stormy sea at night, alone and without any fear.")
    rows = [{"image": img, "caption": cap, "story": t} for t in (good, bad)]
    for name, r in zip(("consistent", "contradicting"), evaluate_run(rows, os.path.join(os.path.dirname(__file__), "selftest.csv"))):
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("image", "caption", "story")})
    
    # Test repetition
    print("\nRepetition test:")
    rep_good = repetition_score(good)
    rep_bad = repetition_score(bad)
    print("good:", rep_good)
    print("bad:", rep_bad)
    
    # Test continuity
    print("\nContinuity test:")
    ctx1 = {"characters": ["cat"], "objects": ["bag", "floor"], "region_descriptions": ["cat in bag"]}
    ctx2 = {"characters": ["cat"], "objects": ["bag", "sun"], "region_descriptions": ["cat sleeping"]}
    cont = continuity_score([ctx1, ctx2], good)
    print("continuity:", cont)