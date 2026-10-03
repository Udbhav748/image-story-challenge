"""Grounding metric for Image -> Story (Module 10: evaluation + runtime logging).

grounding_score = 0.4*clip_n + 0.4*(1 - nli_contra_mean) + 0.2*(1 - conflict)
  clip_n    = clip((clip_image_story_mean - 0.15) / (0.30 - 0.15), 0, 1)
              (CLIP ViT-B/32 cosine for matching text is ~0.25-0.32, unrelated ~0.10-0.18)
  nli_contra_mean = mean P(contradiction | premise=caption, hypothesis=story sentence)
  conflict  = 1 if a color/material word in the story contradicts the caption, else 0
grounding_pass = grounding_score >= 0.60 AND no attribute conflict AND length_valid.
"""
import os, re, time, csv, functools
from contextlib import ContextDecorator
os.environ.setdefault("HF_HOME", r"D:\AI-Models\huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
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

def score(image_path, caption, story):
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
    toks = re.findall(r"[a-z']+", story.lower())
    tri = list(zip(toks, toks[1:], toks[2:]))
    return {"word_count": wc, "length_valid": ok_len,
            # reported only, not part of grounding_score: story cut off mid-sentence, and repetition (1.0 = no repeated trigrams)
            "truncated": not story.strip().rstrip('"\'').endswith((".", "!", "?")),
            "distinct3": len(set(tri)) / len(tri) if tri else 1.0,
            "clip_image_story_mean": clip_mean, "clip_image_story_min": float(sent_sims.min()),
            "clip_image_caption": cap_sim,
            "nli_contra_mean": float(c.mean()), "nli_contra_max": float(c.max()),
            "attribute_conflict": bad, "grounding_score": g,
            "grounding_pass": bool(g >= THRESHOLD and not bad and ok_len),
            "runtime_s": time.perf_counter() - t0}

def evaluate_run(rows, out_csv):
    """rows: list of {image, caption, story}. Writes CSV, prints means and failure counts."""
    res = [{**r, **score(r["image"], r["caption"], r["story"])} for r in rows]
    for r in res: r["attribute_conflict"] = " ".join(r["attribute_conflict"])
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    num = ["clip_image_story_mean", "clip_image_story_min", "nli_contra_mean", "nli_contra_max", "grounding_score", "runtime_s"]
    print("n =", len(res), {k: round(sum(r[k] for r in res) / len(res), 3) for k in num})
    print("fail: length", sum(not r["length_valid"] for r in res), "| attr_conflict", sum(bool(r["attribute_conflict"]) for r in res),
          "| grounding", sum(not r["grounding_pass"] for r in res))
    return res

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
