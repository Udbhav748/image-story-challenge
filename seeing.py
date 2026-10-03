"""Seeing stage: Florence-2-base -> rich structured description. Offline, CPU, float32."""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", r"D:\AI-Models\huggingface")

import re
import time
import torch
from PIL import Image

MODEL_ID = "florence-community/Florence-2-base"  # native-format conversion of microsoft/Florence-2-base
NUM_BEAMS = 1  # greedy: beam search tripled CPU time
_STATE = {}
LOAD_TIME_S = None
LOAD_MODE = None


def _load():
    """Lazy, once. Native transformers class first, trust_remote_code fallback."""
    global LOAD_TIME_S, LOAD_MODE
    if "model" in _STATE:
        return _STATE["model"], _STATE["proc"]
    from transformers import AutoProcessor
    t0 = time.time()
    try:
        from transformers import Florence2ForConditionalGeneration
        proc = AutoProcessor.from_pretrained(MODEL_ID)
        model = Florence2ForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        LOAD_MODE = "native"
    except Exception as e:  # original checkpoint is in the remote-code layout
        # Offline caveat: remote code is read from the cached snapshot folder (no network),
        # but it may be copied into HF_HOME/modules on first use.
        from transformers import AutoModelForCausalLM, AutoConfig
        cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        for c in (cfg, getattr(cfg, "text_config", None)):  # transformers-5 compat shims for old remote code
            if c is not None:
                for k in ("forced_bos_token_id", "forced_eos_token_id"):
                    if not hasattr(c, k):
                        setattr(c, k, None)
        proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, config=cfg, trust_remote_code=True, torch_dtype=torch.float32)
        LOAD_MODE = "remote_code (native failed: %s)" % str(e)[:120]
    model.eval()
    _STATE["model"], _STATE["proc"] = model, proc
    LOAD_TIME_S = round(time.time() - t0, 2)
    return model, proc


def _run(model, proc, image, task, max_new_tokens=256):
    inputs = proc(text=task, images=image, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"].to(torch.float32),
            max_new_tokens=max_new_tokens, num_beams=NUM_BEAMS, do_sample=False,
            use_cache=True,
        )
    text = proc.batch_decode(ids, skip_special_tokens=False)[0]
    return text, proc.post_process_generation(text, task=task, image_size=image.size)


def _clean(s):
    return re.sub(r"\s+", " ", re.sub(r"</?s>|<pad>", "", s or "")).strip()


def describe(image_path):
    t0 = time.time()
    model, proc = _load()
    t1 = time.time()
    image = Image.open(image_path).convert("RGB")
    out = {}
    # CPU time budget: detailed caption + object detection only (<OCR> returned junk on anime frames,
    # <CAPTION> is redundant with the detailed caption).
    for key, task, mnt in [("detailed_caption", "<MORE_DETAILED_CAPTION>", 160),
                           ("od", "<OD>", 96)]:
        _, parsed = _run(model, proc, image, task, mnt)
        out[key] = parsed.get(task) if isinstance(parsed, dict) else parsed
    labels, seen = [], set()
    for l in (out["od"] or {}).get("labels", []):
        l = _clean(l).lower()
        if l and l not in seen:
            seen.add(l)
            labels.append(l)
    return {
        "short_caption": "",
        "detailed_caption": _clean(out["detailed_caption"]),
        "objects": labels,
        "ocr_text": "",
        "runtime_s": round(time.time() - t1, 2),
        "model_load_s": round(t1 - t0, 2),
    }


def to_llm_context(desc, max_words=120):
    parts = [desc.get("detailed_caption") or desc.get("short_caption", "")]
    if desc.get("objects"):
        parts.append("Objects visible: " + ", ".join(desc["objects"]) + ".")
    if desc.get("ocr_text"):
        parts.append("Text in image: " + desc["ocr_text"])
    words = " ".join(parts).split()
    if len(words) > max_words:
        words = words[:max_words]
    return " ".join(words)


if __name__ == "__main__":
    import sys, json
    for p in sys.argv[1:]:
        d = describe(p)
        print(p)
        print(json.dumps(d, indent=1))
        print("CONTEXT:", to_llm_context(d))
    print("load_mode:", LOAD_MODE, "load_s:", LOAD_TIME_S)
