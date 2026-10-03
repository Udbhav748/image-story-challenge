"""Before/after comparison on the same images, same metric, offline.

A (baseline): image -> BLIP one sentence -> Qwen story
B (improved): image -> Florence-2 detailed description -> same Qwen story
Both stories are scored by metric.score; runtime is recorded per stage.
Usage: HF_HUB_OFFLINE=1 python compare.py "D:\\Downloads\\images" [output.csv]   (default output: results.csv)
"""
import os, sys, time, glob, csv
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import baseline, seeing, metric

FIX = "--fix-length" in sys.argv  # same length/cut-off fix applied to BOTH pipelines (shared story stage)
sys.argv = [a for a in sys.argv if a != "--fix-length"]
folder = sys.argv[1] if len(sys.argv) > 1 else r"D:\Downloads\images"
paths = sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith((".png", ".jpg", ".jpeg")))
rows = []
for p in paths:
    name = os.path.basename(p)
    a = baseline.run(p, FIX)                              # baseline path
    d = seeing.describe(p)                                # improved seeing stage
    ctx = seeing.to_llm_context(d)
    t = time.time(); story_b = baseline.write_story(ctx, FIX); story_s = time.time() - t
    for variant, cap, story, see_s, st_s in [
        ("A_baseline", a["caption"], a["story"], a["caption_s"], a["story_s"]),
        ("B_improved", ctx, story_b, d["runtime_s"], story_s),
    ]:
        m = metric.score(p, cap, story)
        rows.append({"image": name, "variant": variant, "context": cap, "story": story,
                     "see_s": round(see_s, 2), "story_s": round(st_s, 2), **m})
        print(f"{name:24s} {variant}  grounding={m['grounding_score']:.3f} pass={m['grounding_pass']} words={m['word_count']}", flush=True)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[2] if len(sys.argv) > 2 else ("results_fixlen.csv" if FIX else "results.csv"))
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

for v in ("A_baseline", "B_improved"):
    r = [x for x in rows if x["variant"] == v]
    n = len(r)
    print(f"{v}: mean grounding={sum(x['grounding_score'] for x in r)/n:.3f} "
          f"pass={sum(bool(x['grounding_pass']) for x in r)}/{n} "
          f"length_valid={sum(bool(x['length_valid']) for x in r)}/{n} "
          f"mean see_s={sum(x['see_s'] for x in r)/n:.1f} mean story_s={sum(x['story_s'] for x in r)/n:.1f}")
print("wrote", out)
