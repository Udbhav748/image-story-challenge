"""Export results.csv to stories.json: one JSON object per image, baseline and improved side by side."""
import json
import pandas as pd

d = pd.read_csv("results.csv")
out = []
for img, g in d.groupby("image"):
    obj = {"image": img}
    for _, r in g.iterrows():
        key = "baseline" if r.variant.startswith("A") else "improved"
        obj[key] = {
            "description_given_to_story_model": r.context,
            "story": r.story,
            "word_count": int(r.word_count),
            "length_valid": bool(r.length_valid),
            "grounding_score": round(float(r.grounding_score), 3),
            "grounding_pass": bool(r.grounding_pass),
            "clip_image_story_mean": round(float(r.clip_image_story_mean), 3),
            "nli_contradiction_mean": round(float(r.nli_contra_mean), 3),
            "seeing_seconds": float(r.see_s),
            "story_seconds": float(r.story_s),
        }
    out.append(obj)

with open("stories.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"wrote stories.json with {len(out)} objects")
