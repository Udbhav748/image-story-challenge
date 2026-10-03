"""One story across all images, in filename order, from the improved (Florence-2) per-image stories.

Usage: python combined_story.py [stories.json]   -> writes combined_story.json
The same local Qwen model writes it; nothing is hand-written. Each image contributes its first two sentences
so the prompt stays short enough for a 0.5B model.
"""
import os, sys, json, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import baseline

src = sys.argv[1] if len(sys.argv) > 1 else "stories.json"
items = json.load(open(src, encoding="utf-8"))
scenes = [" ".join(baseline._sentences(it["improved"]["story"])[:2]) for it in items]
numbered = "\n".join(f"Scene {i + 1}: {s}" for i, s in enumerate(scenes))
template = ("Here are {n} scenes from one anime, in order.\n{scenes}\n"
            "Write ONE continuous story of about 250 words that follows these scenes in order, "
            "using only what the scenes say. Return only the story.")
t = time.time()
story = baseline._generate("", template.replace("{n}", str(len(scenes))).replace("{scenes}", numbered), 400)
story = baseline._fit_length(story, hi=300)  # drop a cut-off tail
out = {"images_in_order": [it["image"] for it in items], "scenes_given_to_model": scenes,
       "story": story, "word_count": len(story.split()), "story_seconds": round(time.time() - t, 1)}
json.dump(out, open("combined_story.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(story)
print("words:", out["word_count"])
