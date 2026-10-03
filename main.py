"""Main entry point: Image -> Story pipeline with baseline and improved modes.
Usage:
  python main.py images/                    # improved (Florence-2) on all images
  python main.py images/ --baseline         # baseline (BLIP) on all images
  python main.py images/ --both             # both pipelines, comparison
  python main.py img1.jpg img2.jpg          # specific images
  python main.py images/ --multi            # multi-image story (sequence)
"""
import os, sys, time, json, csv, glob, argparse
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", r"D:\AI-Models\huggingface")

import torch
from PIL import Image

import baseline
import seeing
import context_builder
import metric


def find_images(paths):
    """Expand paths to list of image files."""
    images = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                images.extend(sorted(path.glob(ext)))
        elif path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            images.append(path)
    return [str(p) for p in images]


def run_baseline(image_path, fix_length=False):
    """Run baseline pipeline: BLIP caption -> Qwen story."""
    try:
        return baseline.run(image_path, fix_length)
    except Exception as e:
        return {"image": str(image_path), "error": f"baseline failed: {e}"}


def run_improved(image_path, fix_length=False):
    """Run improved pipeline: Florence-2 structured -> context -> Qwen story."""
    try:
        t0 = time.time()
        desc = seeing.describe(image_path)
        see_s = time.time() - t0
        
        ctx = context_builder.build_single_image_context(desc)
        t1 = time.time()
        story = baseline.write_story(ctx, fix_length)
        story_s = time.time() - t1
        
        return {
            "image": str(image_path),
            "caption": ctx,  # context used as caption for metric
            "story": story,
            "word_count": len(story.split()),
            "length_valid": 80 <= len(story.split()) <= 120,
            "caption_s": round(see_s, 2),
            "story_s": round(story_s, 2),
            "vision_json": desc,  # full structured output
        }
    except Exception as e:
        return {"image": str(image_path), "error": f"improved failed: {e}"}


def run_multi_image_story(image_paths, fix_length=False):
    """Run multi-image story generation with continuity."""
    try:
        descs = []
        total_see_s = 0
        for p in image_paths:
            t0 = time.time()
            desc = seeing.describe(p)
            total_see_s += time.time() - t0
            descs.append(desc)
        
        seq_ctx = context_builder.build_sequence_context(descs)
        prompt = context_builder.build_story_prompt(seq_ctx, len(descs), target_words=100*len(descs))
        
        t1 = time.time()
        tok, model = baseline._qwen()
        msgs = [{"role": "system", "content": "You are a creative storyteller."},
                {"role": "user", "content": prompt}]
        prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt_text, return_tensors="pt")
        torch.manual_seed(baseline.SEED)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=400, do_sample=False, repetition_penalty=1.05)
        story = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        # Apply length fitting if requested
        if fix_length:
            story = baseline._fit_length(story, hi=120*len(image_paths))
        
        story_s = time.time() - t1
        
        return {
            "images": [os.path.basename(p) for p in image_paths],
            "story": story,
            "word_count": len(story.split()),
            "see_s": round(total_see_s, 2),
            "story_s": round(story_s, 2),
            "vision_json": {d["image_id"]: {k: v for k, v in d.items() if k not in ["runtime_s", "model_load_s"]} for d in descs},
        }
    except Exception as e:
        return {"images": [os.path.basename(p) for p in image_paths], "error": f"multi failed: {e}"}


def run_comparison(image_paths, fix_length=False, output_csv="results.csv"):
    """Run both pipelines on all images and evaluate."""
    all_rows = []
    all_vision = {}
    
    for p in image_paths:
        name = os.path.basename(p)
        print(f"\nProcessing {name}...")
        
        # Baseline
        a = run_baseline(p, fix_length)
        if "error" in a:
            print(f"  Baseline error: {a['error']}")
            continue
        
        # Improved
        desc = seeing.describe(p)
        all_vision[name] = desc
        ctx = context_builder.build_single_image_context(desc)
        
        t = time.time()
        story_b = baseline.write_story(ctx, fix_length)
        story_s = time.time() - t
        
        for variant, cap, story, see_s, st_s in [
            ("A_baseline", a["caption"], a["story"], a["caption_s"], a["story_s"]),
            ("B_improved", ctx, story_b, desc["runtime_s"], story_s),
        ]:
            m = metric.score(p, cap, story)
            row = {
                "image": name, "variant": variant, "context": cap, "story": story,
                "see_s": round(see_s, 2), "story_s": round(st_s, 2), **m
            }
            all_rows.append(row)
            print(f"  {variant}: grounding={m['grounding_score']:.3f} pass={m['grounding_pass']} words={m['word_count']} rep={m['repetition_rate']:.3f}")
    
    # Write results CSV
    if all_rows:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nWrote {output_csv} with {len(all_rows)} rows")
    
    # Write vision JSON
    vision_out = output_csv.replace(".csv", "_vision.json")
    context_builder.save_vision_json(list(all_vision.values()), vision_out)
    print(f"Wrote {vision_out}")
    
    # Print summary
    for v in ("A_baseline", "B_improved"):
        r = [x for x in all_rows if x["variant"] == v]
        if r:
            n = len(r)
            print(f"\n{v}: n={n}")
            print(f"  mean grounding={sum(x['grounding_score'] for x in r)/n:.3f}")
            print(f"  grounding_pass={sum(bool(x['grounding_pass']) for x in r)}/{n}")
            print(f"  length_valid={sum(bool(x['length_valid']) for x in r)}/{n}")
            print(f"  mean repetition_rate={sum(x['repetition_rate'] for x in r)/n:.4f}")
            print(f"  mean entity_consistency={sum(x['entity_consistency'] for x in r)/n:.3f}")
            print(f"  mean see_s={sum(x['see_s'] for x in r)/n:.1f} mean story_s={sum(x['story_s'] for x in r)/n:.1f}")
    
    return all_rows, all_vision


def main():
    parser = argparse.ArgumentParser(description="Image -> Story pipeline")
    parser.add_argument("paths", nargs="+", help="Image files or directories")
    parser.add_argument("--baseline", action="store_true", help="Run baseline (BLIP) only")
    parser.add_argument("--improved", action="store_true", help="Run improved (Florence-2) only")
    parser.add_argument("--both", action="store_true", help="Run both and compare (default)")
    parser.add_argument("--multi", action="store_true", help="Generate single story across all images")
    parser.add_argument("--fix-length", action="store_true", help="Enforce 80-120 word limit with retries")
    parser.add_argument("--output", default="results.csv", help="Output CSV file")
    parser.add_argument("--vision-json", default="vision.json", help="Output vision JSON file")
    args = parser.parse_args()
    
    # Default to --both if neither specified
    if not args.baseline and not args.improved and not args.both and not args.multi:
        args.both = True
    
    images = find_images(args.paths)
    if not images:
        print("No images found")
        return
    
    print(f"Found {len(images)} image(s)")
    print(f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}")
    
    if args.multi:
        result = run_multi_image_story(images, args.fix_length)
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        print(f"\nMulti-image story ({result['word_count']} words):")
        print(result["story"])
        # Save
        with open("combined_story.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("\nSaved combined_story.json")
        return
    
    if args.baseline:
        for p in images:
            r = run_baseline(p, args.fix_length)
            if "error" in r:
                print(f"Error: {r['error']}")
            else:
                print(json.dumps(r, indent=1))
        return
    
    if args.improved:
        for p in images:
            r = run_improved(p, args.fix_length)
            if "error" in r:
                print(f"Error: {r['error']}")
            else:
                print(json.dumps({k: v for k, v in r.items() if k != "vision_json"}, indent=1))
        return
    
    # Default: comparison
    run_comparison(images, args.fix_length, args.output)


if __name__ == "__main__":
    main()