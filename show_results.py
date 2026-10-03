"""Print captions and stories from results.csv, grouped by image (read them next to the images)."""
import pandas as pd, textwrap

d = pd.read_csv("results.csv")
for img, g in d.groupby("image"):
    print("=" * 100)
    print(img)
    for _, r in g.iterrows():
        print(f"\n[{r.variant}] grounding={r.grounding_score:.3f} words={r.word_count}")
        print("INPUT TO STORY MODEL:", textwrap.shorten(str(r.context), 230))
        print("STORY:", textwrap.shorten(str(r.story), 420))
