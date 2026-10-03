# Image -> Story: baseline vs Florence-2 seeing stage

Everything runs locally and offline on CPU. No hosted APIs.

    image -> BLIP caption (one sentence) -> Qwen2.5-0.5B -> story      (A, baseline)
    image -> Florence-2 description (detailed caption + objects) -> same Qwen -> story   (B, improved)

Both stories are scored by the same metric. See `experiments.md` for results, failures and limitations.

## Files

| File | Purpose |
|---|---|
| `baseline.py` | BLIP + Qwen2.5-0.5B pipeline (a stand-in for the instructor's baseline) |
| `seeing.py` | The single change: Florence-2 description of the image |
| `metric.py` | Grounding metric (CLIP + NLI + attribute check) with runtime logging; `python metric.py` runs a self-test |
| `compare.py` | Runs A and B on a folder of images, scores both, writes `results.csv` |
| `export_json.py` | Turns `results.csv` into `stories.json` (one object per image) |
| `show_results.py` | Prints captions and stories for reading next to the images |
| `download_models.py` | Downloads all models once |
| `experiments.md` | Experiment write-up |

## Reproduce

    pip install -r requirements.txt
    python download_models.py            # needs internet, once; weights are a few GB

Then, offline (PowerShell shown; use `export` on macOS/Linux):

    $env:HF_HUB_OFFLINE = "1"
    python compare.py "D:\Downloads\images"     # about 2 minutes per image on CPU
    python export_json.py
    python show_results.py

If your Hugging Face cache is not the default, set `HF_HOME` before running (the code defaults it to
`D:\AI-Models\huggingface` if unset; change that path in `metric.py`, `seeing.py` and `baseline.py`
for another machine).

Python 3.13.9, torch 2.13.0, transformers 5.15.0, CPU only. Greedy decoding with fixed seed, so reruns should
reproduce the same stories on the same machine; runtimes will vary.
