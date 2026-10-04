#!/usr/bin/env python3
"""Main entry point for Image -> Story V2 pipeline."""
import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List

# Apply offline settings
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from PIL import Image

from image_story import (
    PipelineOrchestrator,
    PipelineConfig,
    get_pipeline_config,
    settings,
)
from image_story.domain.schemas import PipelineArtifacts


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def find_images(paths: List[str]) -> List[str]:
    """Expand files/directories to image files, sorted by file name."""
    found = {}
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            files = [path]
        else:
            files = []
        for f in files:
            found[str(f)] = f
    return [str(f) for f in sorted(found.values(), key=lambda f: f.name.lower())]


def save_artifacts(artifacts: PipelineArtifacts, output_dir: str) -> None:
    """Save pipeline artifacts to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main artifacts
    with open(os.path.join(output_dir, f"{artifacts.run_id}_artifacts.json"), "w") as f:
        json.dump(artifacts.to_dict(), f, indent=2, ensure_ascii=False)
    
    # Save story separately for easy access
    if artifacts.story_draft:
        with open(os.path.join(output_dir, f"{artifacts.run_id}_story.txt"), "w") as f:
            f.write(artifacts.story_draft.text)
    
    # Save evaluation
    if artifacts.evaluation:
        with open(os.path.join(output_dir, f"{artifacts.run_id}_evaluation.json"), "w") as f:
            json.dump(artifacts.evaluation.to_dict(), f, indent=2, ensure_ascii=False)
    
    # Save context
    if artifacts.context:
        with open(os.path.join(output_dir, f"{artifacts.run_id}_context.txt"), "w") as f:
            f.write(artifacts.context)
    
    print(f"Artifacts saved to {output_dir}")


def run_single_image(
    orchestrator: PipelineOrchestrator,
    image_path: str,
    evaluate: bool = True,
    save_dir: str | None = None,
) -> PipelineArtifacts:
    """Run pipeline on a single image."""
    print(f"\nProcessing: {Path(image_path).name}")
    t0 = time.perf_counter()
    
    artifacts = orchestrator.run_single_image(image_path, evaluate=evaluate)
    
    total_time = time.perf_counter() - t0
    print(f"  Completed in {total_time:.2f}s")
    
    if artifacts.story_draft:
        print(f"  Story ({artifacts.story_draft.word_count} words):")
        print(f"  {artifacts.story_draft.text[:200]}...")
    
    if artifacts.evaluation:
        eval_result = artifacts.evaluation
        print(f"  Grounding: {eval_result.grounding_score:.3f}")
        print(f"  CLIP: {eval_result.clip_image_story_mean:.3f}")
        print(f"  NLI: {eval_result.nli_contra_mean:.3f}")
        print(f"  Claims: {eval_result.supported_claims} supported, {eval_result.unsupported_claims} unsupported, {eval_result.contradicted_claims} contradicted")
        print(f"  Narrative quality: {eval_result.narrative_coherence:.3f}")
    
    if save_dir:
        save_artifacts(artifacts, save_dir)
    
    return artifacts


def run_multi_image(
    orchestrator: PipelineOrchestrator,
    image_paths: List[str],
    evaluate: bool = True,
    save_dir: str | None = None,
) -> PipelineArtifacts:
    """Run pipeline on multiple images as a sequence."""
    print(f"\nProcessing {len(image_paths)} images as sequence")
    t0 = time.perf_counter()
    
    artifacts = orchestrator.run_multi_image(image_paths, evaluate=evaluate)
    
    total_time = time.perf_counter() - t0
    print(f"  Completed in {total_time:.2f}s")
    
    if artifacts.story_draft:
        print(f"  Story ({artifacts.story_draft.word_count} words):")
        print(f"  {artifacts.story_draft.text[:300]}...")
    
    if artifacts.evaluation:
        eval_result = artifacts.evaluation
        print(f"  Grounding: {eval_result.grounding_score:.3f}")
        print(f"  Continuity: {eval_result.continuity_score:.3f}")
        print(f"  Narrative quality: {eval_result.narrative_coherence:.3f}")
    
    if save_dir:
        save_artifacts(artifacts, save_dir)
    
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Image -> Story V2 Pipeline")
    parser.add_argument("paths", nargs="+", help="Image files or directories")
    parser.add_argument(
        "--mode",
        choices=["fast", "standard", "full", "baseline"],
        default="standard",
        help="Pipeline mode (default: standard)"
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Generate one story across all images (sequence mode)"
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation"
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Output directory for artifacts"
    )
    parser.add_argument(
        "--config",
        help="Path to custom config YAML"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use"
    )
    return parser


def main():
    args = build_parser().parse_args()
    
    # Apply settings
    settings.device = args.device
    settings.apply_env()
    
    # Load config
    if args.config:
        from image_story.config.settings import load_config_from_yaml
        config = load_config_from_yaml(args.config)
    else:
        config = get_pipeline_config(args.mode)
    
    config.seed = args.seed
    config.device = args.device
    
    print(f"Image -> Story V2 Pipeline")
    print(f"Mode: {config.mode}")
    print(f"Device: {config.device}")
    print(f"Seed: {config.seed}")
    print(f"GroundingDINO: {config.use_grounding_dino}")
    print(f"OCR: {config.use_ocr}")
    print(f"FAISS: {config.use_faiss}")
    print(f"Creative Planner: {config.use_creative_planner}")
    print(f"Verification: {config.use_verification}")
    
    # Find images
    images = find_images(args.paths)
    if not images:
        print("No images found")
        return
    
    print(f"Found {len(images)} image(s)")
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(config)
    
    try:
        if args.multi:
            if len(images) < 2:
                print("Multi-image mode requires at least 2 images")
                return
            run_multi_image(orchestrator, images, evaluate=not args.no_eval, save_dir=args.output_dir)
        else:
            for img_path in images:
                run_single_image(orchestrator, img_path, evaluate=not args.no_eval, save_dir=args.output_dir)
    finally:
        orchestrator.cleanup()
    
    print("\nDone!")


if __name__ == "__main__":
    main()