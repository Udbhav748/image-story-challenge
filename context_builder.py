"""Context builder: deterministic construction of LLM prompt from structured vision output."""
import json
from typing import List, Dict, Any, Optional


def build_single_image_context(desc: Dict[str, Any], max_words: int = 180) -> str:
    """
    Build context for a single image from structured vision output.
    Deterministic, no randomness.
    """
    parts = []
    
    # Scene
    if desc.get("scene"):
        parts.append(f"Scene: {desc['scene']}")
    
    # Characters
    chars = desc.get("characters", [])
    if chars:
        parts.append(f"Characters: {', '.join(chars)}")
    
    # Objects (prioritize OD labels over region descriptions)
    objects = desc.get("objects", [])
    od_objects = [o for o in objects if o in {"building", "house", "window", "person", "footwear", 
                                                "human face", "pig", "bowl", "chair", "lantern", 
                                                "stool", "car", "tree", "flower", "bird"}]
    region_objects = [o for o in objects if o not in od_objects]
    obj_list = od_objects + region_objects[:5]  # limit region descriptions
    if obj_list:
        parts.append(f"Objects: {', '.join(obj_list)}")
    
    # Actions
    actions = desc.get("actions", [])
    if actions:
        parts.append(f"Actions: {', '.join(actions)}")
    
    # Spatial relations
    spatial = desc.get("spatial_relations", [])
    if spatial:
        parts.append(f"Spatial: {', '.join(spatial[:3])}")
    
    # Region descriptions (most informative, limit to top 3)
    regions = desc.get("region_descriptions", [])
    if regions:
        # Deduplicate region descriptions
        seen = set()
        unique_regions = []
        for r in regions:
            if r not in seen:
                seen.add(r)
                unique_regions.append(r)
        for r in unique_regions[:3]:
            parts.append(f"Region: {r}")
    
    # Style/mood
    if desc.get("style_or_mood"):
        parts.append(f"Style: {desc['style_or_mood']}")
    
    # OCR text
    if desc.get("ocr_text"):
        parts.append(f"Text: {desc['ocr_text']}")
    
    context = ". ".join(parts) + "."
    
    # Trim to max_words
    words = context.split()
    if len(words) > max_words:
        context = " ".join(words[:max_words]) + "..."
    
    return context


def build_sequence_context(descs: List[Dict[str, Any]], max_words_per_image: int = 120) -> str:
    """
    Build sequence-level context for multiple images.
    Maintains order, emphasizes continuity.
    """
    if not descs:
        return ""
    
    parts = ["SEQUENCE OF IMAGES (in order):"]
    
    # Track recurring entities across frames
    all_characters = []
    all_objects = []
    all_locations = []
    
    for i, desc in enumerate(descs):
        img_id = desc.get("image_id", f"image_{i+1}")
        parts.append(f"\n--- IMAGE {i+1} ({img_id}) ---")
        
        # Scene/location
        scene = desc.get("scene", "")
        if scene:
            parts.append(f"Location: {scene}")
            # Track location keywords
            for kw in ["street", "city", "car", "forest", "restaurant", "kitchen", "balcony", "town", "room"]:
                if kw in scene.lower() and kw not in all_locations:
                    all_locations.append(kw)
        
        # Characters
        chars = desc.get("characters", [])
        if chars:
            parts.append(f"Characters: {', '.join(chars)}")
            for c in chars:
                if c not in all_characters:
                    all_characters.append(c)
        
        # Objects
        objects = desc.get("objects", [])
        od_objects = [o for o in objects if len(o.split()) <= 2]  # prefer simple OD labels
        if od_objects:
            parts.append(f"Objects: {', '.join(od_objects[:8])}")
            for o in od_objects:
                if o not in all_objects:
                    all_objects.append(o)
        
        # Actions
        actions = desc.get("actions", [])
        if actions:
            parts.append(f"Actions: {', '.join(actions)}")
        
        # Key region descriptions (max 2 per image)
        regions = desc.get("region_descriptions", [])
        seen = set()
        unique_regions = []
        for r in regions:
            if r not in seen:
                seen.add(r)
                unique_regions.append(r)
        for r in unique_regions[:2]:
            parts.append(f"Detail: {r}")
        
        # Style/mood
        if desc.get("style_or_mood"):
            parts.append(f"Mood: {desc['style_or_mood']}")
    
    # Add continuity hints
    if len(descs) > 1:
        parts.append("\n--- CONTINUITY NOTES ---")
        if all_characters:
            parts.append(f"Recurring characters: {', '.join(all_characters)}")
        if all_locations:
            parts.append(f"Recurring locations: {', '.join(all_locations)}")
        if all_objects:
            parts.append(f"Recurring objects: {', '.join(all_objects[:10])}")
        parts.append("Maintain character identity and location consistency across frames.")
        parts.append("Connect events naturally; do not reset the story at each image.")
    
    context = " ".join(parts)
    
    # Trim if needed
    words = context.split()
    max_total = max_words_per_image * len(descs)
    if len(words) > max_total:
        context = " ".join(words[:max_total]) + "..."
    
    return context


def build_story_prompt(context: str, num_images: int = 1, target_words: int = 100) -> str:
    """Build the prompt for the story generation model."""
    if num_images == 1:
        return (
            f"Write a short story of {target_words-20} to {target_words+20} words based only on this description of an image:\n"
            f"{context}\n"
            f"Return only the story."
        )
    else:
        return (
            f"Here are {num_images} consecutive images from a sequence.\n"
            f"{context}\n"
            f"Write ONE continuous story of about {target_words*num_images} words that follows these images in order, "
            f"using only what the descriptions say. Maintain character identity and location consistency. "
            f"Connect events naturally between frames. Do not describe each image separately. Return only the story."
        )


def save_vision_json(descs: List[Dict[str, Any]], output_path: str) -> None:
    """Save structured vision output for all images as JSON."""
    output = {}
    for desc in descs:
        img_id = desc.get("image_id", "unknown")
        # Save only the structured fields, not runtime metadata
        output[img_id] = {
            "scene": desc.get("scene", ""),
            "description": desc.get("description", ""),
            "objects": desc.get("objects", []),
            "characters": desc.get("characters", []),
            "actions": desc.get("actions", []),
            "relationships": desc.get("relationships", []),
            "ocr_text": desc.get("ocr_text", ""),
            "spatial_relations": desc.get("spatial_relations", []),
            "style_or_mood": desc.get("style_or_mood", ""),
            "region_descriptions": desc.get("region_descriptions", []),
        }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    from seeing import describe
    
    if len(sys.argv) < 2:
        print("Usage: python context_builder.py <image1> [image2 ...]")
        sys.exit(1)
    
    descs = [describe(p) for p in sys.argv[1:]]
    
    print("=== Single Image Contexts ===")
    for d in descs:
        ctx = build_single_image_context(d)
        print(f"\n--- {d['image_id']} ---")
        print(ctx)
    
    if len(descs) > 1:
        print("\n=== Sequence Context ===")
        seq_ctx = build_sequence_context(descs)
        print(seq_ctx)
        
        print("\n=== Story Prompt (multi) ===")
        prompt = build_story_prompt(seq_ctx, len(descs))
        print(prompt[:500] + "...")